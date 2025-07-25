import sys
import os
from loguru import logger
from isaacgym import gymtorch, gymapi, gymutil
import torch
from humanoidverse.utils.torch_utils import to_torch, torch_rand_float
import numpy as np
from termcolor import colored
from collections import deque
import cv2
from datetime import datetime
from humanoidverse.envs.env_utils.terrain import Terrain
from rich.progress import Progress
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
from pathlib import Path
import time


class IsaacGym(BaseSimulator):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.simulator_config = config.simulator.config
        self.robot_config = config.robot
        self.visualize_viewer = False
        if config.save_rendering_dir is not None:
            self.save_rendering_dir = Path(config.save_rendering_dir)

    def set_headless(self, headless):
        # call super
        super().set_headless(headless)

    def setup(self):
        self.sim_params = self._parse_sim_params()
        self.sim_dt = self.sim_params.dt

        self.physics_engine = gymapi.SIM_PHYSX
        self.gym = gymapi.acquire_gym()

        sim_device_type, self.sim_device_id = gymutil.parse_device_str(str(self.sim_device))

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and self.sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        if sim is None:
            logger.error("*** Failed to create sim")
            quit()
        logger.info("Creating Sim...", "green")

        self.sim = sim

    def _parse_sim_params(self):
        # TODO: this sim params are not loaded from the config file @Jiawei
        # initialize sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / self.simulator_config.sim.fps
        sim_params.up_axis = gymapi.UP_AXIS_Z
        # sim_params.up_axis = 1  # 0 is y, 1 is z
        sim_params.gravity = gymapi.Vec3(0., 0., -9.81)
        sim_params.num_client_threads = 0

        sim_params.physx.solver_type = self.simulator_config.sim.physx.solver_type
        sim_params.physx.num_position_iterations = self.simulator_config.sim.physx.num_position_iterations
        sim_params.physx.num_velocity_iterations = self.simulator_config.sim.physx.num_velocity_iterations
        sim_params.physx.num_threads = self.simulator_config.sim.physx.num_threads
        sim_params.physx.use_gpu = True
        sim_params.physx.num_subscenes = 0
        sim_params.physx.max_gpu_contact_pairs = (
            self.config.robot.contact_pairs_multiplier * 1024 * 1024
        )
        sim_params.use_gpu_pipeline = True

        gymutil.parse_sim_config(self.simulator_config.sim, sim_params)
        return sim_params

    def setup_terrain(self, mesh_type):
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.simulator_config.terrain, self.config.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        logger.info('Creating plane terrain')
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.simulator_config.terrain.static_friction
        plane_params.dynamic_friction = self.simulator_config.terrain.dynamic_friction
        plane_params.restitution = self.simulator_config.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
        logger.info('Created plane terrain \t| static_friction: {:.2f} \t| dynamic_friction: {:.2f} \t| restitution: {:.2f}'.format(
            plane_params.static_friction, plane_params.dynamic_friction, plane_params.restitution))
        # breakpoint()
        
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.simulator_config.terrain.static_friction
        hf_params.dynamic_friction = self.simulator_config.terrain.dynamic_friction
        hf_params.restitution = self.simulator_config.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        """
        logger.info('Creating trimesh terrain')
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.simulator_config.terrain.static_friction
        tm_params.dynamic_friction = self.simulator_config.terrain.dynamic_friction
        tm_params.restitution = self.simulator_config.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        logger.info('Created trimesh terrain')

    def load_assets(self):
        asset_root = self.robot_config.asset.asset_root
        asset_file = self.robot_config.asset.urdf_file
        self.robot_asset = self._setup_robot_asset_when_env_created(asset_root, asset_file, self.robot_config.asset)
        self.num_dof, self.num_bodies, self.dof_names, self.body_names = self._setup_robot_props_when_env_created()
        
        # assert if  aligns with config
        assert self.num_dof == len(self.robot_config.dof_names), "Number of DOFs must be equal to number of actions"
        assert self.num_bodies == len(self.robot_config.body_names), "Number of bodies must be equal to number of body names"
        assert self.dof_names == self.robot_config.dof_names, "DOF names must match the config"
        assert self.body_names == self.robot_config.body_names, "Body names must match the config"

    def _setup_robot_asset_when_env_created(self, asset_root, asset_file, asset_cfg):
        asset_path = os.path.join(asset_root, asset_file)
        gym_asset_root = os.path.dirname(asset_path)
        gym_asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()

        def set_value_if_not_none(prev_value, new_value):
            return new_value if new_value is not None else prev_value

        asset_config_options = [
            "default_dof_drive_mode",
            "collapse_fixed_joints",
            "replace_cylinder_with_capsule",
            "flip_visual_attachments",
            "fix_base_link",
            "density",
            "angular_damping",
            "linear_damping",
            "max_angular_velocity",
            "max_linear_velocity",
            "armature",
            "thickness",
            "disable_gravity",
        ]
        for option in asset_config_options:
            option_value = set_value_if_not_none(
                getattr(asset_options, option), getattr(asset_cfg, option)
            )
            setattr(asset_options, option, option_value)

        self.robot_asset = self.gym.load_asset(self.sim, gym_asset_root, gym_asset_file, asset_options)
        
        if False:
            body_idx_left = self.gym.find_asset_rigid_body_index(self.robot_asset, "left_ankle_roll_link")
            body_idx_right = self.gym.find_asset_rigid_body_index(self.robot_asset, "right_ankle_roll_link")
            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_forward_dynamics_forces = True
            sensor_props.enable_constraint_solver_forces = True
            sensor_props.use_world_frame = True
            
            sensor_pose_left = gymapi.Transform(gymapi.Vec3(0.0, 0.0, -0.04))  # Adjust the position as needed
            sensor_pose_right = gymapi.Transform(gymapi.Vec3(0.0, 0.0, -0.04))  # Adjust the position as needed

            sensor_idx_left = self.gym.create_asset_force_sensor(self.robot_asset, body_idx_left, sensor_pose_left, sensor_props)
            sensor_idx_right = self.gym.create_asset_force_sensor(self.robot_asset, body_idx_right, sensor_pose_right, sensor_props)
            # breakpoint()

        
        return self.robot_asset
    
    def _setup_robot_props_when_env_created(self):
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)

        # save body names from the asset
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)

        return self.num_dof, self.num_bodies, self.dof_names, self.body_names

    def create_envs(self, num_envs, env_origins, base_init_state):
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.num_envs = num_envs
        self.env_config = self.config
        self.env_origins = env_origins
        self.base_init_state = base_init_state
        self.envs = []
        self.robot_handles = []
        self._base_com_bias = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self._ground_friction_values = torch.zeros(self.num_envs, self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)
        self.friction_coeffs = torch.ones(self.num_envs, 1, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.ones(self.num_envs, 1, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._link_mass_scale = torch.ones(self.num_envs, len(self.env_config.domain_rand.randomize_link_body_names), dtype=torch.float, device=self.device, requires_grad=False)
        # with Progress() as progress:
        #     task = progress.add_task(
        #         f"Creating {self.num_envs} environments...", total=self.num_envs
        #     )
        if True:
            for i in range(self.num_envs):
                # create env instance
                env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
                self._build_each_env(i, env_handle)
                # progress.update(task, advance=1)
        self.friction_coeffs = self.friction_coeffs.to(self.device)
        return self.envs, self.robot_handles

    def _build_each_env(self, env_id, env_ptr):
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        pos = self.env_origins[env_id].clone()
        pos[:2] += torch_rand_float(-1., 1., (2, 1), device=str(self.device)).squeeze(1)
        start_pose.p = gymapi.Vec3(*pos)

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)
        rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, env_id)
        self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)

        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)

        robot_handle = self.gym.create_actor(env_ptr, 
                                             self.robot_asset, 
                                             start_pose, 
                                             self.env_config.robot.asset.robot_type, 
                                             env_id, 
                                             self.env_config.robot.asset.self_collisions, 0)
        self._body_list = self.gym.get_actor_rigid_body_names(env_ptr, robot_handle)
        dof_props = self._process_dof_props(dof_props_asset, env_id)
        self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props)
        body_props = self.gym.get_actor_rigid_body_properties(env_ptr, robot_handle)
        # print_matrix = lambda mat: print(
        #     f"""[{mat.x.x:.6f},{mat.x.y:.6f},{mat.x.z:.6f},\n{mat.y.x:.6f},{mat.y.y:.6f},{mat.y.z:.6f},\n{mat.z.x:.6f},{mat.z.y:.6f},{mat.z.z:.6f},]""")
        # print_matrix(body_props[0].inertia); print_matrix(body_props[0].invInertia)
        # breakpoint()
        body_props = self._process_rigid_body_props(body_props, env_id)
        # breakpoint()
        self.gym.set_actor_rigid_body_properties(env_ptr, robot_handle, body_props, recomputeInertia=True)
        self.envs.append(env_ptr)
        self.robot_handles.append(robot_handle)

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.env_config.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.env_config.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            if len(props) != self.num_bodies and env_id<3:
                logger.warning("Number of rigid shapes does not match number of bodies")
                logger.warning(f"len(RigidShapeProperties): {len(props)}")
                logger.warning(f"self.num_bodies: {self.num_bodies}")
                # logger.warning("Only randomizing friction of number of bodies")

            num_available_friction_shapes = min(len(props), self.num_bodies)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

            for s in range(num_available_friction_shapes):
                props[s].friction = self.friction_coeffs[env_id]
                self._ground_friction_values[env_id, s] += self.friction_coeffs[env_id].squeeze()
                if env_id<3:
                    logger.debug(f"Friction of shape {s}: {props[s].friction} (after randomization)")
                    
        if 'randomize_restitution' in self.env_config.domain_rand and self.env_config.domain_rand.randomize_restitution:
            if env_id==0:
                self.restitution_coeffs = torch_rand_float(self.env_config.domain_rand.restitution_range[0], self.env_config.domain_rand.restitution_range[1], (self.num_envs, 1), device='cpu')
                
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
                if env_id<3:
                    logger.debug(f"Restitution of shape {s}: {props[s].restitution} (after randomization)")

        if 'debug' in self.env_config:
            if 'restitution' in self.env_config.debug:
                print("DEBUG:: restitution", self.env_config.debug.restitution)
                for s in range(len(props)):
                    props[s].restitution = self.env_config.debug.restitution
            if 'rolling_friction' in self.env_config.debug:
                print("DEBUG:: rolling friction", self.env_config.debug.rolling_friction)
                for s in range(len(props)):
                    props[s].rolling_friction = self.env_config.debug.rolling_friction
            if 'torsion_friction' in self.env_config.debug:
                print("DEBUG:: torsion friction", self.env_config.debug.torsion_friction)
                for s in range(len(props)):
                    props[s].torsion_friction = self.env_config.debug.torsion_friction
                    
            # breakpoint()
        
                    
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

            self.dof_pos_limits_termination = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)

            for i in range(len(props)):
                
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit

                self.dof_pos_limits_termination[i, 0] = m - 0.5 * r * self.env_config.termination_scales.termination_close_to_dof_pos_limit
                self.dof_pos_limits_termination[i, 1] = m + 0.5 * r * self.env_config.termination_scales.termination_close_to_dof_pos_limit
        return props
    def _process_rigid_body_props(self, props, env_id):
        def _perturb_inertia(inertia, rand_range):
            """
            Perturbs the inertia matrix while ensuring it remains symmetric and positive definite.

            Args:
                inertia (gymapi.Matrix3): Original inertia matrix.
                rand_range (tuple): Range for random perturbation.

            Returns:
                gymapi.Matrix3: Perturbed inertia matrix.
                gymapi.Matrix3: Inverse of the perturbed inertia matrix.
            """
            # Convert gymapi.Matrix3 to numpy array
            np_mat = np.array([[inertia.x.x, inertia.x.y, inertia.x.z],
                                [inertia.y.x, inertia.y.y, inertia.y.z],
                                [inertia.z.x, inertia.z.y, inertia.z.z]])
            
            # Ensure symmetry
            np_mat = (np_mat + np_mat.T) / 2
            
            try:
                # Perform Cholesky decomposition
                # M = L @ L.T
                L = np.linalg.cholesky(np_mat)
            except np.linalg.LinAlgError:
                # If decomposition fails, add a larger value to the diagonal and retry
                np_mat += np.eye(3) * 1e-8
                L = np.linalg.cholesky(np_mat)
            
            # Add random perturbation to the Cholesky factor
            perturbation = np.exp(np.random.uniform(np.log(rand_range[0]), np.log(rand_range[1]), size=L.shape))
            # perturbation = np.random.uniform(rand_range[0], rand_range[1], size=L.shape)
            L_perturbed = L * perturbation
            
            # Reconstruct the perturbed positive definite matrix
            new_inertia_matrix = np.dot(L_perturbed, L_perturbed.T)
            
            # Compute the inverse of the perturbed inertia matrix
            inverse_inertia_matrix = np.linalg.inv(new_inertia_matrix)
            
            # Convert back to gymapi.Matrix3
            perturbed_inertia = gymapi.Mat33()
            perturbed_inertia.x.x, perturbed_inertia.x.y, perturbed_inertia.x.z = new_inertia_matrix[0]
            perturbed_inertia.y.x, perturbed_inertia.y.y, perturbed_inertia.y.z = new_inertia_matrix[1]
            perturbed_inertia.z.x, perturbed_inertia.z.y, perturbed_inertia.z.z = new_inertia_matrix[2]
            
            inverse_inertia = gymapi.Mat33()
            inverse_inertia.x.x, inverse_inertia.x.y, inverse_inertia.x.z = inverse_inertia_matrix[0]
            inverse_inertia.y.x, inverse_inertia.y.y, inverse_inertia.y.z = inverse_inertia_matrix[1]
            inverse_inertia.z.x, inverse_inertia.z.y, inverse_inertia.z.z = inverse_inertia_matrix[2]
            
            return perturbed_inertia, inverse_inertia
            # return perturbed_inertia

        if env_id<3:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                logger.debug(f"Mass of body {i}: {p.mass} (before randomization)")
            logger.debug(f"Total mass {sum} (before randomization)")

        # randomize base com
        if self.env_config.domain_rand.randomize_base_com:
            if env_id<3:
                logger.debug("randomizing base com")
            try:
                torso_index = self._body_list.index("torso_link")
            except:
                torso_index = self._body_list.index("pelvis") # for fixed upper URDF we only have pelvis link
            assert torso_index != -1

            com_x_bias = np.random.uniform(self.env_config.domain_rand.base_com_range.x[0], self.env_config.domain_rand.base_com_range.x[1])
            com_y_bias = np.random.uniform(self.env_config.domain_rand.base_com_range.y[0], self.env_config.domain_rand.base_com_range.y[1])
            com_z_bias = np.random.uniform(self.env_config.domain_rand.base_com_range.z[0], self.env_config.domain_rand.base_com_range.z[1])

            self._base_com_bias[env_id, 0] += com_x_bias
            self._base_com_bias[env_id, 1] += com_y_bias
            self._base_com_bias[env_id, 2] += com_z_bias

            props[torso_index].com.x += com_x_bias
            props[torso_index].com.y += com_y_bias
            props[torso_index].com.z += com_z_bias

        # randomize link com
        if 'randomize_link_com' in self.env_config.domain_rand and self.env_config.domain_rand.randomize_link_com:
            if env_id<3:
                logger.debug("randomizing link com")
            for i, body_name in enumerate(self.env_config.domain_rand.randomize_link_body_names):
                body_index = self._body_list.index(body_name)
                assert body_index != -1

                com_x_bias = np.random.uniform(self.env_config.domain_rand.link_com_range.x[0], self.env_config.domain_rand.link_com_range.x[1])
                com_y_bias = np.random.uniform(self.env_config.domain_rand.link_com_range.y[0], self.env_config.domain_rand.link_com_range.y[1])
                com_z_bias = np.random.uniform(self.env_config.domain_rand.link_com_range.z[0], self.env_config.domain_rand.link_com_range.z[1])

                props[body_index].com.x += com_x_bias
                props[body_index].com.y += com_y_bias
                props[body_index].com.z += com_z_bias

                # self._link_com_bias[env_id, i, 0] += com_x_bias
                # self._link_com_bias[env_id, i, 1] += com_y_bias
                # self._link_com_bias[env_id, i, 2] += com_z_bias

        # randomize link mass
        if self.env_config.domain_rand.randomize_link_mass:
            if env_id<3:
                logger.debug("randomizing link mass")
            for i, body_name in enumerate(self.env_config.domain_rand.randomize_link_body_names):
                body_index = self._body_list.index(body_name)
                assert body_index != -1

                mass_scale = np.random.uniform(self.env_config.domain_rand.link_mass_range[0], self.env_config.domain_rand.link_mass_range[1])
                props[body_index].mass *= mass_scale

                self._link_mass_scale[env_id, i] *= mass_scale

        # randomize link inertia
        if 'randomize_link_inertia' in self.env_config.domain_rand and self.env_config.domain_rand.randomize_link_inertia:
            for i, body_name in enumerate(self.env_config.domain_rand.randomize_link_inertia_names):
                body_index = self._body_list.index(body_name)
                assert body_index != -1

                props[body_index].inertia, props[body_index].invInertia = _perturb_inertia(props[body_index].inertia, self.env_config.domain_rand.link_inertia_range)

        if 'heavy_upper' in self.env_config.domain_rand and self.env_config.domain_rand.heavy_upper.enable:
            logger.debug("randomizing heavy upper")
            # print("Heavy Upper! ")
            for i, body_name in enumerate(self.env_config.domain_rand.heavy_upper.body_names):
                body_index = self._body_list.index(body_name)
                assert body_index != -1

                props[body_index].mass *= self.env_config.domain_rand.heavy_upper.ratio
                for axis1 in ['x', 'y', 'z']:
                    for axis2 in ['x', 'y', 'z']:
                        ixx = getattr(getattr(props[body_index].inertia, axis1), axis2)
                        ixx *= self.env_config.domain_rand.heavy_upper.ratio
                        setattr(getattr(props[body_index].inertia, axis1), axis2, ixx)
                        ixx = getattr(getattr(props[body_index].invInertia, axis1), axis2)
                        ixx /= self.env_config.domain_rand.heavy_upper.ratio
                        setattr(getattr(props[body_index].invInertia, axis1), axis2, ixx)
                

        # randomize base mass
        if self.env_config.domain_rand.randomize_base_mass:
            rng = self.env_config.domain_rand.added_mass_range
            try:
                base_index = self._body_list.index("pelvis") # for fixed upper URDF we only have pelvis link
            except:
                base_index = self._body_list.index("torso_link")
            assert base_index != -1
            # raise Exception("index 0 is for world, 13 is for torso!")
            # raise NotImplementedError
            props[base_index].mass += np.random.uniform(rng[0], rng[1])

        if env_id<3:
            sum_mass = 0
            for i in range(len(props)):
                logger.debug(f"Mass of body {i}: {props[i].mass} (after randomization)")
                sum_mass += props[i].mass
            logger.debug(f"Total mass {sum_mass} (afters randomization)")
        return props

    def get_dof_limits_properties(self):
        # assert the isaacgym dof limits are the same as the config
        for i in range(self.num_dof):
            assert abs(self.hard_dof_pos_limits[i, 0].item() - self.robot_config.dof_pos_lower_limit_list[i]) < 1e-5, f"DOF {i} lower limit does not match"
            assert abs(self.hard_dof_pos_limits[i, 1].item() - self.robot_config.dof_pos_upper_limit_list[i]) < 1e-5, f"DOF {i} upper limit does not match"
            assert abs(self.dof_vel_limits[i].item() - self.robot_config.dof_vel_limit_list[i]) < 1e-5, f"DOF {i} velocity limit does not match"
            assert abs(self.torque_limits[i].item() - self.robot_config.dof_effort_limit_list[i]) < 1e-5, f"DOF {i} effort limit does not match"
            # assert self.dof_pos_hard_dof_pos_limitslimits[i, 1].item() == self.robot_config.dof_pos_upper_limit_list[i], f"DOF {i} upper limit does not match"
            # assert self.dof_vel_limits[i].item() == self.robot_config.dof_vel_limit_list[i], f"DOF {i} velocity limit does not match"
            # assert self.torque_limits[i].item() == self.robot_config.dof_effort_limit_list[i], f"DOF {i} effort limit does not match"

        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name):
        return self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], body_name)
    
    def prepare_sim(self):
        self.gym.prepare_sim(self.sim)
        # Refresh tensors BEFORE we acquire them https://forums.developer.nvidia.com/t/isaacgym-preview-4-actor-root-state-returns-nans-with-isaacgymenvs-style-task/223738/4
        self.refresh_sim_tensors()

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)

        # self.gym.get_actor_force_sensor_count(self.envs[0], self.robot_handles[0])
        # jacobian and mass matrix
        robot_name = self.env_config.robot.asset.robot_type
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, robot_name)
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, robot_name)

        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        self.massmatrix = gymtorch.wrap_tensor(_massmatrix)
        # self.force_sensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, -1, 6)

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        self._rigid_body_pos = self._rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        
        self.refresh_sim_tensors()

        self.all_root_states: Tensor = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self._get_num_actors_per_env()
        self.robot_root_states = self.all_root_states.view(
            self.num_envs, num_actors, actor_root_state.shape[-1]
        )[..., 0, :]
        self.base_quat = self.robot_root_states[..., 3:7] # isaacgym uses xyzw

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

    def refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def _get_num_actors_per_env(self):
        return self.all_root_states.shape[0] // self.num_envs
        # num_actors = (
        #     self.root_states.shape[0] - self.total_num_objects
        # ) // self.num_envs
        # return num_actors

    def apply_torques_at_dof(self, torques):
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))

    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        set_env_ids_int32 = set_env_ids.to(torch.int32)
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                    gymtorch.unwrap_tensor(root_states), 
                                                    gymtorch.unwrap_tensor(set_env_ids_int32), 
                                                    len(set_env_ids_int32))
        
    def apply_rigid_body_force_at_pos_tensor(self, force_tensor, pos_tensor):
        self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, 
                                                       gymtorch.unwrap_tensor(force_tensor), 
                                                       gymtorch.unwrap_tensor(pos_tensor), 
                                                       gymapi.ENV_SPACE)

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        set_env_ids_int32 = set_env_ids.to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, 
                                             gymtorch.unwrap_tensor(dof_states), 
                                             gymtorch.unwrap_tensor(set_env_ids_int32), 
                                             len(set_env_ids_int32))

    def simulate_at_each_physics_step(self):
        self.gym.simulate(self.sim)
        if self.sim_device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)

    def setup_viewer(self):
        self.enable_viewer_sync = True
        self.visualize_viewer = True
        self.viewer = self.gym.create_viewer(
            self.sim, gymapi.CameraProperties())
        # subscribe to keyboard shortcuts
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_SPACE, "Pause")
        
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_F, "render_2x")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_G, "render_0.5x")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_H, "render_realtime")
        
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_W, "forward_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_S, "backward_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_A, "left_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_D, "right_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_Q, "heading_left_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_E, "heading_right_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_Z, "zero_command"
        )

        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_P, "push_robots"
        )

        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_N, "next_task"
        # )
        self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "toggle_video_record"
            )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_SEMICOLON, "cancel_video_record"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_X, "walk_stand_toggle"
        )
        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_U, "height_up"
        # )
        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_L, "height_down"
        # )        

        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_UP, "force_left_up"
        # )
        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_DOWN, "force_left_down"
        # )

        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_LEFT, "force_right_down"
        # )
        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_RIGHT, "force_right_up"
        # )

        sim_params = self.sim_params
        if sim_params.up_axis == gymapi.UP_AXIS_Z:
            cam_pos = gymapi.Vec3(2.0, 0.0, 1.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        else:
            cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 15.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # video recording
        self.user_is_recording, self.user_recording_state_change = False, False
        self.user_recording_video_queue_size = 100000
        self.save_rendering_dir.mkdir(parents=True, exist_ok=True)
        self.user_recording_video_path = str(self.save_rendering_dir / f"{self.config.experiment_name}-%s")
    
    def render(self, sync_frame_time=True, _realtime=True):
        # check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()
            
        delete_user_viewer_recordings = False
        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "Pause" and evt.value > 0:
                pause = True
                while pause:
                    time.sleep(0.1)
                    self.gym.draw_viewer(self.viewer, self.sim, True)
                    for evt in self.gym.query_viewer_action_events(self.viewer):
                        if evt.action == "Pause" and evt.value > 0:
                            pause = False
                    if self.gym.query_viewer_has_closed(self.viewer):
                        sys.exit()

            elif evt.action == "render_2x" and evt.value > 0:
                self._target_fps *= 2
                print(f"Current target fps: {self._target_fps}")
            elif evt.action == "render_0.5x" and evt.value > 0:
                self._target_fps /= 2
                print(f"Current target fps: {self._target_fps}")
            elif evt.action == "render_realtime" and evt.value > 0:
                self._target_fps = 50
                print(f"Current target fps: {self._target_fps}")
                
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync
            elif evt.action == "forward_command" and evt.value > 0:
                self.commands[:, 0] += 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "backward_command" and evt.value > 0:
                self.commands[:, 0] -= 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "left_command" and evt.value > 0:
                self.commands[:, 1] -= 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "right_command" and evt.value > 0:
                self.commands[:, 1] += 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "heading_left_command" and evt.value > 0:
                self.commands[:, 3] -= 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "heading_right_command" and evt.value > 0:
                self.commands[:, 3] += 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "zero_command" and evt.value > 0:
                self.commands[:, :4] = 0
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "push_robots" and evt.value > 0:
                logger.info("Push Robots")
                self._push_robots(torch.arange(self.num_envs, device=self.device))
            # elif evt.action == "next_task" and evt.value > 0:
            #     self.next_task()
            elif evt.action == "toggle_video_record" and evt.value > 0:
                # https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/envs/base_interface/isaacgym.py#L179
                self.user_is_recording = not self.user_is_recording
                self.user_recording_state_change = True
            elif evt.action == "cancel_video_record" and evt.value > 0:
                # https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/envs/base_interface/isaacgym.py#L182
                self.user_is_recording = False
                self.user_recording_state_change = False
                delete_user_viewer_recordings = True
            elif evt.action == "walk_stand_toggle" and evt.value > 0:
                self.commands[:, 4] = 1 - self.commands[:, 4]
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "height_up" and evt.value > 0:
                self.commands[:, 4] += 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "height_down" and evt.value > 0:
                self.commands[:, 4] -= 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "force_left_up" and evt.value > 0:
                self.apply_force_tensor[:, self.left_hand_link_index, 2] = 1.0
                logger.info(f"Left hand force: {self.apply_force_tensor[:, self.left_hand_link_index, :]}")
            elif evt.action == "force_left_down" and evt.value > 0:
                self.apply_force_tensor[:, self.left_hand_link_index, 2] -= 1.0
                logger.info(f"Left hand force: {self.apply_force_tensor[:, self.left_hand_link_index, :]}")
            elif evt.action == "force_right_up" and evt.value > 0:
                self.apply_force_tensor[:, self.right_hand_link_index, 2] += 1.0
                logger.info(f"Right hand force: {self.apply_force_tensor[:, self.right_hand_link_index, :]}")
            elif evt.action == "force_right_down" and evt.value > 0:
                self.apply_force_tensor[:, self.right_hand_link_index, 2] -= 1.0
                logger.info(f"Right hand force: {self.apply_force_tensor[:, self.right_hand_link_index, :]}")


        # fetch results
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)

        if _realtime:
            # 初始化或更新帧计时器
            if not hasattr(self, '_last_render_time'):
                self._target_fps = 50
                self._last_render_time = time.time()
                self._frames_to_skip = 0
                self._accumulated_time = 0.0
            self._frame_interval = 1.0 / self._target_fps
            
            current_time = time.time()
            # 计算经过的时间
            elapsed = current_time - self._last_render_time
            self._last_render_time = current_time
            self._accumulated_time += elapsed

            # 计算应该渲染的帧数
            frames_to_render = int(self._accumulated_time / self._frame_interval)
            
            if frames_to_render < 1:
                # 如果时间不够一帧，等待（减速）
                time_to_wait = self._frame_interval - self._accumulated_time
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                self._accumulated_time = 0.0
                return
            else:
                # 如果时间超过一帧，处理跳帧（加速）
                self._frames_to_skip = frames_to_render - 1
                self._accumulated_time -= frames_to_render * self._frame_interval

            # 处理跳帧
            if self._frames_to_skip > 0 and self.enable_viewer_sync and sync_frame_time:
                for _ in range(self._frames_to_skip):
                    self.gym.sync_frame_time(self.sim)
                self._frames_to_skip = 0
            

        # step graphics
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if sync_frame_time:
                self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)

        if self.visualize_viewer:
        # https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/envs/base_interface/isaacgym.py#L198
            if self.user_recording_state_change:
                if self.user_is_recording:
                    self.user_recording_video_queue = deque(
                        maxlen=self.user_recording_video_queue_size
                    )

                    curr_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    self.curr_user_recording_name = (
                        self.user_recording_video_path % curr_date_time
                    )
                    self.user_recording_frame = 0
                    if not os.path.exists(self.curr_user_recording_name):
                        os.makedirs(self.curr_user_recording_name)

                    logger.info(
                        f"Started to record data into folder {self.curr_user_recording_name}"
                    )
                if not self.user_is_recording:
                    images = [
                        img
                        for img in os.listdir(self.curr_user_recording_name)
                        if img.endswith(".png")
                    ]
                    images.sort()
                    sample_frame = cv2.imread(
                        os.path.join(self.curr_user_recording_name, images[0])
                    )
                    height, width, layers = sample_frame.shape

                    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
                    video = cv2.VideoWriter(
                        str(self.curr_user_recording_name) + ".mp4",
                        fourcc,
                        50,
                        (width, height),
                    )

                    for image in images:
                        video.write(
                            cv2.imread(
                                os.path.join(self.curr_user_recording_name, image)
                            )
                        )

                    cv2.destroyAllWindows()
                    video.release()

                    delete_user_viewer_recordings = True

                    logger.info(
                        f"============ Video finished writing {self.curr_user_recording_name}.mp4 ============"
                    )
                else:
                    logger.info("============ Writing video ============")
                self.user_recording_state_change = False

            if self.user_is_recording:
                self.gym.write_viewer_image_to_file(
                    self.viewer,
                    self.curr_user_recording_name
                    + "/%04d.png" % self.user_recording_frame,
                )
                self.user_recording_frame += 1

            if delete_user_viewer_recordings:
                images = [
                    img
                    for img in os.listdir(self.curr_user_recording_name)
                    if img.endswith(".png")
                ]
                # delete all images
                for image in images:
                    os.remove(os.path.join(self.curr_user_recording_name, image))
                os.removedirs(self.curr_user_recording_name)

    def next_task(self):
        pass

    # debug visualization
    def clear_lines(self):
        self.gym.clear_lines(self.viewer)

    def draw_sphere(self, pos, radius, color, env_id, pos_id=None):
        sphere_geom_marker = gymutil.WireframeSphereGeometry(radius, 20, 20, None, color=color)
        sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
        gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose)

    def draw_line(self, start_point, end_point, color, env_id):
        gymutil.draw_line(start_point, end_point, color, self.gym, self.viewer, self.envs[env_id])