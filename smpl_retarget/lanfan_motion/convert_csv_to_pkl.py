import numpy as np
import pandas as pd
import joblib
from scipy.spatial.transform import Rotation as sRot
import torch

def convert_csv_to_pkl(csv_file_path, output_pkl_path):
    """
    Convert LAFAN1 CSV format to the same pkl format as fit_smpl_motion.py
    
    Args:
        csv_file_path: Path to the CSV file
        output_pkl_path: Path to save the output pkl file
    """
    
    # Read CSV data
    data = pd.read_csv(csv_file_path, header=None)
    
    # Extract root joint information (first 7 columns: XYZ + quaternion XYZW)
    root_trans = data.iloc[:, :3].values  # XYZ position
    root_quat = data.iloc[:, 3:7].values  # Quaternion (XYZW format)
    
    # Extract joint angles (remaining columns)
    joint_angles = data.iloc[:, 7:].values
    
    # Get number of frames
    N = data.shape[0]
    
    # Convert quaternion to rotation vector for root_rot
    root_rot_matrices = sRot.from_quat(root_quat).as_matrix()
    root_rot_rotvec = sRot.from_matrix(root_rot_matrices).as_rotvec()
    
    # For G1 robot, we need to construct the pose_aa format
    # The pose_aa should include root rotation + joint angles
    # Based on the README, G1 has 30 joints (excluding root)
    num_joints = joint_angles.shape[1]
    
    # Create pose_aa array: [root_rot(3) + joint_angles(num_joints)]
    pose_aa = np.zeros((N, 3 + num_joints))
    pose_aa[:, :3] = root_rot_rotvec  # Root rotation
    pose_aa[:, 3:] = joint_angles      # Joint angles
    
    # Create DOF array (same as joint angles for this case)
    dof = joint_angles.copy()
    
    # Create the data structure matching fit_smpl_motion.py output
    data_dump = {
        "root_trans_offset": root_trans,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_quat,  # Keep original quaternion format
        "smpl_joints": None,  # This would need SMPL data to compute
        "fps": 30,
        "contact_mask": None  # This would need foot contact detection
    }
    
    # Create the structure matching the all_data format
    all_data = {
        "dance1_subject2_short": data_dump
    }
    
    # Save to pkl file
    joblib.dump(all_data, output_pkl_path)
    print(f"Successfully converted {csv_file_path} to {output_pkl_path}")
    print(f"Data shape: {N} frames, {num_joints} joints")
    print(f"Root translation shape: {root_trans.shape}")
    print(f"Pose AA shape: {pose_aa.shape}")
    print(f"DOF shape: {dof.shape}")
    
    return all_data

def create_detailed_conversion_script():
    """
    Create a more detailed conversion script that handles the G1 robot format properly
    """
    script_content = '''
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.transform import Rotation as sRot
import torch

def convert_g1_csv_to_pkl(csv_file_path, output_pkl_path):
    """
    Convert G1 robot CSV format to pkl format matching fit_smpl_motion.py
    
    G1 joint order from README:
    - root_joint(XYZQXQYQZQW) - 7 values
    - left_hip_pitch_joint
    - left_hip_roll_joint  
    - left_hip_yaw_joint
    - left_knee_joint
    - left_ankle_pitch_joint
    - left_ankle_roll_joint
    - right_hip_pitch_joint
    - right_hip_roll_joint
    - right_hip_yaw_joint
    - right_knee_joint
    - right_ankle_pitch_joint
    - right_ankle_roll_joint
    - waist_yaw_joint
    - waist_roll_joint
    - waist_pitch_joint
    - left_shoulder_pitch_joint
    - left_shoulder_roll_joint
    - left_shoulder_yaw_joint
    - left_elbow_joint
    - left_wrist_roll_joint
    - left_wrist_pitch_joint
    - left_wrist_yaw_joint
    - right_shoulder_pitch_joint
    - right_shoulder_roll_joint
    - right_shoulder_yaw_joint
    - right_elbow_joint
    - right_wrist_roll_joint
    - right_wrist_pitch_joint
    - right_wrist_yaw_joint
    Total: 30 joints (excluding root)
    """
    
    # Read CSV data
    data = pd.read_csv(csv_file_path, header=None)
    
    # Extract root information (first 7 columns)
    root_trans = data.iloc[:, :3].values.astype(np.float32)  # XYZ position
    root_quat = data.iloc[:, 3:7].values.astype(np.float32)  # Quaternion (XYZW format)
    
    # Extract joint angles (remaining 30 columns)
    joint_angles = data.iloc[:, 7:].values.astype(np.float32)
    
    # Get number of frames
    N = data.shape[0]
    
    # Convert quaternion to rotation vector for root rotation
    root_rot_matrices = sRot.from_quat(root_quat).as_matrix()
    root_rot_rotvec = sRot.from_matrix(root_rot_matrices).as_rotvec().astype(np.float32)
    
    # Create pose_aa array matching the format in fit_smpl_motion.py
    # This includes root rotation (3) + all joint rotations
    # For G1, we need to reshape joint angles to match the expected format
    num_joints = joint_angles.shape[1]  # Should be 30 for G1
    
    # Create pose_aa in the format expected by the humanoid model
    # The exact format depends on the robot configuration
    pose_aa = np.zeros((N, 3 + num_joints), dtype=np.float32)
    pose_aa[:, :3] = root_rot_rotvec  # Root rotation as rotation vector
    pose_aa[:, 3:] = joint_angles      # Joint angles
    
    # DOF array (degrees of freedom) - same as joint angles for this format
    dof = joint_angles.copy()
    
    # Create simple contact mask (all feet on ground initially)
    # In a real implementation, this would be computed from foot positions
    contact_mask = np.ones((N, 4), dtype=np.float32)  # 4 contact points (2 per foot)
    
    # Create the data structure matching fit_smpl_motion.py output format
    data_dump = {
        "root_trans_offset": root_trans,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_quat,  # Keep quaternion format for root rotation
        "smpl_joints": None,  # Would need SMPL model to compute
        "fps": 30,
        "contact_mask": contact_mask
    }
    
    # Create the all_data structure
    all_data = {
        "dance1_subject2_short": data_dump
    }
    
    # Save to pkl file
    joblib.dump(all_data, output_pkl_path)
    
    print(f"Successfully converted {csv_file_path} to {output_pkl_path}")
    print(f"Data details:")
    print(f"  - Frames: {N}")
    print(f"  - Joints: {num_joints}")
    print(f"  - Root translation shape: {root_trans.shape}")
    print(f"  - Root quaternion shape: {root_quat.shape}")
    print(f"  - Pose AA shape: {pose_aa.shape}")
    print(f"  - DOF shape: {dof.shape}")
    print(f"  - Contact mask shape: {contact_mask.shape}")
    
    return all_data

if __name__ == "__main__":
    csv_file = "dance1_subject2_short.csv"
    pkl_file = "dance1_subject2_short.pkl"
    
    convert_g1_csv_to_pkl(csv_file, pkl_file)
'''
    
    return script_content

if __name__ == "__main__":
    # Convert the CSV file
    csv_file = "dance1_subject2_short.csv"
    pkl_file = "dance1_subject2_short.pkl"
    
    convert_csv_to_pkl(csv_file, pkl_file)
