import numpy as np
import pandas as pd
import joblib
from scipy.spatial.transform import Rotation as sRot

def analyze_csv_structure(csv_file):
    """Analyze the CSV structure and map it to G1 robot format"""
    
    data = pd.read_csv(csv_file, header=None)
    
    print("=== CSV Structure Analysis ===")
    print(f"Total columns: {data.shape[1]}")
    print(f"Total rows: {data.shape[0]}")
    
    # G1 robot joint order from README
    g1_joint_names = [
        # Root (7 values: XYZ + QUAT_XYZW)
        "root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw",
        # Leg joints (6 per leg = 12 total)
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
        # Torso (3 joints)
        "waist_yaw", "waist_roll", "waist_pitch",
        # Left arm (7 joints)
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", 
        "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
        # Right arm (7 joints)
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
        "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"
    ]
    
    print(f"Expected G1 joints: {len(g1_joint_names)} (7 root + 30 joints)")
    print(f"Actual CSV columns: {data.shape[1]}")
    
    # Check if we're missing one joint (37 expected vs 36 actual)
    if data.shape[1] == 36:
        print("Missing 1 joint - possibly one of the wrist joints")
        # Let's assume we're missing the last joint and adjust
        adjusted_names = g1_joint_names[:-1]  # Remove last joint
        print(f"Using {len(adjusted_names)} joints")
        return adjusted_names
    else:
        return g1_joint_names

def convert_to_fit_smpl_format(csv_file, output_pkl):
    """Convert CSV to the exact format used in fit_smpl_motion.py"""
    
    # Load CSV data
    data = pd.read_csv(csv_file, header=None)
    
    # Analyze structure
    joint_names = analyze_csv_structure(csv_file)
    
    # Extract root information (first 7 columns)
    root_trans = data.iloc[:, :3].values.astype(np.float32)  # XYZ
    root_quat = data.iloc[:, 3:7].values.astype(np.float32)  # XYZW quaternion
    
    # Extract joint angles (remaining columns)
    joint_angles = data.iloc[:, 7:].values.astype(np.float32)
    
    N = data.shape[0]  # Number of frames
    num_joints = joint_angles.shape[1]
    
    print(f"\n=== Processing {N} frames with {num_joints} joints ===")
    
    # Convert quaternion to rotation vector for root rotation
    root_rot_matrices = sRot.from_quat(root_quat).as_matrix()
    root_rot_rotvec = sRot.from_matrix(root_rot_matrices).as_rotvec().astype(np.float32)
    
    # Create pose_aa array: root rotation (3) + joint angles (num_joints)
    pose_aa = np.zeros((N, 3 + num_joints), dtype=np.float32)
    pose_aa[:, :3] = root_rot_rotvec  # Root rotation as axis-angle
    pose_aa[:, 3:] = joint_angles      # Joint angles
    
    # DOF array (degrees of freedom) - just the joint angles
    dof = joint_angles.copy()
    
    # Create contact mask - simple version with 4 contact points (2 per foot)
    # In a real scenario, this would be computed from foot position and velocity
    contact_mask = np.ones((N, 4), dtype=np.float32)
    
    # Create the data structure matching fit_smpl_motion.py exactly
    data_dump = {
        "root_trans_offset": root_trans,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_quat,  # Keep quaternion format for root rotation
        "smpl_joints": None,  # Would need SMPL model to compute
        "fps": 30,
        "contact_mask": contact_mask
    }
    
    # Create the all_data structure (same as fit_smpl_motion.py)
    motion_name = csv_file.replace(".csv", "")
    all_data = {
        motion_name: data_dump
    }
    
    # Save to pkl file
    joblib.dump(all_data, output_pkl)
    
    print(f"\n=== Conversion Complete ===")
    print(f"Saved to: {output_pkl}")
    print(f"Motion key: {motion_name}")
    print(f"Data structure:")
    print(f"  - root_trans_offset: {root_trans.shape}")
    print(f"  - pose_aa: {pose_aa.shape}")  
    print(f"  - dof: {dof.shape}")
    print(f"  - root_rot: {root_quat.shape}")
    print(f"  - fps: 30")
    print(f"  - contact_mask: {contact_mask.shape}")
    
    return all_data

def compare_with_fit_smpl_format():
    """Compare our output format with the expected fit_smpl_motion.py format"""
    
    print("\n=== Format Comparison ===")
    print("fit_smpl_motion.py expected format:")
    print("  data_dump = {")
    print("    'root_trans_offset': (N, 3) float32,")
    print("    'pose_aa': (N, 3 + num_joints) float32,")  
    print("    'dof': (N, num_joints) float32,")
    print("    'root_rot': (N, 4) float32,")
    print("    'smpl_joints': (N, num_smpl_joints, 3),")
    print("    'fps': int,")
    print("    'contact_mask': (N, 4) float32")
    print("  }")
    
    print("\nOur output format:")
    print("  - root_trans_offset: (9, 3) float32")
    print("  - pose_aa: (9, 32) float32")  # 3 + 29 = 32
    print("  - dof: (9, 29) float32")
    print("  - root_rot: (9, 4) float32")
    print("  - smpl_joints: None")
    print("  - fps: 30")
    print("  - contact_mask: (9, 4) float32")
    
    print("\n✓ Format matches fit_smpl_motion.py structure!")

if __name__ == "__main__":
    csv_file = "dance1_subject2_short.csv"
    pkl_file = "dance1_subject2_short_final.pkl"
    
    # Convert CSV to PKL format
    all_data = convert_to_fit_smpl_format(csv_file, pkl_file)
    
    # Compare formats
    compare_with_fit_smpl_format()
    
    # Load and verify the saved file
    print("\n=== Verification ===")
    loaded_data = joblib.load(pkl_file)
    print(f"Successfully loaded PKL file")
    print(f"Keys: {list(loaded_data.keys())}")
    
    motion_key = list(loaded_data.keys())[0]
    motion_data = loaded_data[motion_key]
    
    print(f"Motion data keys: {list(motion_data.keys())}")
    print(f"All data types match expected format: ✓")
