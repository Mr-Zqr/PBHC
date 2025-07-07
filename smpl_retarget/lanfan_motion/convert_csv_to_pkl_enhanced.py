import numpy as np
import pandas as pd
import joblib
from scipy.spatial.transform import Rotation as sRot
import torch

def load_and_inspect_pkl(pkl_path):
    """Load and inspect the generated pkl file"""
    data = joblib.load(pkl_path)
    
    print("=== PKL File Structure ===")
    print(f"Top level keys: {list(data.keys())}")
    
    for key, value in data.items():
        print(f"\n=== Data for key: {key} ===")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_value is not None:
                    if isinstance(sub_value, np.ndarray):
                        print(f"  {sub_key}: {sub_value.shape} {sub_value.dtype}")
                        if sub_value.ndim <= 2 and sub_value.size <= 20:
                            print(f"    Sample: {sub_value[:2] if sub_value.ndim > 1 else sub_value}")
                    else:
                        print(f"  {sub_key}: {type(sub_value)} = {sub_value}")
                else:
                    print(f"  {sub_key}: None")
    
    return data

def create_enhanced_conversion_script():
    """
    Create an enhanced conversion script with better G1 robot format handling
    """
    
    # Read the original CSV to understand the structure
    csv_data = pd.read_csv("dance1_subject2_short.csv", header=None)
    print("=== Original CSV Structure ===")
    print(f"CSV shape: {csv_data.shape}")
    print(f"Total columns: {csv_data.shape[1]}")
    
    # According to README, G1 format should be:
    # root_joint(XYZQXQYQZQW) - 7 values + 30 joint values = 37 total
    # But our CSV has different number of columns
    
    print(f"Expected for G1: 7 (root) + 30 (joints) = 37 columns")
    print(f"Actual CSV columns: {csv_data.shape[1]}")
    
    # Let's check the first few values
    print("\n=== First row values ===")
    first_row = csv_data.iloc[0].values
    print(f"First 7 values (root): {first_row[:7]}")
    print(f"Remaining values (joints): {first_row[7:]}")
    print(f"Number of joint values: {len(first_row[7:])}")
    
    return csv_data

def convert_g1_csv_to_pkl_enhanced(csv_file_path, output_pkl_path):
    """
    Enhanced conversion function with proper G1 robot format handling
    """
    
    # Read CSV data
    data = pd.read_csv(csv_file_path, header=None)
    
    # Extract root information (first 7 columns: XYZ + QUAT_XYZW)
    root_trans = data.iloc[:, :3].values.astype(np.float32)  # XYZ position
    root_quat = data.iloc[:, 3:7].values.astype(np.float32)  # Quaternion XYZW
    
    # Extract joint angles (remaining columns)
    joint_angles = data.iloc[:, 7:].values.astype(np.float32)
    
    # Get number of frames
    N = data.shape[0]
    num_joints = joint_angles.shape[1]
    
    print(f"Processing {N} frames with {num_joints} joints")
    
    # Convert quaternion to rotation vector for root rotation
    # Note: The quaternion format in CSV might be XYZW, but scipy expects XYZW
    root_rot_matrices = sRot.from_quat(root_quat).as_matrix()
    root_rot_rotvec = sRot.from_matrix(root_rot_matrices).as_rotvec().astype(np.float32)
    
    # Create pose_aa array matching the format in fit_smpl_motion.py
    # This should match the robot's kinematic structure
    pose_aa = np.zeros((N, 3 + num_joints), dtype=np.float32)
    pose_aa[:, :3] = root_rot_rotvec  # Root rotation as rotation vector
    pose_aa[:, 3:] = joint_angles      # Joint angles
    
    # DOF array (degrees of freedom) - joint angles only
    dof = joint_angles.copy()
    
    # Create a simple contact mask
    # For G1, we would typically have 4 contact points (2 per foot)
    # Setting all to 1.0 initially (feet on ground)
    contact_mask = np.ones((N, 4), dtype=np.float32)
    
    # Create the data structure matching fit_smpl_motion.py output format
    data_dump = {
        "root_trans_offset": root_trans,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_quat,  # Keep quaternion format
        "smpl_joints": None,  # Would need SMPL model computation
        "fps": 30,
        "contact_mask": contact_mask
    }
    
    # Create the all_data structure with proper key naming
    motion_name = csv_file_path.split("/")[-1].replace(".csv", "")
    all_data = {
        motion_name: data_dump
    }
    
    # Save to pkl file
    joblib.dump(all_data, output_pkl_path)
    
    print(f"\n=== Conversion Summary ===")
    print(f"Input: {csv_file_path}")
    print(f"Output: {output_pkl_path}")
    print(f"Motion name: {motion_name}")
    print(f"Frames: {N}")
    print(f"Joints: {num_joints}")
    print(f"Root translation shape: {root_trans.shape}")
    print(f"Root quaternion shape: {root_quat.shape}")
    print(f"Pose AA shape: {pose_aa.shape}")
    print(f"DOF shape: {dof.shape}")
    print(f"Contact mask shape: {contact_mask.shape}")
    
    return all_data

if __name__ == "__main__":
    # First, inspect the CSV structure
    csv_structure = create_enhanced_conversion_script()
    
    # Convert the CSV file with enhanced function
    csv_file = "dance1_subject2_short.csv"
    pkl_file = "dance1_subject2_short_enhanced.pkl"
    
    all_data = convert_g1_csv_to_pkl_enhanced(csv_file, pkl_file)
    
    # Load and inspect the result
    print("\n" + "="*50)
    print("INSPECTING GENERATED PKL FILE")
    print("="*50)
    loaded_data = load_and_inspect_pkl(pkl_file)
