#!/usr/bin/env python3
"""
简单的G1机器人运动可视化启动脚本
"""

import subprocess
import sys
import os

def main():
    # 默认文件路径
    pkl_file = "dance1_subject2_enhanced.pkl"
    urdf_file = "/home/zqr/devel/PBHC/description/robots/g1/g1_23dof_lock_wrist.urdf"
    
    # 检查文件是否存在
    if not os.path.exists(pkl_file):
        print(f"Error: PKL file not found: {pkl_file}")
        print("Please run the CSV to PKL conversion first:")
        print("python csv_to_pkl_converter.py --input dance1_subject2.csv")
        return
    
    if not os.path.exists(urdf_file):
        print(f"Error: URDF file not found: {urdf_file}")
        return
    
    # 运行可视化
    print("Starting G1 robot motion visualization...")
    print(f"PKL file: {pkl_file}")
    print(f"URDF file: {urdf_file}")
    print()
    
    try:
        subprocess.run([
            sys.executable, "isaac_gym_visualizer.py",
            "--pkl_file", pkl_file,
            "--urdf_file", urdf_file
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Visualization failed: {e}")
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")

if __name__ == "__main__":
    main()
