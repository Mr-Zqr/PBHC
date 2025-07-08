#!/usr/bin/env python3
"""
G1 Robot Motion Visualizer
使用matplotlib可视化G1机器人运动数据

Usage:
    python g1_motion_viewer.py motion.pkl
"""

import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def load_motion_data(pkl_path):
    """加载运动数据"""
    data = joblib.load(pkl_path)
    motion_name = list(data.keys())[0]
    motion_data = data[motion_name]
    
    print(f"Loaded motion: {motion_name}")
    print(f"Frames: {motion_data['root_trans_offset'].shape[0]}")
    print(f"FPS: {motion_data.get('fps', 30)}")
    
    return motion_data


def plot_motion_overview(motion_data, save_path=None):
    """绘制运动概览"""
    root_trans = motion_data['root_trans_offset']
    contact_mask = motion_data['contact_mask']
    N = root_trans.shape[0]
    time_steps = np.arange(N)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 根部轨迹 (XY平面)
    ax1 = axes[0, 0]
    ax1.plot(root_trans[:, 0], root_trans[:, 1], 'b-', linewidth=2)
    ax1.scatter(root_trans[0, 0], root_trans[0, 1], c='g', s=100, label='Start')
    ax1.scatter(root_trans[-1, 0], root_trans[-1, 1], c='r', s=100, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('XY Trajectory')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # 高度变化
    ax2 = axes[0, 1]
    ax2.plot(time_steps, root_trans[:, 2], 'k-', linewidth=2)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Root Height')
    ax2.grid(True)
    
    # 接触状态
    ax3 = axes[1, 0]
    ax3.plot(time_steps, contact_mask[:, 0], 'b-', label='Left Foot', linewidth=2)
    ax3.plot(time_steps, contact_mask[:, 1], 'r-', label='Right Foot', linewidth=2)
    ax3.fill_between(time_steps, 0, contact_mask[:, 0], alpha=0.3, color='blue')
    ax3.fill_between(time_steps, 0, contact_mask[:, 1], alpha=0.3, color='red')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Contact')
    ax3.set_title('Foot Contact')
    ax3.legend()
    ax3.grid(True)
    ax3.set_ylim(-0.1, 1.1)
    
    # 速度
    ax4 = axes[1, 1]
    if N > 1:
        vel = np.diff(root_trans, axis=0)
        vel_mag = np.sqrt(np.sum(vel**2, axis=1))
        ax4.plot(time_steps[1:], vel_mag, 'g-', linewidth=2)
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Velocity (m/frame)')
    ax4.set_title('Root Velocity')
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_3d_trajectory(motion_data, save_path=None):
    """绘制3D轨迹"""
    root_trans = motion_data['root_trans_offset']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹
    ax.plot(root_trans[:, 0], root_trans[:, 1], root_trans[:, 2], 'b-', linewidth=2)
    
    # 起点和终点
    ax.scatter(root_trans[0, 0], root_trans[0, 1], root_trans[0, 2], 
               c='g', s=100, label='Start')
    ax.scatter(root_trans[-1, 0], root_trans[-1, 1], root_trans[-1, 2], 
               c='r', s=100, label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Root Trajectory')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D plot to {save_path}")
    
    plt.show()


def animate_motion(motion_data, save_path=None):
    """创建运动动画"""
    root_trans = motion_data['root_trans_offset']
    contact_mask = motion_data['contact_mask']
    fps = motion_data.get('fps', 30)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # XY轨迹动画
    ax1.set_xlim(root_trans[:, 0].min() - 0.5, root_trans[:, 0].max() + 0.5)
    ax1.set_ylim(root_trans[:, 1].min() - 0.5, root_trans[:, 1].max() + 0.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('XY Trajectory')
    ax1.grid(True)
    ax1.axis('equal')
    
    line_traj, = ax1.plot([], [], 'b-', alpha=0.5)
    point_current, = ax1.plot([], [], 'ro', markersize=8)
    
    # 接触状态
    N = len(root_trans)
    ax2.set_xlim(0, N)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Contact')
    ax2.set_title('Foot Contact')
    ax2.grid(True)
    
    frames = np.arange(N)
    line_left, = ax2.plot([], [], 'b-', label='Left Foot')
    line_right, = ax2.plot([], [], 'r-', label='Right Foot')
    vline = ax2.axvline(x=0, color='k', linestyle='--', alpha=0.7)
    ax2.legend()
    
    def animate(frame):
        # 更新轨迹
        line_traj.set_data(root_trans[:frame+1, 0], root_trans[:frame+1, 1])
        point_current.set_data([root_trans[frame, 0]], [root_trans[frame, 1]])
        
        # 更新接触状态
        line_left.set_data(frames[:frame+1], contact_mask[:frame+1, 0])
        line_right.set_data(frames[:frame+1], contact_mask[:frame+1, 1])
        vline.set_xdata([frame])
        
        return line_traj, point_current, line_left, line_right, vline
    
    anim = animation.FuncAnimation(fig, animate, frames=N, interval=1000/fps, 
                                   blit=True, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps//2)
        print(f"Saved animation to {save_path}")
    
    plt.show()
    return anim


def main():
    parser = argparse.ArgumentParser(description="G1 Robot Motion Visualizer")
    parser.add_argument("pkl_file", help="PKL motion file to visualize")
    parser.add_argument("--save", help="Save plots to this prefix")
    parser.add_argument("--animate", action="store_true", help="Show animation")
    parser.add_argument("--3d", action="store_true", help="Show 3D trajectory")
    
    args = parser.parse_args()
    
    # 加载数据
    motion_data = load_motion_data(args.pkl_file)
    
    # 运动概览
    save_overview = f"{args.save}_overview.png" if args.save else None
    plot_motion_overview(motion_data, save_overview)
    
    # 3D轨迹
    if getattr(args, '3d'):
        save_3d = f"{args.save}_3d.png" if args.save else None
        plot_3d_trajectory(motion_data, save_3d)
    
    # 动画
    if args.animate:
        save_anim = f"{args.save}_animation.gif" if args.save else None
        animate_motion(motion_data, save_anim)


if __name__ == "__main__":
    main()
