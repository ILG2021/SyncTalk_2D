import os
import numpy as np
import argparse
from scipy.signal import savgol_filter
from tqdm import tqdm

def smooth_landmarks(dataset_dir, window_length=9, polyorder=3):
    """
    使用 Savitzky-Golay 滤波器对关键点进行时序平滑
    """
    print(f"Processing directory: {dataset_dir}")
    
    # 获取所有的 lms 文件并按顺序排列
    lms_dir = os.path.join(dataset_dir, "landmarks")
    if not os.path.exists(lms_dir):
        print(f"Skipping {dataset_dir}: 'landmarks' folder not found.")
        return

    files = os.listdir(lms_dir)
    lms_files = [f for f in files if f.endswith('.lms')]
    # 按文件名数字排序 (0.lms, 1.lms ...)
    lms_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    if len(lms_files) < window_length:
        print(f"Skipping {dataset_dir}: Too few frames ({len(lms_files)}) for smoothing.")
        return

    # 读取所有关键点到内存
    all_lms = [] # shape: (T, 106, 2)
    for f_name in lms_files:
        lms_path = os.path.join(lms_dir, f_name)
        lms = np.loadtxt(lms_path)
        all_lms.append(lms)
    
    all_lms = np.array(all_lms)
    
    # 进行平滑 (对每个关键点的 x 和 y 坐标分别在时间轴上做滤波)
    # axis 0 是时间轴
    # window_length: 窗口越大越平滑，但可能丢失快速动作细节。9~11 是经验值。
    smoothed_lms = savgol_filter(all_lms, window_length=window_length, polyorder=polyorder, axis=0)
    
    # 将平滑后的关键点写回文件
    print(f"Smoothing {len(lms_files)} frames...")
    for i, f_name in enumerate(lms_files):
        lms_path = os.path.join(lms_dir, f_name)
        # 保留原有格式 (x y)
        np.savetxt(lms_path, smoothed_lms[i], fmt='%.6f')
    
    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset directory (e.g., ./dataset/anchor_name) containing 'landmarks' folder, OR a parent directory containing multiple datasets.")
    args = parser.parse_args()
    
    root_dir = args.path
    
    # 自动判断是单个数据集还是父目录
    if os.path.exists(os.path.join(root_dir, "landmarks")):
        # 单个数据集
        smooth_landmarks(root_dir)
    else:
        # 父目录 (比如 ./dataset/may_anchor)，遍历下面的子目录
        subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for sub in subdirs:
            smooth_landmarks(sub)

if __name__ == "__main__":
    main()
