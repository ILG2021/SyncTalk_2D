import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def process_single_dir(dataset_dir):
    img_dir = os.path.join(dataset_dir, "full_body_img")
    lms_dir = os.path.join(dataset_dir, "landmarks")
    save_path = os.path.join(dataset_dir, "images.npy")
    
    if os.path.exists(save_path):
        print(f"[INFO] images.npy already exists in {dataset_dir}. Skipping...")
        return

    if not os.path.exists(img_dir) or not os.path.exists(lms_dir):
        print(f"[WARN] Skipping {dataset_dir}: missing full_body_img or landmarks")
        return

    # 按文件名数字排序
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')], 
                   key=lambda x: int(os.path.splitext(x)[0]))
    
    if len(files) == 0:
        return

    print(f"[INFO] Processing {len(files)} frames in {dataset_dir} -> images.npy")
    
    processed_images = []
    
    for img_name in tqdm(files):
        img_path = os.path.join(img_dir, img_name)
        lms_name = os.path.splitext(img_name)[0] + ".lms"
        lms_path = os.path.join(lms_dir, lms_name)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Failed to read {img_path}")
            continue
            
        # 读取关键点
        lms_str = []
        if not os.path.exists(lms_path):
            # 如果关键点缺失，复用上一帧或跳过，这里选择跳过但保持占位可能更好？
            # 简单起见，假设关键点完备
            print(f"[ERROR] Missing landmarks for {img_name}")
            continue

        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            lms_list = []
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)
            lms = np.array(lms_list, dtype=np.int32)
        
        # === 核心裁剪逻辑 (与原 datasetsss_328.py 保持一致) ===
        xmin = lms[1][0]
        ymin = lms[52][1]
        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width
        
        # 边界检查
        h, w = img.shape[:2]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        crop_img = img[ymin:ymax, xmin:xmax]
        # Resize 到 328x328
        crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_AREA)
        
        processed_images.append(crop_img)
    
    # 转换为 numpy 数组并保存
    # shape: (N, 328, 328, 3), dtype: uint8
    if len(processed_images) > 0:
        np_data = np.array(processed_images, dtype=np.uint8)
        np.save(save_path, np_data)
        print(f"[SUCCESS] Saved {np_data.shape} images to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset directory or parent directory")
    args = parser.parse_args()
    
    root_dir = args.path
    # 自动递归逻辑：检查是单个数据集还是包含多个子数据集
    if os.path.exists(os.path.join(root_dir, "full_body_img")):
        process_single_dir(root_dir)
    else:
        # 认为是父目录，遍历子目录
        subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for sub in subdirs:
            process_single_dir(sub)

if __name__ == "__main__":
    main()
