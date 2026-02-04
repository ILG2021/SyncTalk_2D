import os
import argparse
import subprocess
import sys

def batch_process(input_dir, asr_mode):
    # 支持的视频格式
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    
    # 获取目录下所有视频文件
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    video_files = [f for f in files if os.path.splitext(f)[1].lower() in video_extensions]
    
    if not video_files:
        print(f"[WARN] No video files found in {input_dir}")
        return

    print(f"[INFO] Found {len(video_files)} videos to process.")
    
    process_script = os.path.join("data_utils", "process.py")
    
    for video in video_files:
        video_path = os.path.join(input_dir, video)
        video_name = os.path.splitext(video)[0]
        
        # 为每个视频创建一个目录 (如果还没有的话)
        # 建议结构：./dataset/person_name/video_name/
        target_dir = os.path.join(input_dir, video_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # 将视频移动或拷贝到目标目录（process.py 需要视频在目标目录下）
        target_video_path = os.path.join(target_dir, video)
        if not os.path.exists(target_video_path):
             import shutil
             print(f"[INFO] Moving {video} to {target_dir}...")
             shutil.move(video_path, target_video_path)
        
        print(f"\n[EXEC] Processing: {video_name}...")
        try:
            # 调用原本的 process.py 进行详细处理
            subprocess.run([
                sys.executable, process_script, 
                target_video_path, 
                "--asr", asr_mode
            ], check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed to process {video_name}, skipping...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help="Directory containing multiple video files")
    parser.add_argument('--asr', type=str, default='hubert', choices=['ave', 'hubert'], help="ASR mode")
    args = parser.parse_args()
    
    batch_process(args.dir, args.asr)
