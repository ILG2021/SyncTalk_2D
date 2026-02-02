
import os
import argparse
import subprocess
import sys
import shutil

def main():
    parser = argparse.ArgumentParser(description="Batch process multiple videos for a person.")
    parser.add_argument('person_dir', type=str, help="Path to the person's dataset folder containing raw videos")
    parser.add_argument('--asr', type=str, default='whisper', choices=['ave', 'hubert', 'whisper'], help="ASR mode")
    args = parser.parse_args()

    # 1. 确保预处理根目录存在
    preprocess_root = os.path.join(args.person_dir, "preprocess")
    os.makedirs(preprocess_root, exist_ok=True)

    # 2. 识别所有视频文件
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.m4v')
    videos = [f for f in os.listdir(args.person_dir) if f.lower().endswith(video_extensions)]
    
    if not videos:
        print(f"[ERROR] No raw videos found in {args.person_dir}")
        print(f"Supported extensions: {video_extensions}")
        return

    print(f"\n{'='*60}")
    print(f"[INFO] Found {len(videos)} videos in {args.person_dir}")
    print(f"[INFO] Using ASR mode: {args.asr}")
    print(f"{'='*60}\n")

    for i, video_file in enumerate(videos):
        # 使用视频文件名作为 part 名称（去除扩展名）
        video_name = os.path.splitext(video_file)[0]
        # 兼容处理文件名中的空格或特殊字符
        part_name = video_name.replace(" ", "_").replace(".", "_")
        part_dir = os.path.join(preprocess_root, part_name)
        
        os.makedirs(part_dir, exist_ok=True)
        
        # 原始视频路径
        src_video_path = os.path.abspath(os.path.join(args.person_dir, video_file))
        # 目标视频路径（放在 preprocess 子目录下）
        dst_video_path = os.path.join(part_dir, video_file)
        
        print(f"Task [{i+1}/{len(videos)}]: Processing {video_file} -> {part_name}/")
        
        # 如果子目录没有视频，拷贝过去
        if not os.path.exists(dst_video_path):
             print(f"  - Copying video...")
             shutil.copy2(src_video_path, dst_video_path)
        
        # 调用核心处理脚本
        cmd = [
            sys.executable, 
            os.path.abspath("data_utils/process.py"), 
            os.path.abspath(dst_video_path), 
            "--asr", args.asr
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"  - [OK] {video_file} processed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"  - [FAILED] Error processing {video_file}.\n")
            continue

    print(f"{'='*60}")
    print(f"[FINISHED] All {len(videos)} videos are preprocessed into {preprocess_root}")
    print(f"Now you can run training pointing to this folder.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
