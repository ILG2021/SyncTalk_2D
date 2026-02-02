import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
def extract_audio(path, out_path, sample_rate=16000):
    
    print(f'[INFO] ===== extract audio from "{path}" to "{out_path}" =====')
    cmd = f'ffmpeg -i "{path}" -f wav -ar {sample_rate} "{out_path}" -y'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')
    
def extract_images(path):
    
    
    video_name = os.path.basename(path)
    base_dir = os.path.dirname(path)
    full_body_dir = os.path.join(base_dir, "full_body_img")
    if not os.path.exists(full_body_dir):
        os.makedirs(full_body_dir, exist_ok=True)
    
    counter = 0
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    name, ext = os.path.splitext(path)
    # 增加对 .mov 的判断，或者如果 fps 不是 25，统一转码
    if ext.lower() == '.mov' or abs(fps - 25) > 0.1:
        print(f'[INFO] Converting {ext} video to 25fps mp4 for compatibility...')
        out_25 = f"{name}_25fps.mp4"
        # 使用 libx264 确保兼容性
        cmd = f'ffmpeg -i "{path}" -vf "fps=25" -c:v libx264 -c:a aac "{out_25}" -y'
        os.system(cmd)
        path = out_25
        # 重新打开转码后的视频
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)

    if abs(fps - 25) > 0.6: 
        raise ValueError(f"Your video fps should be 25, but it is {fps}!!!")
        
    print("extracting images...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(full_body_dir, str(counter) + '.jpg'), frame)
        counter += 1


def get_audio_feature(wav_path, asr_mode="ave"):
    print(f"extracting audio feature using {asr_mode}...")
    import sys
    import subprocess

    if asr_mode == "ave":
        script_path = os.path.abspath("./data_utils/ave/test_w2l_audio.py")
    elif asr_mode == "hubert":
        script_path = os.path.abspath("./data_utils/hubert.py")
    elif asr_mode == "whisper":
        script_path = os.path.abspath("./data_utils/whisper_extract.py")
    else:
        print(f"[ERROR] Unsupported ASR mode: {asr_mode}")
        return

    wav_path = os.path.abspath(wav_path)

    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        return

    # 使用 subprocess.run 替代 os.system，它能更好地处理 Windows 的空格和引号
    try:
        subprocess.run(
            [sys.executable, script_path, "--wav_path", wav_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] get_audio_feature failed with return code {e.returncode}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
    
def get_landmark(path, landmarks_dir):
    print("detecting landmarks...")
    base_dir = os.path.dirname(path)
    full_img_dir = os.path.join(base_dir, "full_body_img")
    
    from get_landmark import Landmark
    landmark = Landmark()
    
    for img_name in tqdm(os.listdir(full_img_dir)):
        if not img_name.endswith(".jpg"):
            continue
        img_path = os.path.join(full_img_dir, img_name)
        lms_path = os.path.join(landmarks_dir, img_name.replace(".jpg", ".lms"))
        pre_landmark, x1, y1 = landmark.detect(img_path)
        with open(lms_path, "w") as f:
            for p in pre_landmark:
                x, y = p[0]+x1, p[1]+y1
                f.write(str(x))
                f.write(" ")
                f.write(str(y))
                f.write("\n")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    parser.add_argument('--asr', type=str, default='ave', choices=['ave', 'hubert', 'whisper'], help="ASR mode for audio features")
    opt = parser.parse_args()

    base_dir = os.path.dirname(opt.path)
    wav_path = os.path.join(base_dir, 'aud.wav')
    landmarks_dir = os.path.join(base_dir, 'landmarks')

    os.makedirs(landmarks_dir, exist_ok=True)
    
    extract_audio(opt.path, wav_path)
    extract_images(opt.path)
    get_landmark(opt.path, landmarks_dir)
    get_audio_feature(wav_path, opt.asr)
    
    