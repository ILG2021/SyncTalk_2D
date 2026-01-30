import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet_328 import Model
from tqdm import tqdm
from utils import AudioEncoder, AudDataset, get_audio_features
# from unet2 import Model
# from unet_att import Model

import time

parser = argparse.ArgumentParser(description='Train',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--asr', type=str, default="ave")
parser.add_argument('--name', type=str, default="May")
parser.add_argument('--audio_path', type=str, default="demo/talk_hb.wav")
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--parsing', type=bool, default=False)
parser.add_argument('--face_restore', action='store_true', help='Use GFPGAN to enhance face/mouth clarity')
parser.add_argument('--up_scale', type=int, default=1, help='Upscale factor for face restoration')
args = parser.parse_args()

checkpoint_path = os.path.normpath(os.path.join("./checkpoint", args.name))
# 获取checkpoint_path目录下数字最大的.pth文件，按照int排序
checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
checkpoint_files.sort(key=lambda x: int(x.split(".")[0]))
checkpoint = os.path.join(checkpoint_path, checkpoint_files[-1])
print(checkpoint)

audio_filename = os.path.basename(args.audio_path).split(".")[0]
checkpoint_name = os.path.basename(checkpoint).split(".")[0]
save_path = os.path.normpath(os.path.join("./result", f"{args.name}_{audio_filename}_{checkpoint_name}.mp4"))

dataset_dir = os.path.normpath(os.path.join("./dataset", args.name))
audio_path = args.audio_path
mode = args.asr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioEncoder().to(device).eval()
ckpt = torch.load('model/checkpoints/audio_visual_encoder.pth')
model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
dataset = AudDataset(audio_path)
data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
outputs = []
for mel in data_loader:
    mel = mel.to(device)
    with torch.no_grad():
        out = model(mel)
    outputs.append(out)
outputs = torch.cat(outputs, dim=0).cpu()


# Function to smooth audio features
def smooth_audio_features(features, window_size=3):
    if window_size <= 1:
        return features
    features_np = features.numpy()
    smoothed = np.copy(features_np)
    for i in range(len(features_np)):
        start = max(0, i - window_size // 2)
        end = min(len(features_np), i + window_size // 2 + 1)
        smoothed[i] = np.mean(features_np[start:end], axis=0)
    return torch.from_numpy(smoothed)


# Apply smoothing to audio features (window_size 3 is subtle but effective)
outputs = smooth_audio_features(outputs, window_size=3)

first_frame, last_frame = outputs[:1], outputs[-1:]
audio_feats = torch.cat([first_frame.repeat(1, 1), outputs, last_frame.repeat(1, 1)],
                        dim=0).numpy()
img_dir = os.path.join(dataset_dir, "full_body_img")
lms_dir = os.path.join(dataset_dir, "landmarks")
len_img = len(os.listdir(img_dir)) - 1
exm_img = cv2.imread(os.path.join(img_dir, "0.jpg"))
h, w = exm_img.shape[:2]
if args.parsing:
    parsing_dir = os.path.join(dataset_dir, "parsing")

temp_video_path = save_path.replace(".mp4", "temp.mp4")
if mode == "hubert" or mode == "ave":
    video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
if mode == "wenet":
    video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
step_stride = 0
img_idx = 0

net = Model(6, mode).cuda()
net.load_state_dict(torch.load(checkpoint))
net.eval()

# Initialize Face Restorer (GFPGAN)
restorer = None
if args.face_restore:
    try:
        from gfpgan import GFPGANer
        # If you use CodeFormer, you can adapt this part
        # model_path should point to your GFPGAN weights, e.g., 'weights/GFPGANv1.4.pth'
        model_path = 'model/checkpoints/GFPGANv1.4.pth'
        if not os.path.exists(model_path):
            print(f"[INFO] GFPGAN weight not found at {model_path}. Downloading...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
            torch.hub.download_url_to_file(url, model_path)
            print(f"[INFO] GFPGAN downloaded to {model_path}")
        
        restorer = GFPGANer(
            model_path=model_path,
            upscale=args.up_scale,
            arch='clean',
            channel_multiplier=2,
            device=device
        )
        print("[INFO] Face Restorer (GFPGAN) loaded successfully.")
    except ImportError:
        print("[ERROR] GFPGAN not found. Please install it via 'pip install gfpgan'.")
        args.face_restore = False


# Function to smooth coordinates
def smooth_coordinates(coords, window_size=5):
    coords = np.array(coords, dtype=np.float32)
    smoothed = np.copy(coords)
    for i in range(len(coords)):
        start = max(0, i - window_size // 2)
        end = min(len(coords), i + window_size // 2 + 1)
        smoothed[i] = np.mean(coords[start:end], axis=0)
    return smoothed.astype(np.int32)


print(f"[INFO] Pre-processing landmarks from {lms_dir} for smoothing...")
all_lms_files = sorted([f for f in os.listdir(lms_dir) if f.endswith('.lms')],
                       key=lambda x: int(x.split('.')[0]))
total_lms = len(all_lms_files)
raw_coords = []
for f_name in all_lms_files:
    lms_path = os.path.join(lms_dir, f_name)
    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.int32)
    xmin = lms[1][0]
    ymin = lms[52][1]
    xmax = lms[31][0]
    width = xmax - xmin
    ymax = ymin + width
    raw_coords.append([xmin, ymin, xmax, ymax])

# Apply smoothing
smoothed_coords_list = smooth_coordinates(raw_coords, window_size=5)
# Map frame index to smoothed coordinates
smoothed_coords = {int(all_lms_files[i].split('.')[0]): smoothed_coords_list[i] for i in range(total_lms)}

for i in tqdm(range(audio_feats.shape[0])):
    if img_idx > len_img - 1:
        step_stride = -1
    if img_idx < 1:
        step_stride = 1
    img_idx += step_stride

    current_frame_idx = img_idx + args.start_frame
    img_path = os.path.join(img_dir, f"{current_frame_idx}.jpg")

    if args.parsing:  # 读取语义分割图,[0, 0, 255]的区域使用ori img,不用pred的结果
        parsing_path = os.path.join(parsing_dir, f"{current_frame_idx}.png")
        parsing = cv2.imread(parsing_path)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: could not read image at {img_path}")
        continue

    # Use smoothed coordinates
    if current_frame_idx in smoothed_coords:
        xmin, ymin, xmax, ymax = smoothed_coords[current_frame_idx]
    else:
        # Fallback if frame index is missing (should not happen with sorted list)
        lms_path = os.path.join(lms_dir, f"{current_frame_idx}.lms")
        lms_list = []
        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)
        lms = np.array(lms_list, dtype=np.int32)
        xmin = lms[1][0]
        ymin = lms[52][1]
        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width

    crop_img = img[ymin:ymax, xmin:xmax]
    crop_img_par = crop_img.copy()
    if args.parsing:  # 读取语义分割图,[0, 0, 255]的区域使用ori img,不用pred的结果
        crop_parsing_img = parsing[ymin:ymax, xmin:xmax]
        # crop_parsing_img = cv2.resize(crop_parsing_img, (328, 328), cv2.INTER_AREA)
    h_crop, w_crop = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
    crop_img_ori = crop_img.copy()
    img_real_ex = crop_img[4:324, 4:324].copy()
    img_real_ex_ori = img_real_ex.copy()
    # if args.parsing:
    # img_real_ex_ori_ori = img_real_ex.copy()
    img_masked = cv2.rectangle(img_real_ex_ori, (5, 5, 310, 305), (0, 0, 0), -1)
    img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
    img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)

    img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
    img_masked_T = torch.from_numpy(img_masked / 255.0)
    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]

    audio_feat = get_audio_features(audio_feats, i)
    if mode == "hubert":
        audio_feat = audio_feat.reshape(32, 32, 32)
    if mode == "wenet":
        audio_feat = audio_feat.reshape(256, 16, 32)
    if mode == "ave":
        audio_feat = audio_feat.reshape(32, 16, 16)
    audio_feat = audio_feat[None]
    audio_feat = audio_feat.cuda()
    img_concat_T = img_concat_T.cuda()

    with torch.no_grad():
        pred = net(img_concat_T, audio_feat)[0]

    pred = pred.cpu().numpy().transpose(1, 2, 0) * 255
    pred = np.array(pred, dtype=np.uint8)
    
    # Apply Face Restoration (Super-Resolution)
    if args.face_restore and restorer is not None:
        # We restore the crop_img_ori after putting pred back into it
        # or restore pred directly. Restoring the whole crop_img_ori (328x328) is better.
        crop_img_ori[4:324, 4:324] = pred
        # The input to GFPGANer.enhance is (img, has_aligned, only_center_face, paste_back)
        _, _, restored_face = restorer.enhance(crop_img_ori, has_aligned=False, only_center_face=True, paste_back=True)
        crop_img_ori = restored_face
    else:
        crop_img_ori[4:324, 4:324] = pred
    
    crop_img_ori = cv2.resize(crop_img_ori, (w_crop, h_crop), interpolation=cv2.INTER_CUBIC)
    if args.parsing:  # 读取语义分割图,[0, 0, 255]和[255, 255, 255]的区域使用ori img,不用pred的结果
        parsing_mask = (crop_parsing_img == [0, 0, 255]).all(axis=2) | (crop_parsing_img == [255, 255, 255]).all(axis=2)
        crop_img_ori[parsing_mask] = crop_img_par[parsing_mask]
    img[ymin:ymax, xmin:xmax] = crop_img_ori
    # y_gap = lms[16][1] - lms[52][1] + h//10
    # print(y_gap, h, h//10, width)
    # crop_img_ori = crop_img_ori[:y_gap,:]
    # cv2.imwrite(f"./temp/{i}.jpg", crop_img_ori)
    # img[ymin:ymin+y_gap, xmin:xmax] = crop_img_ori
    video_writer.write(img)
video_writer.release()

# Quote paths for ffmpeg to handle spaces
ffmpeg_cmd = f'ffmpeg -i "{temp_video_path}" -i "{audio_path}" -c:v libx264 -c:a aac -crf 20 "{save_path}" -y'
os.system(ffmpeg_cmd)

if os.path.exists(temp_video_path):
    os.remove(temp_video_path)
print(f"[INFO] ===== save video to {save_path} =====")
