"""
Inference script with Super Resolution enhancement.
Uses 328px model to generate video, then applies SR to the entire video.
"""
import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet_328 import Model
from utils import AudioEncoder, AudDataset, get_audio_features

# Try to import super resolution models
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    HAS_REALESRGAN = True
except ImportError:
    HAS_REALESRGAN = False
    print("[WARN] Real-ESRGAN not found. Run: pip install realesrgan")

try:
    from gfpgan import GFPGANer
    HAS_GFPGAN = True
except ImportError:
    HAS_GFPGAN = False
    print("[WARN] GFPGAN not found. Run: pip install gfpgan")


def setup_realesrgan(scale=2):
    """Setup Real-ESRGAN model."""
    if not HAS_REALESRGAN:
        return None
    
    if scale == 4:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = 'model/realesrgan/RealESRGAN_x4plus.pth'
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        model_path = 'model/realesrgan/RealESRGAN_x2plus.pth'
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
    
    if not os.path.exists(model_path):
        print(f"[INFO] Downloading Real-ESRGAN x{scale} model...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(url, model_path)
    
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True
    )
    return upsampler


def setup_gfpgan(bg_upsampler=None):
    """Setup GFPGAN model."""
    if not HAS_GFPGAN:
        return None
    
    model_path = 'model/gfpgan/GFPGANv1.4.pth'
    
    if not os.path.exists(model_path):
        print(f"[INFO] Downloading GFPGAN model...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import urllib.request
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        urllib.request.urlretrieve(url, model_path)
    
    face_enhancer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=bg_upsampler
    )
    return face_enhancer


def enhance_frame(frame, sr_model, sr_type, scale):
    """Enhance a single frame."""
    if sr_model is None:
        return frame
    
    if sr_type == 'gfpgan':
        _, _, output = sr_model.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
        if output is None:
            return cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        return output
    else:
        output, _ = sr_model.enhance(frame, outscale=scale)
        return output


def upscale_video(input_path, output_path, sr_model, sr_type, scale, audio_path):
    """Apply super resolution to entire video."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_width = width * scale
    out_height = height * scale
    
    print(f"[INFO] Upscaling video: {width}x{height} -> {out_width}x{out_height}")
    
    temp_sr_video = output_path.replace('.mp4', '_sr_temp.mp4')
    writer = cv2.VideoWriter(temp_sr_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_width, out_height))
    
    for _ in tqdm(range(total_frames), desc="Super Resolution"):
        ret, frame = cap.read()
        if not ret:
            break
        
        enhanced = enhance_frame(frame, sr_model, sr_type, scale)
        writer.write(enhanced)
    
    cap.release()
    writer.release()
    
    # Merge audio
    print("[INFO] Merging audio with upscaled video...")
    ffmpeg_cmd = f'ffmpeg -i "{temp_sr_video}" -i "{audio_path}" -c:v libx264 -c:a aac -crf 18 "{output_path}" -y'
    os.system(ffmpeg_cmd)
    
    if os.path.exists(temp_sr_video):
        os.remove(temp_sr_video)


parser = argparse.ArgumentParser(description='Inference with Super Resolution',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--asr', type=str, default="ave")
parser.add_argument('--name', type=str, default="May")
parser.add_argument('--audio_path', type=str, default="demo/talk_hb.wav")
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--parsing', type=bool, default=False)
parser.add_argument('--sr_type', type=str, default="realesrgan", choices=['realesrgan', 'gfpgan', 'none'],
                    help="Super resolution type: realesrgan, gfpgan, or none")
parser.add_argument('--sr_scale', type=int, default=2, choices=[2, 4], help="Super resolution scale factor")
parser.add_argument('--face_enhance', action='store_true', 
                    help='Use GFPGAN for face + Real-ESRGAN for background')
args = parser.parse_args()

checkpoint_path = os.path.normpath(os.path.join("./checkpoint", args.name))
checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
checkpoint_files.sort(key=lambda x: int(x.split(".")[0]))
checkpoint = os.path.join(checkpoint_path, checkpoint_files[-1])
print(f"[INFO] Using checkpoint: {checkpoint}")

audio_filename = os.path.basename(args.audio_path).split(".")[0]
checkpoint_name = os.path.basename(checkpoint).split(".")[0]
sr_suffix = f"_sr_{args.sr_type}_x{args.sr_scale}" if args.sr_type != 'none' else ""
save_path = os.path.normpath(os.path.join("./result", f"{args.name}_{audio_filename}_{checkpoint_name}{sr_suffix}.mp4"))
base_video_path = os.path.normpath(os.path.join("./result", f"{args.name}_{audio_filename}_{checkpoint_name}_base.mp4"))

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

outputs = smooth_audio_features(outputs, window_size=3)

first_frame, last_frame = outputs[:1], outputs[-1:]
audio_feats = torch.cat([first_frame.repeat(1, 1), outputs, last_frame.repeat(1, 1)], dim=0).numpy()

img_dir = os.path.join(dataset_dir, "full_body_img")
lms_dir = os.path.join(dataset_dir, "landmarks")
len_img = len(os.listdir(img_dir)) - 1
exm_img = cv2.imread(os.path.join(img_dir, "0.jpg"))
h, w = exm_img.shape[:2]

if args.parsing:
    parsing_dir = os.path.join(dataset_dir, "parsing")

temp_video_path = base_video_path.replace(".mp4", "_temp.mp4")
if mode == "hubert" or mode == "ave":
    video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
if mode == "wenet":
    video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))

step_stride = 0
img_idx = 0

net = Model(6, mode).cuda()
net.load_state_dict(torch.load(checkpoint))
net.eval()

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

smoothed_coords_list = smooth_coordinates(raw_coords, window_size=5)
smoothed_coords = {int(all_lms_files[i].split('.')[0]): smoothed_coords_list[i] for i in range(total_lms)}

# Step 1: Generate base video with 328 model
print(f"[INFO] Step 1: Generating base video with 328 model...")
for i in tqdm(range(audio_feats.shape[0]), desc="Generating frames"):
    if img_idx > len_img - 1:
        step_stride = -1
    if img_idx < 1:
        step_stride = 1
    img_idx += step_stride

    current_frame_idx = img_idx + args.start_frame
    img_path = os.path.join(img_dir, f"{current_frame_idx}.jpg")

    if args.parsing:
        parsing_path = os.path.join(parsing_dir, f"{current_frame_idx}.png")
        parsing = cv2.imread(parsing_path)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: could not read image at {img_path}")
        continue

    if current_frame_idx in smoothed_coords:
        xmin, ymin, xmax, ymax = smoothed_coords[current_frame_idx]
    else:
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
    if args.parsing:
        crop_parsing_img = parsing[ymin:ymax, xmin:xmax]
    h_crop, w_crop = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
    crop_img_ori = crop_img.copy()
    img_real_ex = crop_img[4:324, 4:324].copy()
    img_real_ex_ori = img_real_ex.copy()
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
    
    crop_img_ori[4:324, 4:324] = pred
    crop_img_ori = cv2.resize(crop_img_ori, (w_crop, h_crop), interpolation=cv2.INTER_CUBIC)
    
    if args.parsing:
        parsing_mask = (crop_parsing_img == [0, 0, 255]).all(axis=2) | (crop_parsing_img == [255, 255, 255]).all(axis=2)
        crop_img_ori[parsing_mask] = crop_img_par[parsing_mask]
    
    img[ymin:ymax, xmin:xmax] = crop_img_ori
    video_writer.write(img)

video_writer.release()

# Merge audio with base video
print("[INFO] Merging audio with base video...")
ffmpeg_cmd = f'ffmpeg -i "{temp_video_path}" -i "{audio_path}" -c:v libx264 -c:a aac -crf 20 "{base_video_path}" -y'
os.system(ffmpeg_cmd)

if os.path.exists(temp_video_path):
    os.remove(temp_video_path)

# Step 2: Apply super resolution to the entire video
if args.sr_type != 'none':
    print(f"[INFO] Step 2: Applying {args.sr_type} x{args.sr_scale} super resolution to video...")
    
    if args.face_enhance:
        # GFPGAN for face + Real-ESRGAN for background
        bg_upsampler = setup_realesrgan(scale=args.sr_scale)
        sr_model = setup_gfpgan(bg_upsampler=bg_upsampler)
        sr_type = 'gfpgan'
    elif args.sr_type == 'gfpgan':
        sr_model = setup_gfpgan()
        sr_type = 'gfpgan'
    else:
        sr_model = setup_realesrgan(scale=args.sr_scale)
        sr_type = 'realesrgan'
    
    if sr_model is not None:
        upscale_video(base_video_path, save_path, sr_model, sr_type, args.sr_scale, audio_path)
        # Clean up base video
        if os.path.exists(base_video_path):
            os.remove(base_video_path)
        print(f"[INFO] ===== Saved SR video to {save_path} =====")
    else:
        print(f"[ERROR] Failed to setup SR model. Base video saved to {base_video_path}")
else:
    # No SR, just rename base video
    os.rename(base_video_path, save_path)
    print(f"[INFO] ===== Saved video to {save_path} =====")
