import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet_512 import Model
from utils import AudioEncoder, AudDataset, get_audio_features
import time

parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--asr', type=str, default="ave")
parser.add_argument('--name', type=str, default="May")
parser.add_argument('--audio_path', type=str, default="demo/talk_hb.wav")
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--parsing', type=bool, default=False)
args = parser.parse_args()

checkpoint_path = os.path.join("./checkpoint", args.name)
checkpoint = os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path), key=lambda x: int(x.split(".")[0]))[-1])
print("Using:", checkpoint)

save_path = os.path.join("./result", args.name+"_"+args.audio_path.split("/")[-1].split(".")[0]+"_512.mp4")
dataset_dir = os.path.join("./dataset", args.name)
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
first_frame, last_frame = outputs[:1], outputs[-1:]
audio_feats = torch.cat([first_frame.repeat(1, 1), outputs, last_frame.repeat(1, 1)], dim=0).numpy()

img_dir = os.path.join(dataset_dir, "full_body_img/")
lms_dir = os.path.join(dataset_dir, "landmarks/")
len_img = len(os.listdir(img_dir)) - 1
exm_img = cv2.imread(img_dir+"0.jpg")
h_full, w_full = exm_img.shape[:2]

if mode=="hubert" or mode=="ave":
    video_writer = cv2.VideoWriter(save_path.replace(".mp4", "temp.mp4"), cv2.VideoWriter_fourcc('M','J','P', 'G'), 25, (w_full, h_full))
else:
    video_writer = cv2.VideoWriter(save_path.replace(".mp4", "temp.mp4"), cv2.VideoWriter_fourcc('M','J','P', 'G'), 20, (w_full, h_full))

img_idx = 0
step_stride = 1
net = Model(6, mode).cuda().eval()
net.load_state_dict(torch.load(checkpoint))

for i in tqdm(range(audio_feats.shape[0])):
    if img_idx > len_img - 1: step_stride = -1
    if img_idx < 1: step_stride = 1
    img_idx += step_stride
    
    img = cv2.imread(img_dir + str(img_idx+args.start_frame)+'.jpg')
    lms_path = lms_dir + str(img_idx+args.start_frame)+'.lms'
    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.int32)
    xmin, ymin = lms[1][0], lms[52][1]
    xmax = lms[31][0]
    width = xmax - xmin
    ymax = ymin + width
    
    crop_img = img[ymin:ymax, xmin:xmax]
    h, w = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (520, 520), interpolation=cv2.INTER_CUBIC)
    crop_img_ori = crop_img.copy()
    img_real_ex = crop_img[4:516, 4:516].copy()
    img_masked = cv2.rectangle(img_real_ex.copy(),(8,8,496,488),(0,0,0),-1)
    
    img_real_ex_T = torch.from_numpy(img_real_ex.transpose(2,0,1).astype(np.float32) / 255.0).cuda()[None]
    img_masked_T = torch.from_numpy(img_masked.transpose(2,0,1).astype(np.float32) / 255.0).cuda()[None]
    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=1)
    
    audio_feat = get_audio_features(audio_feats, i)
    if mode=="hubert": audio_feat = audio_feat.reshape(32,32,32)
    elif mode=="wenet": audio_feat = audio_feat.reshape(256,16,32)
    elif mode=="ave": audio_feat = audio_feat.reshape(32,16,16)
    audio_feat = audio_feat[None].cuda()
    
    with torch.no_grad():
        pred = net(img_concat_T, audio_feat)[0]
    
    pred = (pred.cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
    crop_img_ori[4:516, 4:516] = pred
    crop_img_ori = cv2.resize(crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC)
    img[ymin:ymax, xmin:xmax] = crop_img_ori
    video_writer.write(img)

video_writer.release()
os.system(f"ffmpeg -i {save_path.replace('.mp4', 'temp.mp4')} -i {audio_path} -c:v libx264 -c:a aac -crf 20 {save_path} -y")
if os.path.exists(save_path.replace(".mp4", "temp.mp4")): os.remove(save_path.replace(".mp4", "temp.mp4"))
print(f"[INFO] ===== save video to {save_path} =====")
