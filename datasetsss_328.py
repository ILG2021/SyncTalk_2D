import os
import cv2
import torch
import random
import numpy as np
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    
    def __init__(self, img_dir, mode):
        self.mode = mode
        
        # 1. 尝试加载预处理好的 images.npy (mmap 模式)
        self.npy_path = os.path.join(img_dir, "images.npy")
        if os.path.exists(self.npy_path):
            print(f"[INFO] Loading cached images from {self.npy_path} (mmap)...")
            self.images = np.load(self.npy_path, mmap_mode='r')
            self.use_npy = True
        else:
            raise ValueError(f"images.npy NOT found in {img_dir}. Please run 'python data_utils/preprocess_to_npy.py {img_dir}' first.")

        # 2. 加载音频特征
        if self.mode == "wenet":
            audio_path = os.path.join(img_dir, "aud_wenet.npy")
        if self.mode == "hubert":
            audio_path = os.path.join(img_dir, "aud_hu.npy")
        if self.mode == "ave":
            audio_path = os.path.join(img_dir, "aud_ave.npy")
            
        self.audio_feats = np.load(audio_path).astype(np.float32)
        print(f"[INFO] Loaded dataset: {len(self.images)} frames, Audio: {self.audio_feats.shape}")
        
    def __len__(self):
        return min(len(self.images), self.audio_feats.shape[0])
    
    def get_audio_features(self, features, index):
        left = index - 8
        right = index + 8
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        auds = torch.from_numpy(features[left:right])
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
        return auds
    
    def get_audio_features_1(self, features, index):
        left = index - 8
        pad_left = 0
        if left < 0:
            pad_left = -left
            left = 0
        auds = features[left:index]
        auds = torch.from_numpy(auds)
        if pad_left > 0:
            auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds], dim=0)
        return auds
    
    # 这里的输入 img 直接就是已经 Crop 好的 (328, 328) numpy array
    def process_img_from_npy(self, img, img_ex):
        # 原逻辑：
        # crop_img = cv2.resize(...) -> 得到 328x328
        # img_real = crop_img[4:324, 4:324] -> 得到 320x320
        # img_masked = mask(img_real)
        
        # 1. Target Frame (img)
        img_real = img[4:324, 4:324].copy()
        img_real_ori = img_real.copy()
        # 遮挡嘴部
        img_masked = cv2.rectangle(img_real, (5, 5, 310, 305), (0, 0, 0), -1)
        
        # 2. Reference Frame (img_ex)
        img_real_ex = img_ex[4:324, 4:324].copy()
        
        # 3. Transpose & Normalize
        img_real_ex_T = torch.from_numpy(img_real_ex.transpose(2, 0, 1).astype(np.float32) / 255.0)
        img_real_T = torch.from_numpy(img_real_ori.transpose(2, 0, 1).astype(np.float32) / 255.0)
        img_masked_T = torch.from_numpy(img_masked.transpose(2, 0, 1).astype(np.float32) / 255.0)
        
        # 4. Concatenate Input
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)

        return img_concat_T, img_real_T

    def __getitem__(self, idx):
        # 边界保护
        idx_next = idx + 1 if idx < self.__len__() - 1 else idx

        # 直接从 mmap array 读取，无 IO
        img = self.images[idx]
        img_next = self.images[idx_next]
        
        # 随机参考帧
        ex_int = random.randint(0, self.__len__()-1)
        img_ex = self.images[ex_int]

        # 处理当前帧
        img_concat_T, img_real_T = self.process_img_from_npy(img, img_ex)
        audio_feat = self.get_audio_features(self.audio_feats, idx) 
        
        # 处理下一帧 (参考帧不变)
        img_concat_T_next, img_real_T_next = self.process_img_from_npy(img_next, img_ex)
        audio_feat_next = self.get_audio_features(self.audio_feats, idx_next)

        # Reshape Audio Features
        if self.mode == "wenet":
            audio_feat = audio_feat.reshape(256,16,32)
            audio_feat_next = audio_feat_next.reshape(256,16,32)
        if self.mode == "hubert":
            audio_feat = audio_feat.reshape(32,32,32)
            audio_feat_next = audio_feat_next.reshape(32,32,32)
        if self.mode == "ave":
            audio_feat = audio_feat.reshape(32,16,16)
            audio_feat_next = audio_feat_next.reshape(32,16,16)
        
        return img_concat_T, img_real_T, audio_feat, img_concat_T_next, img_real_T_next, audio_feat_next
    
        