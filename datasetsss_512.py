import os
import cv2
import torch
import random
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    
    def __init__(self, img_dir, mode):
    
        self.img_path_list = []
        self.lms_path_list = []
        self.mode = mode
        
        full_body_img_dir = os.path.join(img_dir, "full_body_img")
        landmarks_dir = os.path.join(img_dir, "landmarks")

        num_imgs = len(os.listdir(full_body_img_dir))
        for i in range(num_imgs):
            img_path = os.path.join(full_body_img_dir, str(i)+".jpg")
            lms_path = os.path.join(landmarks_dir, str(i)+".lms")
            self.img_path_list.append(img_path)
            self.lms_path_list.append(lms_path)
        
        if self.mode == "wenet":
            self.audio_feats = np.load(os.path.join(img_dir, "aud_wenet.npy"))
        if self.mode == "hubert":
            self.audio_feats = np.load(os.path.join(img_dir, "aud_hu.npy"))
        if self.mode == "ave":
            self.audio_feats = np.load(os.path.join(img_dir, "aud_ave.npy"))
            
        self.audio_feats = self.audio_feats.astype(np.float32)
        print(img_dir)
        print(self.audio_feats.shape)
        print(len(self.img_path_list))
        
    def __len__(self):
        return self.audio_feats.shape[0]-1
    
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
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0)
        return auds
    
    def process_img(self, img, lms_path, img_ex, lms_path_ex):

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
        crop_img = cv2.resize(crop_img, (520, 520), cv2.INTER_AREA)
        img_real = crop_img[4:516, 4:516].copy()
        img_real_ori = img_real.copy()
        img_masked = cv2.rectangle(img_real,(8,8,496,488),(0,0,0),-1)
        
        crop_img_ex = img_ex[ymin:ymax, xmin:xmax]
        crop_img_ex = cv2.resize(crop_img_ex, (520, 520), cv2.INTER_AREA)
        img_real_ex = crop_img_ex[4:516, 4:516].copy()
        
        img_real_ori = img_real_ori.transpose(2,0,1).astype(np.float32)
        img_masked = img_masked.transpose(2,0,1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
        
        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)

        return img_concat_T, img_real_T

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]
        
        ex_int = random.randint(0, self.__len__()-1)
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_path_ex = self.lms_path_list[ex_int]
        
        img_concat_T, img_real_T = self.process_img(img, lms_path, img_ex, lms_path_ex)
        audio_feat = self.get_audio_features(self.audio_feats, idx) 
        
        if self.mode == "wenet":
            audio_feat = audio_feat.reshape(256,16,32)
        if self.mode == "hubert":
            audio_feat = audio_feat.reshape(32,32,32)
        if self.mode == "ave":
            audio_feat = audio_feat.reshape(32,16,16)
        
        return img_concat_T, img_real_T, audio_feat
