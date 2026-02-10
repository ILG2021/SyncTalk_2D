import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
from torch import optim
import random
import argparse



class Dataset(object):
    def __init__(self, dataset_dir, mode):
        self.mode = mode
        
        # 1. 加载 images.npy (mmap)
        self.npy_path = os.path.join(dataset_dir, "images.npy")
        if os.path.exists(self.npy_path):
            print(f"[INFO] SyncNet Loading cached images from {self.npy_path} (mmap)...")
            self.images = np.load(self.npy_path, mmap_mode='r')
        else:
            raise ValueError(f"images.npy NOT found in {dataset_dir}. Please run 'python data_utils/preprocess_to_npy.py {dataset_dir}' first.")
        
        # 2. 加载音频特征
        if mode=="wenet":
            audio_feats_path = os.path.join(dataset_dir, "aud_wenet.npy")
        if mode=="hubert":
            audio_feats_path = os.path.join(dataset_dir, "aud_hu.npy")
        if mode=="ave":
            audio_feats_path = os.path.join(dataset_dir, "aud_ave.npy")
            
        self.audio_feats = np.load(audio_feats_path).astype(np.float32)
        print(f"[INFO] Loaded SyncNet dataset: {len(self.images)} frames.")
        
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
    
    def process_img_from_npy(self, img):
        # 原逻辑：
        # crop_img (328x328)
        # img_real = crop_img[4:324, 4:324]
        
        img_real = img[4:324, 4:324].copy() # 320x320
        # Transpose & Normalize
        img_real_T = torch.from_numpy(img_real.transpose(2,0,1).astype(np.float32) / 255.0)
        
        return img_real_T

    def __getitem__(self, idx):
        # 极速加载
        img = self.images[idx]
        
        img_real_T = self.process_img_from_npy(img)
        audio_feat = self.get_audio_features(self.audio_feats, idx) 
        # audio_feat = self.audio_feats[idx]
        # print(audio_feat.shape)
        # asd
        
        # audio_feat = audio_feat.reshape(128,16,32)
        if self.mode=="ave":
            audio_feat = audio_feat.reshape(32,16,16)
        else:
            audio_feat = audio_feat.reshape(32,32,32)
        y = torch.ones(1).float()
        
        return img_real_T, audio_feat, y

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.residual = residual
    
    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class SyncNet_color(nn.Module):
    def __init__(self, mode):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(3, 32, kernel_size=(7, 7), stride=1, padding=3),
            Conv2d(32, 32, kernel_size=5, stride=2, padding=1),


            Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )
            
        
        p1 = 128
        p2 = (1, 2)
        if mode == "hubert":
            p1 = 32
            p2 = (2, 2)
        if mode == "ave":
            p1 = 32
            p2 = 1
        self.audio_encoder = nn.Sequential(
            Conv2d(p1, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(128, 256, kernel_size=3, stride=p2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 256, kernel_size=3, stride=2, padding=2),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, face_sequences, audio_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences) # (B, 512, 3, 3)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        
        return audio_embedding, face_embedding

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss
    
def train(save_dir, dataset_dir, mode, epochs):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    all_datasets = []
    dataset_root = dataset_dir.strip()
    
    if os.path.isdir(dataset_root):
        if os.path.exists(os.path.join(dataset_root, "full_body_img")):
            candidate_dirs = [dataset_root]
        else:
            candidate_dirs = [os.path.join(dataset_root, d) for d in os.listdir(dataset_root) 
                            if os.path.isdir(os.path.join(dataset_root, d))]
            print(f"Scanning for sub-datasets in: {dataset_root}")
    else:
        raise ValueError(f"Dataset path {dataset_root} is not a valid directory.")

    for d_dir in candidate_dirs:
        if os.path.exists(os.path.join(d_dir, "full_body_img")):
            print(f"  [FOUND] Loading sub-dataset: {d_dir}")
            all_datasets.append(Dataset(d_dir, mode=mode))
    
    if len(all_datasets) == 0:
        raise ValueError(f"No valid datasets found in {dataset_root}.")

    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset(all_datasets)

    train_data_loader = DataLoader(
        combined_dataset, batch_size=128, shuffle=True,
        num_workers=16, pin_memory=True, persistent_workers=True)
    model = SyncNet_color(mode).cuda()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=0.001)
    
    start_epoch = 0
    # 自动检测检查点
    checkpoint_path = None
    if os.path.exists(save_dir):
        checkpoints = [f for f in os.listdir(save_dir) if f.endswith('.pth') and f[:-4].isdigit()]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x[:-4]))
            latest_checkpoint = checkpoints[-1]
            checkpoint_path = os.path.join(save_dir, latest_checkpoint)
            start_epoch = int(latest_checkpoint[:-4])
            print(f"检测到检查点，从 epoch {start_epoch} 恢复训练: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path))

    best_loss = 1000000
    for epoch in range(start_epoch, epochs):
        for batch in train_data_loader:
            imgT, audioT, y = batch
            imgT = imgT.cuda()
            audioT = audioT.cuda()
            y = y.cuda()
            audio_embedding, face_embedding = model(imgT, audioT)
            loss = cosine_loss(audio_embedding, face_embedding, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), os.path.join(save_dir, str(epoch+1)+'.pth'))
            
            
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='test', type=str)
    parser.add_argument('--dataset_dir', default='./dataset/May', type=str)
    parser.add_argument('--asr', default='ave', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    opt = parser.parse_args()
    
    train(opt.save_dir, opt.dataset_dir, opt.asr, opt.epochs)