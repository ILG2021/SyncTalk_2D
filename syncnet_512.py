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
        self.img_path_list = []
        self.lms_path_list = []
        full_body_img_dir = os.path.join(dataset_dir, "full_body_img")
        landmarks_dir = os.path.join(dataset_dir, "landmarks")
        for i in range(len(os.listdir(full_body_img_dir))):
            img_path = os.path.join(full_body_img_dir, str(i) + ".jpg")
            lms_path = os.path.join(landmarks_dir, str(i) + ".lms")
            self.img_path_list.append(img_path)
            self.lms_path_list.append(lms_path)
        if mode=="wenet":
            audio_feats_path = os.path.join(dataset_dir, "aud_wenet.npy")
        if mode=="hubert":
            audio_feats_path = os.path.join(dataset_dir, "aud_hu.npy")
        if mode=="ave":
            audio_feats_path = os.path.join(dataset_dir, "aud_ave.npy")
        self.mode = mode
        self.audio_feats = np.load(audio_feats_path)
        self.audio_feats = self.audio_feats.astype(np.float32)
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
    def process_img(self, img, lms_path):
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
        img_real = img_real.transpose(2,0,1).astype(np.float32)
        img_real_T = torch.from_numpy(img_real / 255.0)
        return img_real_T
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]
        
        # Randomly choose between positive and negative samples
        # This makes the SyncNet training meaningful
        if random.random() > 0.5:
            # Positive sample: Matching audio features for the current frame
            audio_feat = self.get_audio_features(self.audio_feats, idx)
            y = torch.ones(1).float()
        else:
            # Negative sample: Random audio features from a different frame
            random_idx = random.randint(0, self.__len__() - 1)
            # Ensure it's not the same frame
            while abs(random_idx - idx) < 5:
                random_idx = random.randint(0, self.__len__() - 1)
            audio_feat = self.get_audio_features(self.audio_feats, random_idx)
            y = torch.zeros(1).float()

        img_real_T = self.process_img(img, lms_path)
        
        if self.mode=="ave":
            audio_feat = audio_feat.reshape(32,16,16)
        else:
            audio_feat = audio_feat.reshape(32,32,32)
        
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

class SyncNet_color(nn.Module):
    def __init__(self, mode):
        super(SyncNet_color, self).__init__()
        self.face_encoder = nn.Sequential(
            Conv2d(3, 32, kernel_size=(7, 7), stride=1, padding=3),
            Conv2d(32, 32, kernel_size=5, stride=2, padding=1), # 255
            Conv2d(32, 64, kernel_size=5, stride=2, padding=1), # 127
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 64
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 32
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 16
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1), # 8
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1), # 4
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=2, stride=1, padding=0), # 3
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
    def forward(self, face_sequences, audio_sequences):
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)
        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        return audio_embedding, face_embedding

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    # Cosine similarity is in [-1, 1], but BCELoss expects [0, 1].
    # Map [-1, 1] to [0, 1] and clamp to avoid numerical issues pushing value out of [0, 1].
    d = (d + 1) / 2
    d = torch.clamp(d, min=1e-7, max=1-1e-7)
    loss = logloss(d.unsqueeze(1), y)
    return loss
def train(save_dir, dataset_dir, mode):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    train_dataset = Dataset(dataset_dir, mode=mode)
    train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    model = SyncNet_color(mode).cuda()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    best_loss = 1000000
    for epoch in range(100):
        total_loss = 0
        for batch in train_data_loader:
            imgT, audioT, y = batch
            imgT = imgT.cuda()
            audioT = audioT.cuda()
            y = y.cuda()
            audio_embedding, face_embedding = model(imgT, audioT)
            loss = cosine_loss(audio_embedding, face_embedding, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, str(epoch+1)+'.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='syncnet_512_ckpt', type=str)
    parser.add_argument('--dataset_dir', default='./dataset/May', type=str)
    parser.add_argument('--asr', default='ave', type=str)
    opt = parser.parse_args()
    train(opt.save_dir, opt.dataset_dir, opt.asr)
