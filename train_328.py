import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasetsss_328 import MyDataset
from syncnet_328 import SyncNet_color
from unet_328 import Model
import random
import torchvision.models as models


def get_args():
    parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_syncnet', action='store_true',
                        help="if use syncnet, you need to set 'syncnet_checkpoint'")
    parser.add_argument('--syncnet_checkpoint', type=str, default="")
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--save_dir', type=str, help="trained model save path.")
    parser.add_argument('--see_res', action='store_false', default=True,
                        help="Set to disable result visualization during training.")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--asr', type=str, default="hubert", choices=["hubert", "ave", "wenet", "whisper"])
    parser.add_argument('--temporal', action='store_true', default=False, help="Use temporal consistency loss.")
    parser.add_argument('--temporal_weight', type=float, default=1.0, help="Weight for temporal consistency loss.")

    return parser.parse_args()


args = get_args()
use_syncnet = args.use_syncnet


# Loss functions
class PerceptualLoss():

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


def train(net, epoch, batch_size, lr):
    content_loss = PerceptualLoss(torch.nn.MSELoss())
    if use_syncnet:
        if args.syncnet_checkpoint == "":
            raise ValueError("Using syncnet, you need to set 'syncnet_checkpoint'.Please check README")

        syncnet = SyncNet_color(args.asr).eval().cuda()
        syncnet.load_state_dict(torch.load(args.syncnet_checkpoint))
        for param in syncnet.parameters():
            param.requires_grad = False
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset_dir_list = []
    # 如果路径包含逗号，拆分为多个路径
    dirs = [d.strip() for d in args.dataset_dir.split(',')]
    for d in dirs:
        if os.path.exists(os.path.join(d, "full_body_img")):
            dataset_dir_list.append(d)
        else:
            # 扫描下一级子目录
            for sub in os.listdir(d):
                sub_path = os.path.join(d, sub)
                if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "full_body_img")):
                    dataset_dir_list.append(sub_path)
    
    print(f"[INFO] Found {len(dataset_dir_list)} datasets: {dataset_dir_list}")
    
    dataloader_list = []
    dataset_list= []
    for dataset_dir in dataset_dir_list:
        dataset = MyDataset(dataset_dir, args.asr, temporal=args.temporal)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4,
                                      pin_memory=True, persistent_workers=True)
        dataloader_list.append(train_dataloader)
        dataset_list.append(dataset)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.L1Loss()

    start_epoch = 0
    # 自动检测检查点
    if os.path.exists(save_dir):
        checkpoints = [f for f in os.listdir(save_dir) if f.endswith('.pth') and f[:-4].isdigit()]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x[:-4]))
            latest_checkpoint = checkpoints[-1]
            checkpoint_path = os.path.join(save_dir, latest_checkpoint)
            start_epoch = int(latest_checkpoint[:-4]) + 1
            print(f"检测到检查点，从 epoch {start_epoch} 恢复训练: {checkpoint_path}")
            net.load_state_dict(torch.load(checkpoint_path))

    for e in range(start_epoch, epoch):
        net.train()
        random_i = random.randint(0, len(dataset_dir_list) - 1)
        dataset = dataset_list[random_i]
        train_dataloader = dataloader_list[random_i]

        with tqdm(total=len(dataset), desc=f'Epoch {e + 1}/{epoch}', unit='img') as p:
            for batch in train_dataloader:
                if args.temporal:
                    imgs, labels, audio_feat, imgs_prev, labels_prev, audio_feat_prev = batch
                    imgs_prev = imgs_prev.cuda()
                    labels_prev = labels_prev.cuda()
                    audio_feat_prev = audio_feat_prev.cuda()
                else:
                    imgs, labels, audio_feat = batch
                
                imgs = imgs.cuda()
                labels = labels.cuda()
                audio_feat = audio_feat.cuda()
                
                preds = net(imgs, audio_feat)
                
                if args.temporal:
                    preds_prev = net(imgs_prev, audio_feat_prev)
                    # Temporal Consistency Loss: MSE of differences
                    diff_pred = preds - preds_prev
                    diff_real = labels - labels_prev
                    loss_temporal = nn.functional.mse_loss(diff_pred, diff_real)
                else:
                    loss_temporal = 0

                if use_syncnet:
                    y = torch.ones([preds.shape[0], 1]).float().cuda()
                    a, v = syncnet(preds, audio_feat)
                    sync_loss = cosine_loss(a, v, y)
                
                loss_PerceptualLoss = content_loss.get_loss(preds, labels)
                loss_pixel = criterion(preds, labels)
                
                loss = loss_pixel + loss_PerceptualLoss * 0.01
                if use_syncnet:
                    loss += 10 * sync_loss
                if args.temporal:
                    loss += args.temporal_weight * loss_temporal

                p.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                p.update(imgs.shape[0])

        if (e + 1) % 5 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, str(e) + '.pth'))
        if args.see_res:
            net.eval()
            if args.temporal:
                img_concat_T, img_real_T, audio_feat, _, _, _ = dataset.__getitem__(random.randint(0, dataset.__len__()))
            else:
                img_concat_T, img_real_T, audio_feat = dataset.__getitem__(random.randint(0, dataset.__len__()))
            img_concat_T = img_concat_T[None].cuda()
            audio_feat = audio_feat[None].cuda()
            with torch.no_grad():
                pred = net(img_concat_T, audio_feat)[0]
            pred = pred.cpu().numpy().transpose(1, 2, 0) * 255
            pred = np.array(pred, dtype=np.uint8)
            img_real = img_real_T.numpy().transpose(1, 2, 0) * 255
            img_real = np.array(img_real, dtype=np.uint8)
            if not os.path.exists("./train_tmp_img"):
                os.makedirs("./train_tmp_img")
            cv2.imwrite("./train_tmp_img/epoch_" + str(e) + ".jpg", pred)
            cv2.imwrite("./train_tmp_img/epoch_" + str(e) + "_real.jpg", img_real)


if __name__ == '__main__':
    net = Model(6, mode=args.asr).cuda()
    train(net, args.epochs, args.batchsize, args.lr)