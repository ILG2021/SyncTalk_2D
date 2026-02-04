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
    parser.add_argument('--asr', type=str, default="hubert")
    parser.add_argument('--temporal', action='store_true', help="Enable temporal consistency loss")
    parser.add_argument('--temporal_weight', type=float, default=0.1, help="Weight for temporal loss")
    parser.add_argument('--syncnet_weight', type=float, default=1.0, help="Weight for syncnet loss")

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

    all_datasets = []
    dataset_root = args.dataset_dir.strip()
    
    if os.path.isdir(dataset_root):
        # 检查当前目录下是否有核心数据文件夹
        if os.path.exists(os.path.join(dataset_root, "full_body_img")):
            candidate_dirs = [dataset_root]
        else:
            # 遍历所有一级子目录
            candidate_dirs = [os.path.join(dataset_root, d) for d in os.listdir(dataset_root) 
                            if os.path.isdir(os.path.join(dataset_root, d))]
            print(f"Scanning for sub-datasets in: {dataset_root}")
    else:
        raise ValueError(f"Dataset path {dataset_root} is not a valid directory.")

    for d_dir in candidate_dirs:
        # 验证该子目录是否是有效的预处理输出 (包含图像文件夹)
        if os.path.exists(os.path.join(d_dir, "full_body_img")):
            print(f"  [FOUND] Loading sub-dataset: {d_dir}")
            all_datasets.append(MyDataset(d_dir, args.asr))
    
    if len(all_datasets) == 0:
        raise ValueError(f"No valid datasets found in {dataset_root}. Make sure subfolders contain 'full_body_img'.")

    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset(all_datasets)
    train_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, 
                                  drop_last=True, num_workers=4, pin_memory=True, 
                                  persistent_workers=True)

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
        with tqdm(total=len(combined_dataset), desc=f'Epoch {e + 1}/{epoch}', unit='img') as p:
            for batch in train_dataloader:
                # 兼容旧版本 Dataset (3 returns) 和新版本 Temporal Dataset (6 returns)
                if len(batch) == 6:
                    imgs, labels, audio_feat, imgs_next, labels_next, audio_feat_next = batch
                else:
                    imgs, labels, audio_feat = batch
                    # 如果 Dataset 没有返回 next frame，就无法计算 Temporal
                    if args.temporal:
                        raise ValueError("Temporal loss requires updated dataset returning 6 elements.")
                
                imgs = imgs.cuda()
                labels = labels.cuda()
                audio_feat = audio_feat.cuda()
                
                # --- Forward T ---
                preds = net(imgs, audio_feat)
                
                # --- Loss Calculation T ---
                if use_syncnet:
                    y = torch.ones([preds.shape[0], 1]).float().cuda()
                    a, v = syncnet(preds, audio_feat)
                    sync_loss = cosine_loss(a, v, y)
                loss_PerceptualLoss = content_loss.get_loss(preds, labels)
                loss_pixel = criterion(preds, labels) # L1 Loss
                
                total_loss = loss_pixel + loss_PerceptualLoss * 0.01
                if use_syncnet:
                    total_loss += args.syncnet_weight * sync_loss

                # --- Temporal Loss ---
                if args.temporal and len(batch) == 6:
                    imgs_next = imgs_next.cuda()
                    labels_next = labels_next.cuda()
                    audio_feat_next = audio_feat_next.cuda()
                    
                    # Forward T+1
                    preds_next = net(imgs_next, audio_feat_next)
                    
                    # Calculate velocity: (Pred_t+1 - Pred_t) vs (Real_t+1 - Real_t)
                    # 我们希望生成视频的“速度”和真实视频的“速度”一致
                    diff_pred = preds_next - preds
                    diff_real = labels_next - labels
                    loss_temporal = criterion(diff_pred, diff_real) # L1 Loss on velocity
                    
                    total_loss += args.temporal_weight * loss_temporal
                    p.set_postfix(**{'loss': total_loss.item(), 'temp_loss': loss_temporal.item()})
                else:
                    p.set_postfix(**{'loss': total_loss.item()})

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()
                p.update(imgs.shape[0])

        if (e + 1) % 5 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, str(e) + '.pth'))
        if args.see_res:
            net.eval()
        if args.see_res:
            net.eval()
            # 随机从某个子数据集中取样预览
            random_ds = random.choice(all_datasets)
            batch_sample = random_ds.__getitem__(random.randint(0, len(random_ds) - 1))
            
            # 兼容不同长度的返回
            if len(batch_sample) == 6:
                img_concat_T, img_real_T, audio_feat = batch_sample[0], batch_sample[1], batch_sample[2]
            else:
                img_concat_T, img_real_T, audio_feat = batch_sample
                
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