import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio,
                      inp * expand_ratio,
                      3,
                      stride,
                      1,
                      groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class DoubleConvDW(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=2):

        super(DoubleConvDW, self).__init__() 
        self.double_conv = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride=stride, use_res_connect=False, expand_ratio=2),
            InvertedResidual(out_channels, out_channels, stride=1, use_res_connect=True, expand_ratio=2)
        )

    def forward(self, x):
        return self.double_conv(x)

class InConvDw(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InConvDw, self).__init__() 
        self.inconv = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride=1, use_res_connect=False, expand_ratio=2)
        )
    def forward(self, x):
        return self.inconv(x)

class Down(nn.Module):
    
    def __init__(self, in_channels, out_channels):

        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConvDW(in_channels, out_channels, stride=2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv =  DoubleConvDW(in_channels, out_channels, stride=1)

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], axis=1)
        
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class AudioConvWenet(nn.Module):
    def __init__(self):
        super(AudioConvWenet, self).__init__()
        ch = [32, 64, 128, 256, 512]
        self.conv1 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv2 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        
        self.conv3 = nn.Conv2d(ch[3], ch[3], kernel_size=3, padding=1, stride=(1,2))
        self.bn3 = nn.BatchNorm2d(ch[3])
        
        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        
        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.ReLU()
        
        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.conv7(x)
        return x
    
class AudioConvHubert(nn.Module):
    def __init__(self):
        super(AudioConvHubert, self).__init__()
        ch = [32, 64, 128, 256, 512]
        self.conv1 = InvertedResidual(ch[0], ch[1], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv2 = InvertedResidual(ch[1], ch[2], stride=1, use_res_connect=False, expand_ratio=2)
        
        self.conv3 = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1, stride=(2,2))
        self.bn3 = nn.BatchNorm2d(ch[3])
        
        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        
        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.ReLU()
        
        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.conv7(x)
        return x

class AudioConvAve(nn.Module): 
    def __init__(self):
        super(AudioConvAve, self).__init__()
        ch = [32, 64, 128, 256, 512]
        self.conv1 = InvertedResidual(ch[0], ch[1], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv2 = InvertedResidual(ch[1], ch[2], stride=1, use_res_connect=False, expand_ratio=2)
        
        self.conv3 = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(ch[3])
        
        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        
        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.ReLU()
        
        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
    
    def forward(self, x):        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.conv7(x)
        return x
    
class Model(nn.Module):
    def __init__(self,n_channels=6, mode='hubert'):
        super(Model, self).__init__()
        self.n_channels = n_channels
        ch = [32, 64, 128, 256, 512]
        
        if mode=='hubert':
            print("hubert")   
            self.audio_model = AudioConvHubert()
        elif mode=='wenet':
            print("wenet")  
            self.audio_model = AudioConvWenet()
        elif mode=='ave':
            print("ave")
            self.audio_model = AudioConvAve()
            
        self.fuse_conv = nn.Sequential(
            DoubleConvDW(ch[4]*2, ch[4], stride=1),
            DoubleConvDW(ch[4], ch[3], stride=1)
        )

        self.inc = InConvDw(n_channels, ch[0])
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], ch[4])
        self.down5 = Down(ch[4], ch[4])

        self.up1 = Up(ch[4], ch[3]//2)
        self.up2 = Up(ch[3], ch[2]//2)
        self.up3 = Up(ch[2], ch[1]//2)
        self.up4 = Up(ch[1], ch[0])
        self.up5 = Up(ch[1], ch[0]//2)

        self.outc = OutConv(ch[0]//2, 3)

    def forward(self, x, audio_feat):

        x1 = self.inc(x) # 512
        x2 = self.down1(x1) # 256
        x3 = self.down2(x2) # 128
        x4 = self.down3(x3) # 64
        x5 = self.down4(x4) # 32
        x6 = self.down5(x5) # 16

        audio_feat  = self.audio_model(audio_feat) # 16x16

        x6 = torch.cat([x6, audio_feat], axis=1)
        x6 = self.fuse_conv(x6)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        out = self.outc(x)
        out = F.sigmoid(out)
        return out
