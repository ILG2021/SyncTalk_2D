
import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import librosa
from transformers import HubertModel, Wav2Vec2Processor

def extract_hubert(wav_path, device='cuda'):
    print(f"[INFO] Extracting Hubert features from {wav_path}...")
    
    # 使用和 Ultralight-Digital-Human 一致的模型
    model_name = "facebook/hubert-large-ls960-ft"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name).to(device)
    model.eval()

    # 使用 librosa 加载，强制单声道和 16000Hz 采样率
    speech, _ = librosa.load(wav_path, sr=16000, mono=True)
    
    # 处理音频输入
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    
    with torch.no_grad():
        # 提取特征。参考 Ultralight-Digital-Human，通常使用中间层特征
        outputs = model(input_values, output_hidden_states=True)
        # 使用第 12 层特征 (indices 0-24, 12 是中间层)
        # 或者使用最后一层，这里我们使用 hidden_states[12]
        feats = outputs.hidden_states[20].squeeze(0) # [T_hubert, 1024]

    feats = feats.cpu().numpy()
    
    # Hubert 输出是 50Hz，视频是 25Hz
    # 将每两个 Hubert 帧合并为一个视频帧 (1024 * 2 = 2048 维度)
    T_hu = feats.shape[0]
    
    # 确保长度为偶数便于 reshape
    if T_hu % 2 != 0:
        feats = np.concatenate([feats, feats[-1:]], axis=0)
        T_hu += 1
    
    # 重塑为 [T_video, 2048]
    feats = feats.reshape(-1, 2048)
    
    # 保存路径：SyncNet 和 Dataset 脚本默认读取 aud_hu.npy
    save_path = wav_path.replace('.wav', '_hu.npy')
    np.save(save_path, feats)
    print(f"[SUCCESS] Hubert features saved to {save_path}, final shape: {feats.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_path', type=str, required=True)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        extract_hubert(args.wav_path, device)
    except Exception as e:
        print(f"[ERROR] Hubert extraction failed: {e}")
        import traceback
        traceback.print_exc()
