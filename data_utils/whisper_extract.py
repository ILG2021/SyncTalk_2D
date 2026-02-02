
import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import librosa
from transformers import WhisperModel, WhisperProcessor

def extract_whisper(wav_path, device='cuda'):
    print(f"[INFO] Extracting Whisper-Tiny features from {wav_path}...")
    
    # 使用 whisper-tiny (LatentSync 风格，轻量且稳健)
    model_name = "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name).to(device)
    model.eval()

    # 加载音频
    speech, _ = librosa.load(wav_path, sr=16000, mono=True)
    
    # 处理输入
    input_features = processor(speech, return_tensors="pt", sampling_rate=16000).input_features.to(device)
    
    # Whisper 的 encoder 输出是 50Hz
    with torch.no_grad():
        encoder_outputs = model.encoder(input_features)
        # tiny 版本的 last_hidden_state shape: [1, T_whisper, 384]
        feats = encoder_outputs.last_hidden_state.squeeze(0)

    feats = feats.cpu().numpy()
    
    # 截断到音频实际长度对应的 50Hz 帧数
    audio_len_s = len(speech) / 16000
    expected_hu_frames = int(audio_len_s * 50)
    feats = feats[:expected_hu_frames]
    
    # 对齐到 25FPS 视频
    T_hu = feats.shape[0]
    if T_hu % 2 != 0:
        feats = np.concatenate([feats, feats[-1:]], axis=0)
        T_hu += 1
    
    # 每两帧合并，维度变为 384 * 2 = 768
    feats = feats.reshape(-1, 768)
    
    # 保存路径
    save_path = wav_path.replace('.wav', '_whisper.npy')
    np.save(save_path, feats)
    print(f"[SUCCESS] Whisper features saved to {save_path}, final shape: {feats.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_path', type=str, required=True)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        extract_whisper(args.wav_path, device)
    except Exception as e:
        print(f"[ERROR] Whisper extraction failed: {e}")
        import traceback
        traceback.print_exc()
