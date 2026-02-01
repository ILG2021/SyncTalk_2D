
import os
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import argparse
from transformers import HubertModel, Wav2Vec2Processor

def extract_hubert(wav_path, device='cuda'):
    print(f"Extracting Hubert features from {wav_path}...")
    
    # Load model and processor
    model_name = "facebook/hubert-large-ls960-ft"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name).to(device)
    model.eval()

    # Load audio
    speech, sample_rate = sf.read(wav_path)
    if sample_rate != 16000:
        import librosa
        speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    # Process audio
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    
    with torch.no_grad():
        # Get hidden states, use layer 12 (middle/late layer often good for lip sync) or last layer
        outputs = model(input_values)
        # last_hidden_state shape: [1, T_hubert, 1024]
        feats = outputs.last_hidden_state.squeeze(0) # [T_hubert, 1024]

    feats = feats.cpu().numpy()
    
    # Hubert output is 50Hz, Video is 25Hz
    # Group every 2 Hubert frames into 1 video frame (1024 * 2 = 2048)
    T_hu = feats.shape[0]
    if T_hu % 2 != 0:
        feats = np.concatenate([feats, feats[-1:]], axis=0)
        T_hu += 1
    
    feats = feats.reshape(-1, 2048) # [T_video, 2048]
    
    # Padding: first and last frame repeat (matching ave script behavior)
    first_frame = feats[0:1]
    last_frame = feats[-1:]
    feats = np.concatenate([first_frame, feats, last_frame], axis=0)
    
    save_path = wav_path.replace('.wav', '_hu.npy')
    np.save(save_path, feats)
    print(f"Hubert features saved to {save_path}, shape: {feats.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_path', type=str, required=True)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extract_hubert(args.wav_path, device)
