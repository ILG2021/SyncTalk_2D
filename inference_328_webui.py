
import gradio as gr
import os
import sys
import shutil
import cv2
import torch
import numpy as np
import subprocess
from glob import glob
from tqdm import tqdm

# Add data_utils to path to ensure imports within data_utils work correctly
sys.path.append(os.path.join(os.getcwd(), 'data_utils'))

from unet_328 import Model
from utils import AudioEncoder, AudDataset, get_audio_features

# Attempt to import preprocessing functions
try:
    from data_utils.process import extract_images, get_landmark
except ImportError:
    print("[WARN] Could not import data_utils.process directly. Preprocessing might fail if dependencies are missing.")
    pass

def scan_checkpoints():
    root = "./checkpoint"
    if not os.path.exists(root): return []
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def scan_datasets(person_name):
    if not person_name: return []
    root = os.path.join("./dataset", person_name)
    if not os.path.exists(root): return []
    # Return subdirectories that are likely video datasets
    subdirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    return subdirs

def preprocess_video(video_path, target_dir):
    """
    Refactored from data_utils/process.py to run on a specific target directory
    """
    print(f"[INFO] Preprocessing video: {video_path} -> {target_dir}")
    
    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Copy video to target_dir (renaming to something standard or keeping original)
    video_name = os.path.basename(video_path)
    dist_video_path = os.path.join(target_dir, video_name)
    
    # If file implies a move/copy is needed
    if os.path.abspath(video_path) != os.path.abspath(dist_video_path):
        shutil.copy2(video_path, dist_video_path)
    
    # Use the extract_images logic from process.py
    # Note: extract_images in process.py assumes full_body_img is created relative to the video path's directory
    # So calling it on dist_video_path (which is inside target_dir) will create target_dir/full_body_img
    extract_images(dist_video_path)
    
    # Landmarks
    landmarks_dir = os.path.join(target_dir, "landmarks")
    os.makedirs(landmarks_dir, exist_ok=True)
    get_landmark(dist_video_path, landmarks_dir)
    
    print("[INFO] Preprocessing complete.")

def inference_logic(checkpoint_name, dataset_name, custom_video, audio_path, asr_mode, progress=gr.Progress()):
    if not checkpoint_name:
        raise gr.Error("Please select a checkpoint.")
    
    if not audio_path:
        raise gr.Error("Please upload an audio file.")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 1. Determine Dataset Directory ---
    if custom_video is not None:
        # User provided a custom video. 
        # We need to determine if we have already processed this file.
        # We use the filename to create a subdirectory in dataset/{checkpoint_name}/
        
        # custom_video is the path to the uploaded file
        f_name_full = os.path.basename(custom_video)
        f_name_stem = os.path.splitext(f_name_full)[0]
        
        # Define where this custom dataset should live
        # Using checkpoint_name as parent to keep organized by person
        target_dataset_dir = os.path.join("./dataset", checkpoint_name, f_name_stem)
        
        # Check if it exists and looks valid (has full_body_img)
        if os.path.exists(os.path.join(target_dataset_dir, "full_body_img")):
            print(f"[INFO] Dataset for custom video '{f_name_full}' already exists at {target_dataset_dir}. Using it.")
        else:
            progress(0, desc="Preprocessing Custom Video...")
            # Invoke preprocessing logic
            preprocess_video(custom_video, target_dataset_dir)
            
        dataset_dir = target_dataset_dir
        
    else:
        # Use selected template
        if not dataset_name:
            # Try auto-select
            candidates = scan_datasets(checkpoint_name)
            if candidates:
                dataset_name = candidates[0]
                print(f"[INFO] Auto-selected dataset: {dataset_name}")
            else:
                 # If truly no datasets, check if root has images (legacy structure)
                legacy_check = os.path.join("./dataset", checkpoint_name, "full_body_img")
                if os.path.exists(legacy_check):
                    dataset_dir = os.path.join("./dataset", checkpoint_name)
                    print(f"[INFO] Using legacy root dataset structure at {dataset_dir}")
                else:
                    raise gr.Error("No dataset selected and no custom video provided.")
        
        if not dataset_dir: # if not set by legacy check
             dataset_dir = os.path.join("./dataset", checkpoint_name, dataset_name)
    
    print(f"[INFO] Using Dataset Directory: {dataset_dir}")

    # --- 2. Locate Checkpoint ---
    checkpoint_path = os.path.join("./checkpoint", checkpoint_name)
    if not os.path.exists(checkpoint_path):
         raise gr.Error(f"Checkpoint folder not found: {checkpoint_path}")
         
    checkpoints = sorted([f for f in os.listdir(checkpoint_path) if f.endswith('.pth')], key=lambda x: int(x.split(".")[0]))
    if not checkpoints:
        raise gr.Error("No .pth checkpoints found.")
    checkpoint_file = os.path.join(checkpoint_path, checkpoints[-1])
    print(f"[INFO] Loading Checkpoint: {checkpoint_file}")

    # --- 3. Audio Feature Extraction ---
    progress(0.1, desc="Extracting Audio Features...")
    
    if asr_mode == "ave":
        print("[INFO] Using AVE mode...")
        model_ave = AudioEncoder().to(device).eval()
        ckpt_ave = torch.load('model/checkpoints/audio_visual_encoder.pth')
        model_ave.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt_ave.items()})
        ds_ave = AudDataset(audio_path)
        dl_ave = torch.utils.data.DataLoader(ds_ave, batch_size=64, shuffle=False)
        outputs = []
        for mel in dl_ave:
            mel = mel.to(device)
            with torch.no_grad():
                out = model_ave(mel)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0).cpu()
        first_frame, last_frame = outputs[:1], outputs[-1:]
        audio_feats = torch.cat([first_frame.repeat(1, 1), outputs, last_frame.repeat(1, 1)], dim=0).numpy()
        
    elif asr_mode == "hubert":
        print("[INFO] Using Hubert mode...")
        from transformers import HubertModel, Wav2Vec2Processor
        import librosa
        hubert_model_name = "facebook/hubert-large-ls960-ft"
        processor = Wav2Vec2Processor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name).to(device)
        hubert_model.eval()
        
        speech, _ = librosa.load(audio_path, sr=16000, mono=True)
        input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(device)
        with torch.no_grad():
            outputs = hubert_model(input_values, output_hidden_states=True)
            feats = outputs.hidden_states[12].squeeze(0).cpu().numpy()
            
        T_hu = feats.shape[0]
        if T_hu % 2 != 0:
            feats = np.concatenate([feats, feats[-1:]], axis=0)
        feats = feats.reshape(-1, 2048)
        audio_feats = feats
        
    else:
         raise gr.Error(f"Mode {asr_mode} is not supported in this demo script.")

    # --- 4. Prepare Image Processing ---
    img_dir = os.path.join(dataset_dir, "full_body_img")
    lms_dir = os.path.join(dataset_dir, "landmarks")
    
    if not os.path.exists(img_dir):
        raise gr.Error(f"Missing images at {img_dir}. Preprocessing might have failed.")
    
    # List frames
    frame_list = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')], key=lambda x: int(os.path.splitext(x)[0]))
    if not frame_list:
        raise gr.Error("No frames found in dataset.")
        
    len_img = len(frame_list) - 1
    
    # Read first frame to get dims
    exm_img = cv2.imread(os.path.join(img_dir, frame_list[0]))
    h, w = exm_img.shape[:2]

    # Output Paths
    os.makedirs("./result", exist_ok=True)
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    ckpt_short = os.path.splitext(os.path.basename(checkpoint_file))[0]
    bg_short = os.path.basename(dataset_dir)
    if bg_short == checkpoint_name or bg_short == '.': bg_short = "base"
    
    save_path = os.path.abspath(os.path.join("./result", f"{checkpoint_name}_{bg_short}_{audio_name}_{ckpt_short}.mp4"))
    temp_video_path = save_path.replace(".mp4", "_temp.mp4")

    # Video Writer
    fps = 25
    # If using Wenet, it might be 20, but we only have hubert/ave for now which are typically 25 in this repo
    video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Load Generator
    print("[INFO] Loading Unet...")
    net = Model(6, asr_mode).to(device)
    net.load_state_dict(torch.load(checkpoint_file, map_location=device))
    net.eval()
    
    # --- 5. Inference Loop ---
    step_stride = 0
    img_idx = 0
    start_frame = 0 
    
    print(f"[INFO] Starting Inference: {audio_feats.shape[0]} frames...")
    
    for i in progress.tqdm(iterable=range(audio_feats.shape[0]), desc="Rendering Video"):
        # Ping-pong loop logic for frames
        if img_idx > len_img - 1:
            step_stride = -1
        if img_idx < 1:
            step_stride = 1
        img_idx += step_stride
        
        current_frame_idx = img_idx + start_frame
        
        img_path = os.path.join(img_dir, str(current_frame_idx) + '.jpg')
        lms_path = os.path.join(lms_dir, str(current_frame_idx) + '.lms')
        
        if not os.path.exists(img_path) or not os.path.exists(lms_path):
             continue

        img = cv2.imread(img_path)
        
        # Load Landmarks
        lms_list = []
        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)
        lms = np.array(lms_list, dtype=np.int32)
        
        # Crop Coords (Must match training/process logic)
        xmin = lms[1][0]
        ymin = lms[52][1]
        xmax = lms[31][0]
        width_face = xmax - xmin
        ymax = ymin + width_face
        
        # Crop
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
        crop_img_ori = crop_img.copy()
        
        # Prepare Input Tensor
        img_real_ex = crop_img[4:324, 4:324].copy()
        img_real_ex_ori = img_real_ex.copy()
        
        img_masked = cv2.rectangle(img_real_ex_ori, (5, 5, 310, 305), (0, 0, 0), -1)
        
        img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)
        
        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)
        
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
        
        # Audio Feature
        curr_feat = get_audio_features(audio_feats, i)
        if asr_mode == "hubert":
             curr_feat = curr_feat.reshape(32, 32, 32)
        elif asr_mode == "ave":
             curr_feat = curr_feat.reshape(32, 16, 16)
             
        curr_feat = curr_feat[None].to(device)
        img_concat_T = img_concat_T.to(device)
        
        # Predict
        with torch.no_grad():
             pred = net(img_concat_T, curr_feat)[0]
             
        pred = pred.cpu().numpy().transpose(1, 2, 0) * 255
        pred = np.array(pred, dtype=np.uint8)
        
        # Paste back
        crop_img_ori[4:324, 4:324] = pred
        crop_img_ori = cv2.resize(crop_img_ori, (width_face, width_face), interpolation=cv2.INTER_CUBIC)
        img[ymin:ymax, xmin:xmax] = crop_img_ori
        
        video_writer.write(img)

    video_writer.release()
    
    # --- 6. Mux Audio ---
    print(f"[INFO] Muxing audio to {save_path}...")
    cmd = f'ffmpeg -i "{temp_video_path}" -i "{audio_path}" -c:v libx264 -c:a aac -crf 20 "{save_path}" -y'
    subprocess.run(cmd, shell=True)
    
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
        
    print("[INFO] Done.")
    return save_path


# --- UI Construction ---

def create_demo():
    with gr.Blocks(title="SyncTalk_2D Inference") as demo:
        gr.Markdown("# SyncTalk_2D Inference")
        
        with gr.Row():
            checkpoint_dd = gr.Dropdown(choices=scan_checkpoints(), label="1. Select Checkpoint (Person)", value=None)
            dataset_dd = gr.Dropdown(choices=[], label="2. Select Reference Video Template (from dataset)")
        
        with gr.Row():
            custom_video = gr.File(label="OR Upload Custom Reference Video (overrides template)", file_types=["video"])
            audio_input = gr.File(label="3. Upload Driving Audio", file_types=["audio"])
        
        asr_mode = gr.Dropdown(choices=["hubert", "ave"], value="hubert", label="ASR Mode")
        
        btn = gr.Button("Generate Video", variant="primary")
        
        output_video = gr.Video(label="Result")
        
        # Event Callbacks
        def update_ds_choices(chk):
            choices = scan_datasets(chk)
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
        
        checkpoint_dd.change(fn=update_ds_choices, inputs=checkpoint_dd, outputs=dataset_dd)
        
        btn.click(fn=inference_logic, 
                  inputs=[checkpoint_dd, dataset_dd, custom_video, audio_input, asr_mode], 
                  outputs=output_video)
        
    return demo

if __name__ == "__main__":
    demo = create_demo()
    # Share=True creates a public link, but might not be desired in all envs. 
    # server_name="0.0.0.0" allows access from local network.
    demo.queue().launch(inbrowser=True, server_name="0.0.0.0", server_port=7860) 
