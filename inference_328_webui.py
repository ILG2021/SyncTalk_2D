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

import random
import librosa
from unet_328 import Model
from utils import AudioEncoder, AudDataset, get_audio_features

# Attempt to import preprocessing functions
try:
    from data_utils.process import extract_images, get_landmark
except ImportError:
    print("[WARN] Could not import data_utils.process directly. Preprocessing might fail if dependencies are missing.")
    pass


def scan_checkpoints():
    root = "checkpoint"
    if not os.path.exists(root): return []
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])


def scan_datasets(person_name):
    if not person_name: return []
    root = os.path.join("dataset", person_name)
    if not os.path.exists(root): return []
    # Return subdirectories that are likely video datasets
    subdirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    return subdirs


def get_random_start_frame(aud_path, fps=25):
    if not aud_path or not os.path.exists(aud_path):
        return 0
    try:
        y, sr = librosa.load(aud_path, sr=None)
        # top_db=30 is a common threshold for "silence"
        intervals = librosa.effects.split(y, top_db=30)

        silence_intervals = []
        last_pos = 0
        for start, end in intervals:
            if start > last_pos:
                silence_intervals.append((last_pos, start))
            last_pos = end
        if last_pos < len(y):
            silence_intervals.append((last_pos, len(y)))

        # Filter regions >= 1 second
        min_samples = sr  # 1 second
        valid_regions = [r for r in silence_intervals if (r[1] - r[0]) >= min_samples]

        if not valid_regions:
            print(f"[INFO] No silence period >= 1s found in {aud_path}. Defaulting to frame 0.")
            return 0

        # Randomly select one
        region = random.choice(valid_regions)
        # Start at 300ms after the silence begins
        start_sample = region[0] + int(0.3 * sr)

        # Ensure we don't exceed the region's end
        if start_sample > region[1]:
            start_sample = region[0]

        start_frame = int(start_sample / sr * fps)
        print(f"[INFO] Auto-selected start_frame: {start_frame} (from silence starting at {region[0] / sr:.2f}s)")
        return start_frame
    except Exception as e:
        print(f"[WARN] Failed to auto-select start frame: {e}")
        return 0


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
    extract_images(dist_video_path)

    # Landmarks
    landmarks_dir = os.path.join(target_dir, "landmarks")
    os.makedirs(landmarks_dir, exist_ok=True)
    get_landmark(dist_video_path, landmarks_dir)

    # Extract audio for silence detection
    aud_wav_path = os.path.join(target_dir, "aud.wav")
    from data_utils.process import extract_audio
    extract_audio(dist_video_path, aud_wav_path)

    print("[INFO] Preprocessing complete.")


def inference_logic(checkpoint_name, dataset_name, custom_video, audio_files, asr_mode, progress=gr.Progress()):
    if not checkpoint_name:
        raise gr.Error("请选择一个模型权重（人物）。")

    if not audio_files:
        raise gr.Error("请上传音频文件。")

    # Normalize audio_files to a list of paths
    if not isinstance(audio_files, list):
        audio_paths = [audio_files.name]
    else:
        audio_paths = [f.name for f in audio_files]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. Determine Dataset Directory ---
    if custom_video is not None:
        # User provided a custom video.
        f_name_full = os.path.basename(custom_video)
        f_name_stem = os.path.splitext(f_name_full)[0]

        # Define where this custom dataset should live
        target_dataset_dir = os.path.join("dataset", checkpoint_name, f_name_stem)

        # Check if it exists and looks valid (has full_body_img)
        if os.path.exists(os.path.join(target_dataset_dir, "full_body_img")):
            print(f"[INFO] Dataset for custom video '{f_name_full}' already exists at {target_dataset_dir}. Using it.")
        else:
            progress(0, desc="正在预处理自定义视频...")
            # Invoke preprocessing logic
            preprocess_video(custom_video, target_dataset_dir)

        dataset_dir = target_dataset_dir

    else:
        dataset_dir = None
        # Use selected template
        if not dataset_name:
            # Try auto-select
            candidates = scan_datasets(checkpoint_name)
            if candidates:
                dataset_name = candidates[0]
                print(f"[INFO] Auto-selected dataset: {dataset_name}")
            else:
                # If truly no datasets, check if root has images (legacy structure)
                legacy_check = os.path.join("dataset", checkpoint_name, "full_body_img")
                if os.path.exists(legacy_check):
                    dataset_dir = os.path.join("dataset", checkpoint_name)
                    print(f"[INFO] Using legacy root dataset structure at {dataset_dir}")
                else:
                    raise gr.Error("未选择模板且未提供自定义视频。")

        if not dataset_dir:  # if not set by legacy check
            dataset_dir = os.path.join("dataset", checkpoint_name, dataset_name)

    print(f"[INFO] Using Dataset Directory: {dataset_dir}")

    # --- 2. Locate Checkpoint ---
    checkpoint_path = os.path.join("checkpoint", checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise gr.Error(f"未找到模型文件夹: {checkpoint_path}")

    checkpoints = sorted([f for f in os.listdir(checkpoint_path) if f.endswith('.pth')],
                         key=lambda x: int(x.split(".")[0]))
    if not checkpoints:
        raise gr.Error("文件夹内未找到 .pth 权重文件。")
    checkpoint_file = os.path.join(checkpoint_path, checkpoints[-1])
    print(f"[INFO] Loading Checkpoint: {checkpoint_file}")

    # --- 3. Load Models (Once per batch) ---
    progress(0.05, desc="正在加载生成器模型...")
    net = Model(6, asr_mode).to(device)
    net.load_state_dict(torch.load(checkpoint_file, map_location=device))
    net.eval()

    hubert_model = None
    processor = None
    model_ave = None

    if asr_mode == "hubert":
        from transformers import HubertModel, Wav2Vec2Processor
        hubert_model_name = "facebook/hubert-large-ls960-ft"
        processor = Wav2Vec2Processor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name).to(device)
        hubert_model.eval()
    elif asr_mode == "ave":
        model_ave = AudioEncoder().to(device).eval()
        ckpt_ave = torch.load('model/checkpoints/audio_visual_encoder.pth')
        model_ave.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt_ave.items()})

    # --- 4. Prepare Shared Dataset Info ---
    img_dir = os.path.join(dataset_dir, "full_body_img")
    lms_dir = os.path.join(dataset_dir, "landmarks")

    if not os.path.exists(img_dir):
        raise gr.Error(f"图像目录丢失: {img_dir}。预处理可能失败了。")

    frame_list = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')],
                        key=lambda x: int(os.path.splitext(x)[0]))
    if not frame_list:
        raise gr.Error("数据集中未找到图像帧。")

    len_img = len(frame_list) - 1
    exm_img = cv2.imread(os.path.join(img_dir, frame_list[0]))
    h, w = exm_img.shape[:2]
    fps = 25
    aud_template_wav = os.path.join(dataset_dir, "aud.wav")

    results = []

    # --- 5. Batch Process loop ---
    for batch_idx, audio_path in enumerate(audio_paths):
        # Calculate progress window for this audio
        p_base = batch_idx / len(audio_paths)
        p_step = 1.0 / len(audio_paths)

        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        progress(p_base, desc=f"正在处理 {batch_idx + 1}/{len(audio_paths)}: {audio_name}")

        # 5.1 Audio Feature Extraction for this audio
        if asr_mode == "ave":
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
            speech, _ = librosa.load(audio_path, sr=16000, mono=True)
            input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(device)
            with torch.no_grad():
                outputs = hubert_model(input_values, output_hidden_states=True)
                feats = outputs.hidden_states[20].squeeze(0).cpu().numpy()

            if feats.shape[0] % 2 != 0:
                feats = np.concatenate([feats, feats[-1:]], axis=0)
            audio_feats = feats.reshape(-1, 2048)

        # 5.2 Output Paths
        os.makedirs("result", exist_ok=True)
        ckpt_short = os.path.splitext(os.path.basename(checkpoint_file))[0]
        bg_short = os.path.basename(dataset_dir)
        if bg_short == checkpoint_name or bg_short == '.': bg_short = "base"
        save_path = os.path.abspath(
            os.path.join("result", f"{checkpoint_name}_{bg_short}_{audio_name}_{ckpt_short}.mp4"))
        temp_video_path = save_path.replace(".mp4", "_temp.mp4")

        # 5.3 Start Frame Selection
        # Requirement: Detect silence >= 1s in Template aud.wav, select random, offset 300ms
        start_frame = get_random_start_frame(aud_template_wav, fps=fps)

        # 5.4 Video Inference loop
        video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        img_idx = 0
        step_stride = 1

        print(f"[INFO] Rendering {audio_name} - {audio_feats.shape[0]} frames start from {start_frame}...")

        for i in range(audio_feats.shape[0]):
            # Update progress within the window
            inner_p = i / audio_feats.shape[0]
            progress(p_base + inner_p * p_step, desc=f"正在生成 {audio_name} ({i}/{audio_feats.shape[0]})")

            # Ping-pong loop logic for background frames
            if img_idx > len_img - 1:
                step_stride = -1
            if img_idx < 1:
                step_stride = 1
            img_idx += step_stride

            current_frame_idx = img_idx + start_frame
            # Loop around if start_frame + img_idx exceeds length
            if current_frame_idx > len_img:
                current_frame_idx = current_frame_idx % (len_img + 1)

            img_path = os.path.join(img_dir, str(current_frame_idx) + '.jpg')
            lms_path = os.path.join(lms_dir, str(current_frame_idx) + '.lms')

            if not os.path.exists(img_path) or not os.path.exists(lms_path):
                # Attempt fallback to simple idx
                current_frame_idx = img_idx % (len_img + 1)
                img_path = os.path.join(img_dir, str(current_frame_idx) + '.jpg')
                lms_path = os.path.join(lms_dir, str(current_frame_idx) + '.lms')
                if not os.path.exists(img_path): continue

            img = cv2.imread(img_path)

            # Load Landmarks
            lms_list = []
            with open(lms_path, "r") as f:
                lines = f.read().splitlines()
                for line in lines:
                    arr = np.array(line.split(" "), dtype=np.float32)
                    lms_list.append(arr)
            lms = np.array(lms_list, dtype=np.int32)

            # Crop Coords
            xmin = lms[1][0]
            ymin = lms[52][1]
            xmax = lms[31][0]
            width_face = xmax - xmin
            ymax = ymin + width_face

            # Crop & Resize
            crop_img_raw = img[ymin:ymax, xmin:xmax]
            if crop_img_raw.size == 0: continue
            crop_img = cv2.resize(crop_img_raw, (328, 328), interpolation=cv2.INTER_CUBIC)
            crop_img_ori = crop_img.copy()

            # Prepare Input Tensor
            img_real_ex = crop_img[4:324, 4:324].copy()
            img_masked = cv2.rectangle(img_real_ex.copy(), (5, 5, 310, 305), (0, 0, 0), -1)

            img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
            img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)

            img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
            img_masked_T = torch.from_numpy(img_masked / 255.0)
            img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None].to(device)

            # Audio Feature
            curr_feat = get_audio_features(audio_feats, i)
            if asr_mode == "hubert":
                curr_feat = curr_feat.reshape(32, 32, 32)
            elif asr_mode == "ave":
                curr_feat = curr_feat.reshape(32, 16, 16)

            curr_feat = curr_feat[None].to(device)

            # Predict
            with torch.no_grad():
                pred = net(img_concat_T, curr_feat)[0]

            pred = (pred.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # Paste back
            crop_img_ori[4:324, 4:324] = pred
            crop_img_final = cv2.resize(crop_img_ori, (width_face, width_face), interpolation=cv2.INTER_CUBIC)
            img[ymin:ymax, xmin:xmax] = crop_img_final
            video_writer.write(img)

        video_writer.release()

        # 5.5 Mux Audio
        print(f"[INFO] Muxing audio to {save_path}...")
        cmd = f'ffmpeg -i "{temp_video_path}" -i "{audio_path}" -c:v libx264 -c:a aac -crf 20 "{save_path}" -y'
        subprocess.run(cmd, shell=True, capture_output=True)

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        results.append(save_path)

    print("[INFO] Batch Processing Done.")
    return results[-1], results


# --- UI Construction ---

def create_demo():
    with gr.Blocks(title="SyncTalk_2D 推理工具") as demo:
        gr.Markdown("# SyncTalk_2D 视频生成推理 (批量版)")

        with gr.Row():
            checkpoint_dd = gr.Dropdown(choices=scan_checkpoints(), label="1. 选择模型权重 (人物)", value=None)
            dataset_dd = gr.Dropdown(choices=[], label="2. 选择视频模板 (来自数据集)")

        with gr.Row():
            custom_video = gr.File(label="或者：上传自定义参考视频 (覆盖模板)", file_types=["video"])
            audio_input = gr.File(label="3. 上传驱动音频 (支持批量上传)", file_types=["audio"], file_count="multiple")

        asr_mode = gr.Dropdown(choices=["hubert", "ave"], value="hubert", label="ASR 特征模式")

        btn = gr.Button("开始生成视频", variant="primary")

        with gr.Row():
            output_video = gr.Video(label="最新生成结果")
            output_files = gr.File(label="批次所有视频列表", file_count="multiple")

        # Event Callbacks
        def update_ds_choices(chk):
            choices = scan_datasets(chk)
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

        checkpoint_dd.change(fn=update_ds_choices, inputs=checkpoint_dd, outputs=dataset_dd)

        btn.click(fn=inference_logic,
                  inputs=[checkpoint_dd, dataset_dd, custom_video, audio_input, asr_mode],
                  outputs=[output_video, output_files])

    return demo


if __name__ == "__main__":
    demo = create_demo()
    # Share=True creates a public link, but might not be desired in all envs. 
    # server_name="0.0.0.0" allows access from local network.
    demo.queue().launch(inbrowser=True, server_port=7860)