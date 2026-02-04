#! /bin/bash
# input: bash training_328.sh person_name cuda_id
person_name=$1
cuda_id=$2
asr="hubert"
temporal="--temporal --temporal_weight 0.05"
data_dir="./dataset/$person_name"

# 1. 批量预处理该目录下所有的视频文件
# 它会自动为每个视频生成子文件夹并完成提取
echo "[STEP 1] Batch processing all videos in $data_dir..."
python data_utils/batch_process.py $data_dir --asr $asr

# 1.5. 平滑关键点 (消除画面抖动的关键步骤)
echo "[STEP 1.5] Smoothing landmarks to reduce jitter..."
python data_utils/smooth_landmarks.py $data_dir

# 1.6. 生成极速训练数据 (images.npy)
echo "[STEP 1.6] Preprocessing images to numpy for fast loading..."
python data_utils/preprocess_to_npy.py $data_dir

# 2. 训练 SyncNet (它现在会自动扫描 data_dir 下的所有子文件夹)
echo "[STEP 2] Training SyncNet on all sub-datasets..."
CUDA_VISIBLE_DEVICES=$cuda_id python syncnet_328.py --save_dir ./syncnet_ckpt/$person_name --dataset_dir $data_dir --asr $asr

# 获取最新的 SyncNet 权重
syncnet_checkpoint_dir=$(ls -v ./syncnet_ckpt/$person_name/*.pth | tail -n 1)

# 3. 训练主模型 (它现在也会自动扫描 data_dir 下的所有子文件夹)
echo "[STEP 3] Training main model on all sub-datasets..."
CUDA_VISIBLE_DEVICES=$cuda_id python train_328.py --dataset_dir $data_dir --save_dir ./checkpoint/$person_name --asr $asr --use_syncnet --syncnet_checkpoint $syncnet_checkpoint_dir $temporal
