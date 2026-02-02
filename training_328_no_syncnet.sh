#! /bin/bash
# input: bash training_328_no_syncnet.sh file_name cuda_id
# 这是一个更高效、更稳定的版本，跳过了 SyncNet 预训练，完全依赖 Whisper 特征和时序损失。

file_name=$1
cuda_id=$2
asr="whisper"
temporal="--temporal --temporal_weight 1.0"
person_dir="./dataset/$file_name"
data_dir="./dataset/$file_name/preprocess"

# 1. 优雅地批量处理该人名下的所有原始视频文件
echo "[STEP 1] Batch processing raw videos..."
python data_utils/batch_process.py $person_dir --asr $asr

# 2. 直接开始主模型训练 (跳过 SyncNet)
# 依赖 Whisper 的强特征和 Temporal Loss 来保证口型精准度和画面稳定性
echo "[STEP 2] Starting main training without SyncNet..."
CUDA_VISIBLE_DEVICES=$cuda_id python train_328.py \
    --dataset_dir $data_dir \
    --save_dir ./checkpoint/$file_name \
    --asr $asr \
    $temporal \
    --epochs 100 \
    --batchsize 16

echo "[DONE] Training pipeline finished."
