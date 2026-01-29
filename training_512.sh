#!/bin/bash
# Usage: ./training_512.sh <name> <gpu_id>
name=$1
gpu_id=$2
export CUDA_VISIBLE_DEVICES=$gpu_id
asr="ave"
file_path="./dataset/$name/$name.mp4"
data_dir="./dataset/$name"
echo "--- Step 1: Processing data ---"
python data_utils/process.py $file_path
echo "--- Step 2: Training SyncNet (512px) ---"
python syncnet_512.py --save_dir ./syncnet_512_ckpt/$name --dataset_dir $data_dir --asr $asr
echo "--- Step 3: Finding latest checkpoint ---"
checkpoint_folder="./syncnet_512_ckpt/$name"
syncnet_checkpoint=$(ls -v $checkpoint_folder/*.pth | tail -n 1)
if [ -z "$syncnet_checkpoint" ]; then echo "No checkpoint found"; exit 1; fi
echo "Using checkpoint: $syncnet_checkpoint"
echo "--- Step 4: Training Main Model (512px) ---"
python train_512.py --dataset_dir $data_dir --save_dir ./checkpoint/$name --asr $asr --use_syncnet --syncnet_checkpoint "$syncnet_checkpoint"
