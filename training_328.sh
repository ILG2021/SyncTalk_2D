#! /bin/bash
# input: bash training_328.sh file_name cuda_id
file_name=$1
cuda_id=$2
asr="whisper"
temporal="--temporal --temporal_weight 1.0"
person_dir="./dataset/$file_name"
data_dir="./dataset/$file_name/preprocess"

# 1. 优雅地批量处理该人名下的所有视频文件
python data_utils/batch_process.py $person_dir --asr $asr

# 2. 训练 SyncNet (自动遍历 preprocess 下的所有子目录)
CUDA_VISIBLE_DEVICES=$cuda_id python syncnet_328.py --save_dir ./syncnet_ckpt/$file_name --dataset_dir $data_dir --asr $asr

# 3. 寻找最新的检查点并开始主训练
syncnet_checkpoint_dir=$(ls -v ./syncnet_ckpt/$file_name/*.pth | tail -n 1)
CUDA_VISIBLE_DEVICES=$cuda_id python train_328.py --dataset_dir $data_dir --save_dir ./checkpoint/$file_name --asr $asr --use_syncnet --syncnet_checkpoint $syncnet_checkpoint_dir $temporal
