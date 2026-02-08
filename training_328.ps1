param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$person_name,
    
    [Parameter(Position=1)]
    [string]$asr = "hubert"
)

# 设置变量
$data_dir = "./dataset/$person_name"

# 1. 批量预处理该目录下所有的视频文件
Write-Host "[STEP 1] Batch processing all videos in $data_dir..." -ForegroundColor Cyan
python data_utils/batch_process.py $data_dir --asr $asr

# 1.5. 平滑关键点 (消除画面抖动的关键步骤)
Write-Host "[STEP 1.5] Smoothing landmarks to reduce jitter..." -ForegroundColor Cyan
python data_utils/smooth_landmarks.py $data_dir

# 1.6. 生成极速训练数据 (images.npy)
Write-Host "[STEP 1.6] Preprocessing images to numpy for fast loading..." -ForegroundColor Cyan
python data_utils/preprocess_to_npy.py $data_dir

# hubert不需要syncnet，ave需要
if ($asr -eq "ave") {
    # 2. 训练 SyncNet
    Write-Host "[STEP 2] Training SyncNet on all sub-datasets..." -ForegroundColor Cyan
    python syncnet_328.py --save_dir ./syncnet_ckpt/$person_name --dataset_dir $data_dir --asr $asr

    # 获取最新的 SyncNet 权重
    $ckpt_dir = "./syncnet_ckpt/$person_name"
    $latest_ckpt = Get-ChildItem -Path $ckpt_dir -Filter "*.pth" | Sort-Object { try { [int]($_.BaseName) } catch { 0 } } | Select-Object -Last 1
    
    if ($null -eq $latest_ckpt) {
        Write-Error "Failed to find SyncNet checkpoint in $ckpt_dir"
        exit 1
    }
    
    $syncnet_path = $latest_ckpt.FullName
    Write-Host "Using SyncNet checkpoint: $syncnet_path" -ForegroundColor Green

    # 3. 训练主模型 (使用 SyncNet)
    Write-Host "[STEP 3] Training main model with SyncNet..." -ForegroundColor Cyan
    python train_328.py --dataset_dir $data_dir --save_dir ./checkpoint/$person_name --asr $asr --use_syncnet --syncnet_checkpoint "$syncnet_path" --use_temporal --temporal_weight 0.5
} else {
    # 3. 训练主模型 (Hubert 模式，不需要 SyncNet)
    Write-Host "[STEP 3] Training main model on all sub-datasets..." -ForegroundColor Cyan
    python train_328.py --dataset_dir $data_dir --save_dir ./checkpoint/$person_name --asr $asr --batchsize 8 --epochs 50
    python train_328.py --dataset_dir $data_dir --save_dir ./checkpoint/$person_name --asr $asr --batchsize 8 --epochs 20 --use_temporal --temporal_weight 0.15
}
