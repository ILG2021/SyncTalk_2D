# 使用方法: .\training_328.ps1 <name> <gpu_id>
param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$file_name,
    [Parameter(Mandatory=$true, Position=1)]
    [string]$cuda_id
)

# 设置环境变量
$env:CUDA_VISIBLE_DEVICES = $cuda_id
$asr = "ave"
$file_path = ".\dataset\$file_name\$file_name.mp4"
$data_dir = ".\dataset\$file_name"

Write-Host "--- Step 1: Processing data ---" -ForegroundColor Cyan
python data_utils/process.py $file_path

Write-Host "--- Step 2: Training SyncNet ---" -ForegroundColor Cyan
python syncnet_328.py --save_dir .\syncnet_ckpt\$file_name --dataset_dir $data_dir --asr $asr

Write-Host "--- Step 3: Finding latest checkpoint ---" -ForegroundColor Cyan
# 获取最新的 .pth 文件 (按文件名数字大小排序)
$checkpoint_folder = ".\syncnet_ckpt\$file_name"
$syncnet_checkpoint = Get-ChildItem "$checkpoint_folder\*.pth" | 
                      Sort-Object { [int]($_.BaseName) } | 
                      Select-Object -Last 1

if ($null -eq $syncnet_checkpoint) {
    Write-Error "No checkpoint found in $checkpoint_folder"
    exit 1
}

Write-Host "Using checkpoint: $($syncnet_checkpoint.FullName)" -ForegroundColor Green

Write-Host "--- Step 4: Training Main Model ---" -ForegroundColor Cyan
python train_328.py --dataset_dir $data_dir --save_dir .\checkpoint\$file_name --asr $asr --use_syncnet --syncnet_checkpoint "$($syncnet_checkpoint.FullName)"