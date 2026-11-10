# train_sequential.ps1
# Runs training for B1, then B3 sequentially with Asymmetric Loss

$env:PYTHONUNBUFFERED="1"

Write-Host "========================================================="
Write-Host " Starting Phase 1 Training: EfficientNet-B1"
Write-Host "========================================================="
python scripts/train_phase1_classifier.py --backbone efficientnet_b1 --epochs 40 --batch-size 32 --num-workers 4 2>&1 | Tee-Object -FilePath "checkpoints/train_b1.log"
if ($LASTEXITCODE -ne 0) {
    Write-Host "B1 Training failed! Aborting sequential run."
    exit $LASTEXITCODE
}

Write-Host "========================================================="
Write-Host " Starting Phase 1 Training: EfficientNet-B3"
Write-Host "========================================================="
python scripts/train_phase1_classifier.py --backbone efficientnet_b3 --epochs 40 --batch-size 16 --num-workers 4 2>&1 | Tee-Object -FilePath "checkpoints/train_b3.log"
if ($LASTEXITCODE -ne 0) {
    Write-Host "B3 Training failed!"
    exit $LASTEXITCODE
}

Write-Host "Sequential training complete!"
