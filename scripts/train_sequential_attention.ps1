Write-Host "========================================================="
Write-Host " Starting Phase 2 Attention Training: EfficientNet-B1"
Write-Host "========================================================="
python scripts/train_phase2_attention.py --base-checkpoint checkpoints/best_classifier_efficientnet_b1.pth --batch-size 32 --epochs 20

if ($LASTEXITCODE -ne 0) {
    Write-Error "Phase 2 B1 training failed!"
    exit $LASTEXITCODE
}

Write-Host "========================================================="
Write-Host " Starting Phase 2 Attention Training: EfficientNet-B0"
Write-Host "========================================================="
python scripts/train_phase2_attention.py --base-checkpoint checkpoints/best_classifier_v1.pth --batch-size 32 --epochs 20

if ($LASTEXITCODE -ne 0) {
    Write-Error "Phase 2 B0 training failed!"
    exit $LASTEXITCODE
}

Write-Host "Sequential Phase 2 Attention training complete!"
