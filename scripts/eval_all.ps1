Write-Host "========================================================="
Write-Host " Evaluating Phase 2 Attention: EfficientNet-B1"
Write-Host "========================================================="
python -u scripts/evaluate_final.py --checkpoint checkpoints/best_attention_efficientnet_b1.pth --is-attention > checkpoints/evaluation_report_attention_b1.txt 2>&1

Write-Host "========================================================="
Write-Host " Evaluating Phase 2 Attention: EfficientNet-B0"
Write-Host "========================================================="
python -u scripts/evaluate_final.py --checkpoint checkpoints/best_attention_efficientnet_b0.pth --is-attention > checkpoints/evaluation_report_attention_b0.txt 2>&1

Write-Host "Done evaluating all!"
