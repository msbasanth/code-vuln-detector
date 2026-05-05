# Train BiLSTM-Attention for Experiment G: DL Baseline
# Full Juliet C/C++ 1.3 (118 CWEs), BiLSTM with Self-Attention
# Outputs isolated under outputs/exp_g_bilstm/

$ErrorActionPreference = "Stop"

# Activate virtual environment
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

# HuggingFace offline mode (reuses CodeT5 tokenizer from cache)
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Host "`n===== EXPERIMENT G: BiLSTM-Attention DL Baseline =====" -ForegroundColor Cyan

Write-Host "`n  Training BiLSTM-Attention for Exp G..." -ForegroundColor Yellow

python -m src.train `
    --experiment exp_g_bilstm `
    --model bilstm-attention `
    --epochs 10 `
    --patience 3 `
    --train-path data/processed/train.parquet `
    --test-path data/processed/test.parquet `
    --label-map-path data/processed/label_map.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "  FAILED: BiLSTM-Attention (Exp G)" -ForegroundColor Red
} else {
    Write-Host "  DONE: BiLSTM-Attention (Exp G)" -ForegroundColor Green
}

Write-Host "`n===== EXPERIMENT G TRAINING COMPLETE =====" -ForegroundColor Cyan
