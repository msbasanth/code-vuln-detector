# Evaluate BiLSTM-Attention for Experiment G
# Outputs isolated under outputs/exp_g_bilstm/bilstm-attention/

$ErrorActionPreference = "Stop"

# Activate virtual environment
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

# HuggingFace offline mode
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Host "`n===== EXPERIMENT G: BiLSTM-Attention Evaluation =====" -ForegroundColor Cyan

python -m src.evaluate `
    --experiment exp_g_bilstm `
    --model bilstm-attention `
    --dl-model `
    --test-path data/processed/test.parquet `
    --label-map-path data/processed/label_map.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "  FAILED: BiLSTM-Attention evaluation" -ForegroundColor Red
} else {
    Write-Host "  DONE: BiLSTM-Attention evaluation" -ForegroundColor Green
}

Write-Host "`n===== EXPERIMENT G EVALUATION COMPLETE =====" -ForegroundColor Cyan
