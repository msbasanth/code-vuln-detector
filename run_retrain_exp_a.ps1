# Retrain models for Experiment A (Juliet-118 Baseline)
# Retraining to generate epoch_metrics.json (missing from original run)
# Uses --experiment flag so outputs go to outputs/exp_a_juliet118/

$ErrorActionPreference = "Stop"

# Activate virtual environment
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

# HuggingFace offline mode
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$models = @(
    "Salesforce/codet5-small",
    "Salesforce/codet5-base",
    "microsoft/codebert-base",
    "microsoft/graphcodebert-base"
)

Write-Host "`n===== EXPERIMENT A: Juliet-118 Baseline (retrain for epoch metrics) =====" -ForegroundColor Cyan

foreach ($model in $models) {
    $variant = $model.Split("/")[-1]
    Write-Host "`n  Training $variant for Exp A..." -ForegroundColor Yellow

    python -m src.train `
        --experiment exp_a_juliet118 `
        --model $model `
        --train-path data/processed/train.parquet `
        --test-path data/processed/test.parquet `
        --label-map-path data/processed/label_map.json

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED: $variant (Exp A)" -ForegroundColor Red
    } else {
        Write-Host "  DONE: $variant (Exp A)" -ForegroundColor Green
    }
}

Write-Host "`n===== EXPERIMENT A TRAINING COMPLETE =====" -ForegroundColor Cyan
