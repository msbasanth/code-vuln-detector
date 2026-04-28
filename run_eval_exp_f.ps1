# Evaluate models for Experiment F: Union Extended Training

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

Write-Host "`n===== EVALUATING EXPERIMENT F: Union Extended Training =====" -ForegroundColor Cyan

foreach ($model in $models) {
    $variant = $model.Split("/")[-1]
    Write-Host "  Evaluating $variant..." -ForegroundColor Yellow

    python -m src.evaluate `
        --experiment exp_f_union_6ep `
        --model $model `
        --test-path data/processed/union_test.parquet `
        --label-map-path data/processed/union_label_map.json

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED: $variant (Exp F)" -ForegroundColor Red
    } else {
        Write-Host "  DONE: $variant (Exp F)" -ForegroundColor Green
    }
}

Write-Host "`n===== EXPERIMENT F EVALUATION COMPLETE =====" -ForegroundColor Cyan
