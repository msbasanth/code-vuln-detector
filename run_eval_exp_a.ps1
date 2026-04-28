# Evaluate models for Experiment A (Juliet-118 Baseline)

$ErrorActionPreference = "Stop"

& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$models = @(
    "Salesforce/codet5-small",
    "Salesforce/codet5-base",
    "microsoft/codebert-base",
    "microsoft/graphcodebert-base"
)

Write-Host "`n===== EVALUATING EXPERIMENT A: Juliet-118 Baseline =====" -ForegroundColor Cyan

foreach ($model in $models) {
    $variant = $model.Split("/")[-1]
    Write-Host "  Evaluating $variant..." -ForegroundColor Yellow

    python -m src.evaluate `
        --experiment exp_a_juliet118 `
        --model $model `
        --test-path data/processed/test.parquet `
        --label-map-path data/processed/label_map.json

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED: $variant (Exp A)" -ForegroundColor Red
    } else {
        Write-Host "  DONE: $variant (Exp A)" -ForegroundColor Green
    }
}

Write-Host "`n===== EXPERIMENT A EVALUATION COMPLETE =====" -ForegroundColor Cyan
