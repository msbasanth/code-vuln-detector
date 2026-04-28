# Evaluate models for Experiment D (Big-Vul only) and Experiment E (Union)

$ErrorActionPreference = "Stop"

# Activate virtual environment
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

$models = @(
    "Salesforce/codet5-small",
    "Salesforce/codet5-base",
    "microsoft/codebert-base",
    "microsoft/graphcodebert-base"
)

# --- Experiment D: Big-Vul Only ---
Write-Host "`n===== EVALUATING EXPERIMENT D: Big-Vul Only =====" -ForegroundColor Cyan

foreach ($model in $models) {
    $variant = $model.Split("/")[-1]
    Write-Host "  Evaluating $variant..." -ForegroundColor Yellow

    python -m src.evaluate `
        --experiment exp_d_bigvul `
        --model $model `
        --test-path data/processed/bigvul_test.parquet `
        --label-map-path data/processed/bigvul_label_map.json

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED: $variant (Exp D)" -ForegroundColor Red
    } else {
        Write-Host "  DONE: $variant (Exp D)" -ForegroundColor Green
    }
}

# --- Experiment E: Juliet + Big-Vul Union ---
Write-Host "`n===== EVALUATING EXPERIMENT E: Juliet + Big-Vul Union =====" -ForegroundColor Cyan

foreach ($model in $models) {
    $variant = $model.Split("/")[-1]
    Write-Host "  Evaluating $variant..." -ForegroundColor Yellow

    python -m src.evaluate `
        --experiment exp_e_union `
        --model $model `
        --test-path data/processed/union_test.parquet `
        --label-map-path data/processed/union_label_map.json

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED: $variant (Exp E)" -ForegroundColor Red
    } else {
        Write-Host "  DONE: $variant (Exp E)" -ForegroundColor Green
    }
}

Write-Host "`n===== ALL EVALUATIONS COMPLETE =====" -ForegroundColor Cyan
