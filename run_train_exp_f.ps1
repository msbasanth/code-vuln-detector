# Train models for Experiment F: Union Extended Training
# Same data as Exp E (Juliet + Big-Vul Union, 187 CWEs) but with 6 epochs and patience=3
# Outputs isolated under outputs/exp_f_union_6ep/

$ErrorActionPreference = "Stop"

# Activate virtual environment
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

# HuggingFace offline mode (CodeT5 models need cached weights)
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$models = @(
    "Salesforce/codet5-small",
    "Salesforce/codet5-base",
    "microsoft/codebert-base",
    "microsoft/graphcodebert-base"
)

Write-Host "`n===== EXPERIMENT F: Union Extended Training (6 epochs, patience=3) =====" -ForegroundColor Cyan

foreach ($model in $models) {
    $variant = $model.Split("/")[-1]
    Write-Host "`n  Training $variant for Exp F..." -ForegroundColor Yellow

    python -m src.train `
        --experiment exp_f_union_6ep `
        --model $model `
        --epochs 6 `
        --patience 3 `
        --train-path data/processed/union_train.parquet `
        --test-path data/processed/union_test.parquet `
        --label-map-path data/processed/union_label_map.json

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED: $variant (Exp F)" -ForegroundColor Red
    } else {
        Write-Host "  DONE: $variant (Exp F)" -ForegroundColor Green
    }
}

Write-Host "`n===== EXPERIMENT F TRAINING COMPLETE =====" -ForegroundColor Cyan
