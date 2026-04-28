# Run all B vs C experiments sequentially
# B = Juliet-only-19-CWE baseline, C = Juliet + Big-Vul overlap

$models = @(
    "Salesforce/codet5-small",
    "Salesforce/codet5-base",
    "microsoft/codebert-base",
    "microsoft/graphcodebert-base"
)

# Note: B1 (GraphCodeBERT) is already running separately.
# This script runs B2-B4, then all C1-C4.

Write-Host "=== EXPERIMENT B: Juliet-only 19 CWEs ===" -ForegroundColor Cyan

foreach ($model in $models) {
    $variant = ($model -split "/")[-1]
    Write-Host "`n--- Training Exp B: $variant ---" -ForegroundColor Yellow
    python -m src.train `
        --experiment exp_b_juliet19 `
        --model $model `
        --train-path data/processed/juliet19_train.parquet `
        --test-path data/processed/juliet19_test.parquet `
        --label-map-path data/processed/juliet19_label_map.json

    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: Exp B $variant" -ForegroundColor Red
        exit 1
    }
    Write-Host "DONE: Exp B $variant" -ForegroundColor Green
}

Write-Host "`n=== EXPERIMENT C: Juliet + Big-Vul Overlap ===" -ForegroundColor Cyan

foreach ($model in $models) {
    $variant = ($model -split "/")[-1]
    Write-Host "`n--- Training Exp C: $variant ---" -ForegroundColor Yellow
    python -m src.train `
        --experiment exp_c_combined `
        --model $model `
        --train-path data/processed/combined_train.parquet `
        --test-path data/processed/combined_test.parquet `
        --label-map-path data/processed/combined_label_map.json

    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: Exp C $variant" -ForegroundColor Red
        exit 1
    }
    Write-Host "DONE: Exp C $variant" -ForegroundColor Green
}

Write-Host "`n=== ALL EXPERIMENTS COMPLETE ===" -ForegroundColor Cyan
