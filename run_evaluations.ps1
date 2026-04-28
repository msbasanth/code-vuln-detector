# Run per-model evaluations for all experiments
# Generates per-CWE classification reports and confusion pairs

$ErrorActionPreference = "Stop"

$models = @(
    @{ name = "Salesforce/codet5-small"; variant = "codet5-small" },
    @{ name = "Salesforce/codet5-base"; variant = "codet5-base" },
    @{ name = "microsoft/codebert-base"; variant = "codebert-base" },
    @{ name = "microsoft/graphcodebert-base"; variant = "graphcodebert-base" }
)

# ===== Experiment A: Juliet-118 Baseline =====
Write-Host "`n===== Experiment A: Juliet-118 Baseline =====" -ForegroundColor Cyan
foreach ($m in $models) {
    $variant = $m.variant
    $checkpoint = "outputs/checkpoints/$variant/best.pt"
    if (-not (Test-Path $checkpoint)) {
        Write-Host "  SKIP $variant - no checkpoint at $checkpoint" -ForegroundColor Yellow
        continue
    }
    Write-Host "  Evaluating $variant..." -ForegroundColor Green
    python -m src.evaluate `
        --experiment exp_a_juliet118 `
        --model $m.name `
        --checkpoint $checkpoint `
        --test-path data/processed/test.parquet `
        --label-map-path data/processed/label_map.json
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED: $variant" -ForegroundColor Red
    }
}

# ===== Experiment B: Juliet-19 Subset =====
Write-Host "`n===== Experiment B: Juliet-19 Subset =====" -ForegroundColor Cyan
foreach ($m in $models) {
    $variant = $m.variant
    $checkpoint = "outputs/exp_b_juliet19/checkpoints/$variant/best.pt"
    if (-not (Test-Path $checkpoint)) {
        Write-Host "  SKIP $variant - no checkpoint at $checkpoint" -ForegroundColor Yellow
        continue
    }
    Write-Host "  Evaluating $variant..." -ForegroundColor Green
    python -m src.evaluate `
        --experiment exp_b_juliet19 `
        --model $m.name `
        --checkpoint $checkpoint `
        --test-path data/processed/juliet19_test.parquet `
        --label-map-path data/processed/juliet19_label_map.json
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED: $variant" -ForegroundColor Red
    }
}

# ===== Experiment C: Juliet + Big-Vul Combined =====
Write-Host "`n===== Experiment C: Juliet + Big-Vul Combined =====" -ForegroundColor Cyan
foreach ($m in $models) {
    $variant = $m.variant
    $checkpoint = "outputs/exp_c_combined/checkpoints/$variant/best.pt"
    if (-not (Test-Path $checkpoint)) {
        Write-Host "  SKIP $variant - no checkpoint at $checkpoint" -ForegroundColor Yellow
        continue
    }
    Write-Host "  Evaluating $variant..." -ForegroundColor Green
    python -m src.evaluate `
        --experiment exp_c_combined `
        --model $m.name `
        --checkpoint $checkpoint `
        --test-path data/processed/combined_test.parquet `
        --label-map-path data/processed/combined_label_map.json
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED: $variant" -ForegroundColor Red
    }
}

Write-Host "`n===== ALL EVALUATIONS COMPLETE =====" -ForegroundColor Cyan
