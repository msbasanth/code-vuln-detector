# Retry failed Exp E CodeT5 models with offline HuggingFace hub
# These failed due to transient DNS errors; models are cached locally.
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Host "`n  Retrying codet5-small for Exp E (offline mode)..."
python -m src.train --experiment exp_e_union --model Salesforce/codet5-small --train-path data/processed/union_train.parquet --test-path data/processed/union_test.parquet --label-map-path data/processed/union_label_map.json
if ($LASTEXITCODE -eq 0) { Write-Host "  DONE: codet5-small (Exp E)" } else { Write-Host "  FAILED: codet5-small (Exp E)" }

Write-Host "`n  Retrying codet5-base for Exp E (offline mode)..."
python -m src.train --experiment exp_e_union --model Salesforce/codet5-base --train-path data/processed/union_train.parquet --test-path data/processed/union_test.parquet --label-map-path data/processed/union_label_map.json
if ($LASTEXITCODE -eq 0) { Write-Host "  DONE: codet5-base (Exp E)" } else { Write-Host "  FAILED: codet5-base (Exp E)" }

# Reset env vars
Remove-Item Env:\HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:\TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue

Write-Host "`n  All Exp E retries complete."
