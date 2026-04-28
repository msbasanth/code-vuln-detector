$ErrorActionPreference = "Stop"

# B2: CodeT5-Small on Juliet-19
Write-Host "`n===== B2: CodeT5-Small on Juliet-19 =====" -ForegroundColor Cyan
python -m src.train --experiment exp_b_juliet19 --model Salesforce/codet5-small --train-path data/processed/juliet19_train.parquet --test-path data/processed/juliet19_test.parquet --label-map-path data/processed/juliet19_label_map.json --config config.yaml
if ($LASTEXITCODE -ne 0) { throw "B2 failed" }

# B3: CodeT5-Base on Juliet-19
Write-Host "`n===== B3: CodeT5-Base on Juliet-19 =====" -ForegroundColor Cyan
python -m src.train --experiment exp_b_juliet19 --model Salesforce/codet5-base --train-path data/processed/juliet19_train.parquet --test-path data/processed/juliet19_test.parquet --label-map-path data/processed/juliet19_label_map.json --config config.yaml
if ($LASTEXITCODE -ne 0) { throw "B3 failed" }

# B4: CodeBERT on Juliet-19
Write-Host "`n===== B4: CodeBERT on Juliet-19 =====" -ForegroundColor Cyan
python -m src.train --experiment exp_b_juliet19 --model microsoft/codebert-base --train-path data/processed/juliet19_train.parquet --test-path data/processed/juliet19_test.parquet --label-map-path data/processed/juliet19_label_map.json --config config.yaml
if ($LASTEXITCODE -ne 0) { throw "B4 failed" }

# C1: GraphCodeBERT on Combined
Write-Host "`n===== C1: GraphCodeBERT on Combined =====" -ForegroundColor Cyan
python -m src.train --experiment exp_c_combined --model microsoft/graphcodebert-base --train-path data/processed/combined_train.parquet --test-path data/processed/combined_test.parquet --label-map-path data/processed/combined_label_map.json --config config.yaml
if ($LASTEXITCODE -ne 0) { throw "C1 failed" }

# C2: CodeT5-Small on Combined
Write-Host "`n===== C2: CodeT5-Small on Combined =====" -ForegroundColor Cyan
python -m src.train --experiment exp_c_combined --model Salesforce/codet5-small --train-path data/processed/combined_train.parquet --test-path data/processed/combined_test.parquet --label-map-path data/processed/combined_label_map.json --config config.yaml
if ($LASTEXITCODE -ne 0) { throw "C2 failed" }

# C3: CodeT5-Base on Combined
Write-Host "`n===== C3: CodeT5-Base on Combined =====" -ForegroundColor Cyan
python -m src.train --experiment exp_c_combined --model Salesforce/codet5-base --train-path data/processed/combined_train.parquet --test-path data/processed/combined_test.parquet --label-map-path data/processed/combined_label_map.json --config config.yaml
if ($LASTEXITCODE -ne 0) { throw "C3 failed" }

# C4: CodeBERT on Combined
Write-Host "`n===== C4: CodeBERT on Combined =====" -ForegroundColor Cyan
python -m src.train --experiment exp_c_combined --model microsoft/codebert-base --train-path data/processed/combined_train.parquet --test-path data/processed/combined_test.parquet --label-map-path data/processed/combined_label_map.json --config config.yaml
if ($LASTEXITCODE -ne 0) { throw "C4 failed" }

Write-Host "`n===== ALL 7 RUNS COMPLETE =====" -ForegroundColor Green
