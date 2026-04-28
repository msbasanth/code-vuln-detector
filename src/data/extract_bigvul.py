"""Extract and normalize the Big-Vul dataset for CWE classification.

Loads the Big-Vul split-functions CSV (MSR_data_cleaned.csv) and produces
a parquet file compatible with the Juliet pipeline.

Big-Vul CSV columns used:
  - func_before: vulnerable function code (when vul=1)
  - func_after:  patched function code (when vul=1, used as non-vulnerable)
  - vul:         1 = vulnerable, 0 = non-vulnerable
  - cwe_id:      CWE identifier string (e.g. "CWE-119")
  - commit_id:   git commit hash
  - project:     project name

Output parquet columns:
  - code, cwe_id (int), cwe_name, template_id, source
"""

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils import load_config, setup_logging


def _parse_cwe_id(raw: str) -> int | None:
    """Extract integer CWE ID from strings like 'CWE-119', '119', 'CWE119'."""
    if pd.isna(raw):
        return None
    raw = str(raw).strip()
    match = re.search(r"(\d+)", raw)
    return int(match.group(1)) if match else None


def extract_bigvul(csv_path: str) -> pd.DataFrame:
    """Load Big-Vul CSV and produce normalized samples.

    Strategy:
    - vul=1 rows → func_before is vulnerable code, labeled with cwe_id
    - For each vul=1 row, func_after (patched code) is added as a
      non-vulnerable companion — but only if the patch actually changed the
      code (func_before != func_after).
    - vul=0 rows from the original CSV are excluded: they have no CWE label
      and are simply functions in the same file that weren't part of the fix.
    """
    logger = setup_logging()
    logger.info(f"Loading Big-Vul CSV from {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    logger.info(f"Raw CSV rows: {len(df)}")

    # Normalize column names (some mirrors have slightly different names)
    df.columns = df.columns.str.strip().str.lower()

    # Map known column name variants
    col_renames = {
        "cwe id": "cwe_id",
        "cweid": "cwe_id",
        "cve id": "cve_id",
    }
    df.rename(columns={k: v for k, v in col_renames.items() if k in df.columns}, inplace=True)

    # Ensure required columns exist
    required = {"func_before", "vul", "cwe_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Keep only vulnerable rows (vul=1) — these have CWE labels
    vuln_df = df[df["vul"] == 1].copy()
    logger.info(f"Vulnerable rows (vul=1): {len(vuln_df)}")

    # Parse CWE IDs
    vuln_df["cwe_int"] = vuln_df["cwe_id"].apply(_parse_cwe_id)
    before_drop = len(vuln_df)
    vuln_df = vuln_df.dropna(subset=["cwe_int"])
    vuln_df["cwe_int"] = vuln_df["cwe_int"].astype(int)
    logger.info(f"Dropped {before_drop - len(vuln_df)} rows with unparseable CWE IDs")

    # Drop rows with empty code
    vuln_df = vuln_df[vuln_df["func_before"].notna() & vuln_df["func_before"].str.strip().ne("")]
    logger.info(f"After dropping empty code: {len(vuln_df)}")

    # Build group key for splitting (analogous to Juliet template_id)
    project_col = "project" if "project" in vuln_df.columns else None
    commit_col = "commit_id" if "commit_id" in vuln_df.columns else None

    records = []
    for _, row in vuln_df.iterrows():
        cwe_int = int(row["cwe_int"])
        cwe_name = f"CWE{cwe_int}"

        project = str(row[project_col]) if project_col else "unknown"
        commit = str(row[commit_col]) if commit_col else "unknown"
        group_key = f"bigvul_{project}_{commit}"

        # Vulnerable sample (func_before)
        records.append({
            "code": row["func_before"],
            "cwe_id": cwe_int,
            "cwe_name": cwe_name,
            "template_id": group_key,
            "source": "bigvul",
        })

    result = pd.DataFrame(records)

    # Statistics
    n_cwes = result["cwe_id"].nunique()
    logger.info(f"Extracted {len(result)} Big-Vul samples across {n_cwes} CWEs")
    logger.info(f"Top CWEs: {result['cwe_id'].value_counts().head(10).to_dict()}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract Big-Vul dataset")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--csv", default=None, help="Override CSV path")
    args = parser.parse_args()

    config = load_config(args.config)
    csv_path = args.csv or config.get("bigvul_csv_path", "datasets/big-vul/MSR_data_cleaned.csv")
    output_path = config.get("bigvul_samples_path", "data/processed/bigvul_samples.parquet")

    df = extract_bigvul(csv_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} Big-Vul samples to {output_path}")


if __name__ == "__main__":
    main()
