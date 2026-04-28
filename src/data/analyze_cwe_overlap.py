"""Analyze CWE overlap between Juliet and Big-Vul datasets.

Produces a JSON report showing which CWEs are shared, exclusive
to each dataset, and per-CWE sample counts.
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils import load_config, setup_logging


def analyze_overlap(juliet_path: str, bigvul_path: str) -> dict:
    """Compare CWE distributions between Juliet and Big-Vul."""
    logger = setup_logging()

    juliet = pd.read_parquet(juliet_path)
    bigvul = pd.read_parquet(bigvul_path)

    juliet_cwes = set(juliet["cwe_id"].unique())
    bigvul_cwes = set(bigvul["cwe_id"].unique())

    shared = sorted(juliet_cwes & bigvul_cwes)
    juliet_only = sorted(juliet_cwes - bigvul_cwes)
    bigvul_only = sorted(bigvul_cwes - juliet_cwes)

    logger.info(f"Juliet CWEs: {len(juliet_cwes)}")
    logger.info(f"Big-Vul CWEs: {len(bigvul_cwes)}")
    logger.info(f"Shared CWEs: {len(shared)}")
    logger.info(f"Juliet-only CWEs: {len(juliet_only)}")
    logger.info(f"Big-Vul-only CWEs: {len(bigvul_only)}")

    juliet_counts = juliet["cwe_id"].value_counts().to_dict()
    bigvul_counts = bigvul["cwe_id"].value_counts().to_dict()

    per_cwe = []
    for cwe in sorted(juliet_cwes | bigvul_cwes):
        per_cwe.append({
            "cwe_id": int(cwe),
            "juliet_samples": int(juliet_counts.get(cwe, 0)),
            "bigvul_samples": int(bigvul_counts.get(cwe, 0)),
            "in_juliet": cwe in juliet_cwes,
            "in_bigvul": cwe in bigvul_cwes,
        })

    report = {
        "juliet_total_samples": len(juliet),
        "bigvul_total_samples": len(bigvul),
        "juliet_cwe_count": len(juliet_cwes),
        "bigvul_cwe_count": len(bigvul_cwes),
        "shared_cwe_count": len(shared),
        "juliet_only_cwe_count": len(juliet_only),
        "bigvul_only_cwe_count": len(bigvul_only),
        "shared_cwes": [int(c) for c in shared],
        "juliet_only_cwes": [int(c) for c in juliet_only],
        "bigvul_only_cwes": [int(c) for c in bigvul_only],
        "per_cwe": per_cwe,
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze CWE overlap between datasets")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)

    juliet_path = config["samples_path"]
    bigvul_path = config.get("bigvul_samples_path", "data/processed/bigvul_samples.parquet")
    output_path = config.get("cwe_overlap_report_path", "data/processed/cwe_overlap_report.json")

    report = analyze_overlap(juliet_path, bigvul_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Console summary
    print("\n--- CWE Overlap Summary ---")
    print(f"Juliet:       {report['juliet_cwe_count']} CWEs, {report['juliet_total_samples']} samples")
    print(f"Big-Vul:      {report['bigvul_cwe_count']} CWEs, {report['bigvul_total_samples']} samples")
    print(f"Shared CWEs:  {report['shared_cwe_count']}")
    print(f"Juliet-only:  {report['juliet_only_cwe_count']}")
    print(f"Big-Vul-only: {report['bigvul_only_cwe_count']}")
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
