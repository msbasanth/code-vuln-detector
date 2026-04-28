"""Merge Juliet and Big-Vul datasets for combined training.

Two modes:
  - overlap_only: keep only CWEs present in both datasets (118 classes max)
  - expanded:     keep all CWEs from both datasets (more classes)

Applies preprocessing appropriate to each source (skips Juliet-specific
guard removal for Big-Vul samples).
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.data.preprocess import preprocess, preprocess_bigvul
from src.utils import load_config, setup_logging


def merge_datasets(
    juliet_path: str,
    bigvul_path: str,
    mode: str = "overlap_only",
) -> pd.DataFrame:
    """Merge Juliet and Big-Vul into a single DataFrame.

    Args:
        juliet_path: path to juliet samples.parquet
        bigvul_path: path to bigvul_samples.parquet
        mode: 'overlap_only' or 'expanded'

    Returns:
        Merged DataFrame with columns: code, cwe_id, cwe_name, template_id, source
    """
    logger = setup_logging()

    juliet = pd.read_parquet(juliet_path)
    bigvul = pd.read_parquet(bigvul_path)

    # Ensure source column
    if "source" not in juliet.columns:
        juliet["source"] = "juliet"
    if "source" not in bigvul.columns:
        bigvul["source"] = "bigvul"

    logger.info(f"Juliet: {len(juliet)} samples, {juliet['cwe_id'].nunique()} CWEs")
    logger.info(f"Big-Vul: {len(bigvul)} samples, {bigvul['cwe_id'].nunique()} CWEs")

    if mode == "overlap_only":
        shared_cwes = set(juliet["cwe_id"].unique()) & set(bigvul["cwe_id"].unique())
        bigvul = bigvul[bigvul["cwe_id"].isin(shared_cwes)]
        juliet = juliet[juliet["cwe_id"].isin(shared_cwes)]
        logger.info(f"Overlap-only mode: {len(shared_cwes)} shared CWEs")
        logger.info(f"Juliet after filter: {len(juliet)} samples")
        logger.info(f"Big-Vul after filter: {len(bigvul)} samples")
    elif mode == "expanded":
        logger.info("Expanded mode: keeping all CWEs from both datasets")
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'overlap_only' or 'expanded'.")

    # Preprocess code per source
    logger.info("Preprocessing Juliet samples...")
    juliet["code"] = juliet["code"].apply(preprocess)

    logger.info("Preprocessing Big-Vul samples...")
    bigvul["code"] = bigvul["code"].apply(preprocess_bigvul)

    # Remove empty samples after preprocessing
    juliet = juliet[juliet["code"].str.strip().ne("")]
    bigvul = bigvul[bigvul["code"].str.strip().ne("")]

    # Align columns
    cols = ["code", "cwe_id", "cwe_name", "template_id", "source"]
    merged = pd.concat([juliet[cols], bigvul[cols]], ignore_index=True)

    logger.info(f"Merged dataset: {len(merged)} samples, {merged['cwe_id'].nunique()} CWEs")
    logger.info(f"  Juliet: {(merged['source'] == 'juliet').sum()}")
    logger.info(f"  Big-Vul: {(merged['source'] == 'bigvul').sum()}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge Juliet + Big-Vul datasets")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--mode", default=None, help="Override dataset_mode from config")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment = config.get("experiment", {})
    mode = args.mode or experiment.get("dataset_mode", "overlap_only")

    juliet_path = config["samples_path"]
    bigvul_path = experiment.get("bigvul_samples_path", "data/processed/bigvul_samples.parquet")
    output_path = experiment.get(
        "combined_samples_path",
        f"data/processed/combined_{mode}_samples.parquet",
    )

    merged = merge_datasets(juliet_path, bigvul_path, mode=mode)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_parquet(output_path, index=False)

    print(f"\nSaved {len(merged)} merged samples to {output_path}")
    print(f"CWE classes: {merged['cwe_id'].nunique()}")
    print(f"Sources: {merged['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
