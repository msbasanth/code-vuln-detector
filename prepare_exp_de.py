"""Prepare datasets for Experiment D (Big-Vul only) and Experiment E (Union combined).

Experiment D: Big-Vul samples only, filtered to CWEs with >=5 samples.
Experiment E: Juliet + Big-Vul UNION — all CWEs from both datasets.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.data.preprocess import preprocess, preprocess_bigvul
from src.data.split import template_aware_split, build_label_map
from src.utils import setup_logging, save_label_map


MIN_SAMPLES_PER_CWE = 5


def prepare_exp_d():
    """Prepare Big-Vul only dataset (Exp D)."""
    logger = setup_logging()
    logger.info("=== Preparing Experiment D: Big-Vul Only ===")

    bv = pd.read_parquet("data/processed/bigvul_samples.parquet")
    logger.info(f"Loaded {len(bv)} Big-Vul samples, {bv['cwe_id'].nunique()} CWEs")

    # Filter to CWEs with enough samples for a meaningful train/test split
    cwe_counts = bv["cwe_id"].value_counts()
    valid_cwes = cwe_counts[cwe_counts >= MIN_SAMPLES_PER_CWE].index
    dropped_cwes = cwe_counts[cwe_counts < MIN_SAMPLES_PER_CWE]
    bv = bv[bv["cwe_id"].isin(valid_cwes)].reset_index(drop=True)
    logger.info(
        f"Filtered to CWEs with >={MIN_SAMPLES_PER_CWE} samples: "
        f"{len(bv)} samples, {bv['cwe_id'].nunique()} CWEs "
        f"(dropped {len(dropped_cwes)} CWEs with {dropped_cwes.sum()} samples)"
    )

    # Preprocess code
    logger.info("Preprocessing Big-Vul code...")
    bv["code"] = bv["code"].apply(preprocess_bigvul)
    bv = bv[bv["code"].str.strip().ne("")].reset_index(drop=True)
    logger.info(f"After preprocessing: {len(bv)} samples")

    # Build label map and assign labels
    label_map = build_label_map(bv)
    bv["label"] = bv["cwe_id"].astype(str).map(label_map)

    # Template-aware split
    train_df, test_df = template_aware_split(bv, test_size=0.2, seed=42)

    # Save
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_parquet("data/processed/bigvul_train.parquet", index=False)
    test_df.to_parquet("data/processed/bigvul_test.parquet", index=False)
    save_label_map(label_map, "data/processed/bigvul_label_map.json")

    print(f"\n--- Exp D: Big-Vul Only ---")
    print(f"Total: {len(bv)} samples, {len(label_map)} CWEs")
    print(f"Train: {len(train_df)} ({100 * len(train_df) / len(bv):.1f}%)")
    print(f"Test:  {len(test_df)} ({100 * len(test_df) / len(bv):.1f}%)")
    print(f"CWEs in train: {train_df['cwe_id'].nunique()}")
    print(f"CWEs in test:  {test_df['cwe_id'].nunique()}")


def prepare_exp_e():
    """Prepare Juliet + Big-Vul UNION dataset (Exp E).

    Reuses already-preprocessed Juliet train/test parquets (from Exp A split)
    and already-preprocessed Big-Vul train/test parquets (from Exp D) to avoid
    re-preprocessing 105K+ Juliet samples from scratch.
    """
    logger = setup_logging()
    logger.info("=== Preparing Experiment E: Juliet + Big-Vul Union ===")

    # Load already-preprocessed and split Juliet data
    juliet_train = pd.read_parquet("data/processed/train.parquet")
    juliet_test = pd.read_parquet("data/processed/test.parquet")
    if "source" not in juliet_train.columns:
        juliet_train["source"] = "juliet"
        juliet_test["source"] = "juliet"

    # Load already-preprocessed and split Big-Vul data (from Exp D — all 88 CWEs)
    # Re-preprocess from raw to include ALL CWEs (Exp D filtered to >=5)
    bigvul = pd.read_parquet("data/processed/bigvul_samples.parquet")
    if "source" not in bigvul.columns:
        bigvul["source"] = "bigvul"

    logger.info(f"Juliet train: {len(juliet_train)}, test: {len(juliet_test)}, "
                f"CWEs: {juliet_train['cwe_id'].nunique()}")
    logger.info(f"Big-Vul: {len(bigvul)} samples, {bigvul['cwe_id'].nunique()} CWEs")

    # Preprocess Big-Vul only (Juliet already preprocessed)
    logger.info("Preprocessing Big-Vul samples...")
    bigvul["code"] = bigvul["code"].apply(preprocess_bigvul)
    bigvul = bigvul[bigvul["code"].str.strip().ne("")]

    # Split Big-Vul by template_aware_split (all CWEs, no min-sample filter)
    bv_label_map_tmp = build_label_map(bigvul)
    bigvul["label"] = bigvul["cwe_id"].astype(str).map(bv_label_map_tmp)
    bv_train, bv_test = template_aware_split(bigvul, test_size=0.2, seed=42)
    bv_train = bv_train.drop(columns=["label"])
    bv_test = bv_test.drop(columns=["label"])

    logger.info(f"Big-Vul split: train={len(bv_train)}, test={len(bv_test)}")

    # Union merge — combine Juliet splits with Big-Vul splits
    cols = ["code", "cwe_id", "cwe_name", "template_id", "source"]

    # Drop 'label' from Juliet if present (will rebuild with union label map)
    for c in ["label", "file_path"]:
        if c in juliet_train.columns:
            juliet_train = juliet_train.drop(columns=[c])
        if c in juliet_test.columns:
            juliet_test = juliet_test.drop(columns=[c])

    merged_train = pd.concat([juliet_train[cols], bv_train[cols]], ignore_index=True)
    merged_test = pd.concat([juliet_test[cols], bv_test[cols]], ignore_index=True)
    merged = pd.concat([merged_train, merged_test], ignore_index=True)

    logger.info(
        f"Union: {len(merged)} samples, {merged['cwe_id'].nunique()} CWEs "
        f"(Juliet: {juliet_train['cwe_id'].nunique()}, Big-Vul: {bigvul['cwe_id'].nunique()})"
    )

    # Build union label map from all data
    label_map = build_label_map(merged)
    merged_train["label"] = merged_train["cwe_id"].astype(str).map(label_map)
    merged_test["label"] = merged_test["cwe_id"].astype(str).map(label_map)

    # Save
    os.makedirs("data/processed", exist_ok=True)
    merged.to_parquet("data/processed/union_samples.parquet", index=False)
    merged_train.to_parquet("data/processed/union_train.parquet", index=False)
    merged_test.to_parquet("data/processed/union_test.parquet", index=False)
    save_label_map(label_map, "data/processed/union_label_map.json")

    print(f"\n--- Exp E: Juliet + Big-Vul Union ---")
    print(f"Total: {len(merged)} samples, {len(label_map)} CWEs")
    print(f"Train: {len(merged_train)} ({100 * len(merged_train) / len(merged):.1f}%)")
    print(f"Test:  {len(merged_test)} ({100 * len(merged_test) / len(merged):.1f}%)")
    print(f"CWEs in train: {merged_train['cwe_id'].nunique()}")
    print(f"CWEs in test:  {merged_test['cwe_id'].nunique()}")
    print(f"Sources (train): {merged_train['source'].value_counts().to_dict()}")
    print(f"Sources (test):  {merged_test['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    import sys
    if "--exp-e-only" in sys.argv:
        prepare_exp_e()
    else:
        prepare_exp_d()
        print()
        prepare_exp_e()
