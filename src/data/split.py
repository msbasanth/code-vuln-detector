"""Template-aware dataset splitting for the Juliet Test Suite.

Implements the key contribution from the paper: splitting by template groups
(not individual samples) to prevent structural data leakage.

Also supports combined Juliet + Big-Vul datasets, where Big-Vul samples
are grouped by project+commit to prevent commit-level leakage.
"""

import json
import os
import sys
import argparse

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.data.preprocess import preprocess
from src.utils import load_config, setup_logging, save_label_map


def template_aware_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset ensuring no template appears in both train and test.

    For each CWE category, templates are randomly assigned to train or test.
    All samples from the same template go to the same split.

    CWEs with only 1 template are placed entirely in the train set.
    """
    logger = setup_logging()

    train_indices = []
    test_indices = []

    for cwe_id, cwe_group in df.groupby("cwe_id"):
        templates = cwe_group["template_id"].unique()

        if len(templates) < 2:
            # Only 1 template → all goes to train
            train_indices.extend(cwe_group.index.tolist())
            logger.warning(
                f"CWE-{cwe_id}: only {len(templates)} template(s), "
                f"all {len(cwe_group)} samples go to train"
            )
            continue

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(
            splitter.split(cwe_group, groups=cwe_group["template_id"])
        )

        train_indices.extend(cwe_group.index[train_idx].tolist())
        test_indices.extend(cwe_group.index[test_idx].tolist())

    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)

    # Validate: no template overlap
    train_templates = set(train_df["template_id"].unique())
    test_templates = set(test_df["template_id"].unique())
    overlap = train_templates & test_templates
    assert len(overlap) == 0, f"Template leakage detected! {len(overlap)} overlapping templates"

    logger.info(f"Train: {len(train_df)} samples, {train_df['cwe_id'].nunique()} CWEs, "
                f"{len(train_templates)} templates")
    logger.info(f"Test:  {len(test_df)} samples, {test_df['cwe_id'].nunique()} CWEs, "
                f"{len(test_templates)} templates")

    return train_df, test_df


def build_label_map(df: pd.DataFrame) -> dict:
    """Create a mapping from CWE ID to integer label (sorted for consistency)."""
    cwe_ids = sorted(df["cwe_id"].unique())
    return {str(cwe): idx for idx, cwe in enumerate(cwe_ids)}


def main():
    parser = argparse.ArgumentParser(description="Template-aware dataset split")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--combined", default=None,
                        help="Path to combined (merged) samples parquet. "
                             "If provided, splits the combined dataset instead of Juliet-only.")
    parser.add_argument("--output-prefix", default=None,
                        help="Output file prefix (e.g. 'juliet19' → juliet19_train.parquet). "
                             "Overrides default output paths for experiment isolation.")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging()

    experiment = config.get("experiment", {})

    # Determine input: combined dataset or Juliet-only
    if args.combined:
        samples_path = args.combined
    else:
        samples_path = config["samples_path"]

    logger.info(f"Loading samples from {samples_path}")
    df = pd.read_parquet(samples_path)
    logger.info(f"Loaded {len(df)} samples")

    # Preprocess only if source column is absent (Juliet-only, raw samples)
    # Combined datasets are already preprocessed by merge_datasets.py
    if "source" not in df.columns:
        logger.info("Preprocessing code samples (Juliet-only mode)...")
        df["code"] = df["code"].apply(preprocess)
        df["source"] = "juliet"

    # Remove empty samples after preprocessing
    empty_mask = df["code"].str.strip().eq("")
    if empty_mask.any():
        logger.warning(f"Removing {empty_mask.sum()} empty samples after preprocessing")
        df = df[~empty_mask].reset_index(drop=True)

    # Build label map
    label_map = build_label_map(df)
    df["label"] = df["cwe_id"].astype(str).map(label_map)
    logger.info(f"Label map: {len(label_map)} classes")

    # Template-aware split
    train_df, test_df = template_aware_split(df, test_size=0.2, seed=config["seed"])

    # Determine output paths
    if args.output_prefix:
        prefix = args.output_prefix
        train_path = f"data/processed/{prefix}_train.parquet"
        test_path = f"data/processed/{prefix}_test.parquet"
        label_map_path = f"data/processed/{prefix}_label_map.json"
    elif args.combined:
        train_path = experiment.get("combined_train_path", "data/processed/combined_train.parquet")
        test_path = experiment.get("combined_test_path", "data/processed/combined_test.parquet")
        label_map_path = experiment.get("combined_label_map_path", "data/processed/combined_label_map.json")
    else:
        train_path = config["train_path"]
        test_path = config["test_path"]
        label_map_path = config["label_map_path"]

    # Save outputs
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    save_label_map(label_map, label_map_path)

    logger.info(f"Saved train set to {train_path}")
    logger.info(f"Saved test set to {test_path}")
    logger.info(f"Saved label map to {label_map_path}")

    # Summary statistics
    print("\n--- Split Summary ---")
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)} ({100*len(train_df)/len(df):.1f}%)")
    print(f"Test samples:  {len(test_df)} ({100*len(test_df)/len(df):.1f}%)")
    print(f"CWE classes:   {len(label_map)}")
    print(f"CWEs in train: {train_df['cwe_id'].nunique()}")
    print(f"CWEs in test:  {test_df['cwe_id'].nunique()}")
    if "source" in train_df.columns:
        print(f"Train sources: {train_df['source'].value_counts().to_dict()}")
        print(f"Test sources:  {test_df['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
