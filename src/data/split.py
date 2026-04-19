"""Template-aware dataset splitting for the Juliet Test Suite.

Implements the key contribution from the paper: splitting by template groups
(not individual samples) to prevent structural data leakage.
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
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging()

    # Load extracted samples
    samples_path = config["samples_path"]
    logger.info(f"Loading samples from {samples_path}")
    df = pd.read_parquet(samples_path)
    logger.info(f"Loaded {len(df)} samples")

    # Preprocess code
    logger.info("Preprocessing code samples...")
    df["code"] = df["code"].apply(preprocess)

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

    # Save outputs
    os.makedirs(os.path.dirname(config["train_path"]), exist_ok=True)

    train_df.to_parquet(config["train_path"], index=False)
    test_df.to_parquet(config["test_path"], index=False)
    save_label_map(label_map, config["label_map_path"])

    logger.info(f"Saved train set to {config['train_path']}")
    logger.info(f"Saved test set to {config['test_path']}")
    logger.info(f"Saved label map to {config['label_map_path']}")

    # Summary statistics
    print("\n--- Split Summary ---")
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)} ({100*len(train_df)/len(df):.1f}%)")
    print(f"Test samples:  {len(test_df)} ({100*len(test_df)/len(df):.1f}%)")
    print(f"CWE classes:   {len(label_map)}")
    print(f"CWEs in train: {train_df['cwe_id'].nunique()}")
    print(f"CWEs in test:  {test_df['cwe_id'].nunique()}")


if __name__ == "__main__":
    main()
