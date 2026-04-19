"""Parse the Juliet Test Suite into a labeled dataset of code samples.

Each .c/.cpp file in the testcases/ directory becomes one sample,
labeled by its parent CWE directory. Template IDs are extracted from
filenames for template-aware splitting.
"""

import os
import re
import sys
import argparse

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils import load_config, setup_logging

# Regex from Juliet's py_common.py (adapted for Python named groups)
TESTCASE_RE = re.compile(
    r"^(?P<cwe_dir>CWE(?P<cwe_number>\d+)_(?P<cwe_name>.*))"
    r"__"
    r"(?P<functional_variant>.*)"
    r"_"
    r"(?P<flow_variant>\d+)"
    r"_?"
    r"(?P<subfile>[a-z]|bad|good\d+|base|goodB2G|goodG2B)?"
    r"\.(?P<ext>c|cpp|h)$",
    re.IGNORECASE,
)


def find_cwe_directories(testcases_root: str) -> list[str]:
    """Find all CWE directories under testcases root."""
    cwe_dirs = []
    for name in sorted(os.listdir(testcases_root)):
        full = os.path.join(testcases_root, name)
        if os.path.isdir(full) and name.startswith("CWE"):
            cwe_dirs.append(full)
    return cwe_dirs


def collect_source_files(cwe_dir: str) -> list[str]:
    """Collect all .c/.cpp files from a CWE directory (including s01-s09 subdirs)."""
    files = []
    entries = os.listdir(cwe_dir)

    # Check if this directory uses s01/s02/... subdirectories
    has_subdirs = any(
        e.startswith("s") and e[1:].isdigit() and os.path.isdir(os.path.join(cwe_dir, e))
        for e in entries
    )

    if has_subdirs:
        for entry in sorted(entries):
            subdir = os.path.join(cwe_dir, entry)
            if os.path.isdir(subdir) and entry.startswith("s") and entry[1:].isdigit():
                for f in os.listdir(subdir):
                    if f.endswith((".c", ".cpp", ".h")):
                        files.append(os.path.join(subdir, f))
    else:
        for f in entries:
            if f.endswith((".c", ".cpp", ".h")):
                files.append(os.path.join(cwe_dir, f))

    return files


def parse_filename(filepath: str) -> dict | None:
    """Extract CWE label and template ID from a testcase filename.

    Returns None if the filename doesn't match the testcase pattern
    (e.g., support files, headers).
    """
    filename = os.path.basename(filepath)
    match = TESTCASE_RE.match(filename)
    if not match:
        return None

    cwe_dir_name = match.group("cwe_dir")
    functional_variant = match.group("functional_variant")
    cwe_id = int(match.group("cwe_number"))

    # Template = CWE number + functional variant
    # All flow variants of the same template share this ID
    template_id = f"CWE{cwe_id}__{functional_variant}"

    return {
        "cwe_id": cwe_id,
        "cwe_name": cwe_dir_name,
        "template_id": template_id,
        "flow_variant": match.group("flow_variant"),
        "subfile": match.group("subfile"),
        "extension": match.group("ext"),
    }


def extract_samples(testcases_root: str) -> pd.DataFrame:
    """Walk the Juliet testcases directory and extract all samples."""
    logger = setup_logging()
    logger.info(f"Scanning testcases at: {testcases_root}")

    cwe_dirs = find_cwe_directories(testcases_root)
    logger.info(f"Found {len(cwe_dirs)} CWE directories")

    records = []
    skipped = 0

    for cwe_dir in tqdm(cwe_dirs, desc="Processing CWE directories"):
        source_files = collect_source_files(cwe_dir)

        for filepath in source_files:
            parsed = parse_filename(filepath)
            if parsed is None:
                skipped += 1
                continue

            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    code = f.read()
            except Exception as e:
                logger.warning(f"Failed to read {filepath}: {e}")
                skipped += 1
                continue

            records.append({
                "file_path": os.path.relpath(filepath, testcases_root),
                "code": code,
                "cwe_id": parsed["cwe_id"],
                "cwe_name": parsed["cwe_name"],
                "template_id": parsed["template_id"],
            })

    logger.info(f"Extracted {len(records)} samples, skipped {skipped} files")

    df = pd.DataFrame(records)

    # Report statistics
    n_cwes = df["cwe_id"].nunique()
    n_templates = df["template_id"].nunique()
    logger.info(f"Unique CWE categories: {n_cwes}")
    logger.info(f"Unique templates: {n_templates}")
    logger.info(f"Largest class: CWE-{df['cwe_id'].value_counts().index[0]} "
                f"({df['cwe_id'].value_counts().iloc[0]} samples)")
    logger.info(f"Smallest class: CWE-{df['cwe_id'].value_counts().index[-1]} "
                f"({df['cwe_id'].value_counts().iloc[-1]} samples)")

    return df


def main():
    parser = argparse.ArgumentParser(description="Extract samples from Juliet Test Suite")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    testcases_root = config["dataset_root"]
    output_path = config["samples_path"]

    df = extract_samples(testcases_root)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")


if __name__ == "__main__":
    main()
