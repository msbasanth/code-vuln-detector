"""Compare model metrics before and after Big-Vul integration.

Loads per-epoch metric logs and final evaluation metrics for each
experiment (juliet_only, overlap_only, expanded) and produces a
comparison report.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from src.utils import load_config


MODELS = [
    ("CodeT5-Small", "codet5-small"),
    ("CodeT5-Base", "codet5-base"),
    ("CodeBERT-Base", "codebert-base"),
    ("GraphCodeBERT-Base", "graphcodebert-base"),
]


def load_epoch_metrics(log_dir: str, variant: str) -> dict | None:
    """Load the per-epoch metrics JSON for a model variant."""
    path = os.path.join(log_dir, f"{variant}_epoch_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_eval_metrics(path: str) -> dict | None:
    """Load evaluation metrics JSON."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compare(config: dict, experiments: list[str]):
    """Build comparison table across experiments."""
    results = []

    for model_name, variant in MODELS:
        row = {"model": model_name}

        for exp in experiments:
            # Experiment log dirs: outputs/<exp>/logs/
            exp_log_dir = os.path.join("outputs", exp, "logs")

            data = load_epoch_metrics(exp_log_dir, variant)
            if data and "best_metrics" in data:
                bm = data["best_metrics"]
                row[f"{exp}_best_f1"] = round(bm.get("best_f1", 0), 4)
                row[f"{exp}_best_acc"] = round(bm.get("best_acc", 0), 4)
                row[f"{exp}_best_precision"] = round(bm.get("best_precision", 0), 4)
                row[f"{exp}_best_recall"] = round(bm.get("best_recall", 0), 4)
                row[f"{exp}_best_f1_epoch"] = bm.get("best_f1_epoch", "N/A")
                row[f"{exp}_epochs_trained"] = len(data.get("epoch_metrics", []))
            else:
                row[f"{exp}_best_f1"] = "N/A"
                row[f"{exp}_best_acc"] = "N/A"
                row[f"{exp}_best_precision"] = "N/A"
                row[f"{exp}_best_recall"] = "N/A"
                row[f"{exp}_best_f1_epoch"] = "N/A"
                row[f"{exp}_epochs_trained"] = "N/A"

        # Compute delta between first experiment (baseline) and others
        baseline = experiments[0]
        for exp in experiments[1:]:
            j_f1 = row.get(f"{baseline}_best_f1")
            e_f1 = row.get(f"{exp}_best_f1")
            if isinstance(j_f1, (int, float)) and isinstance(e_f1, (int, float)):
                row[f"{exp}_f1_delta"] = round(e_f1 - j_f1, 4)
            else:
                row[f"{exp}_f1_delta"] = "N/A"

        results.append(row)

    return results


def print_comparison(results: list[dict], experiments: list[str]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("Big-Vul Integration Impact — Best Metrics Comparison")
    print("=" * 90)

    # Header
    header = f"{'Model':<22}"
    for exp in experiments:
        header += f"  {exp:>16} F1"
    if len(experiments) > 1:
        for exp in experiments[1:]:
            header += f"  {exp:>12} Δ F1"
    print(header)
    print("-" * len(header))

    for row in results:
        line = f"{row['model']:<22}"
        for exp in experiments:
            val = row.get(f"{exp}_best_f1", "N/A")
            line += f"  {val:>19}" if isinstance(val, str) else f"  {val:>19.4f}"
        if len(experiments) > 1:
            for exp in experiments[1:]:
                val = row.get(f"{exp}_f1_delta", "N/A")
                if isinstance(val, (int, float)):
                    sign = "+" if val >= 0 else ""
                    line += f"  {sign}{val:>14.4f}"
                else:
                    line += f"  {val:>15}"
        print(line)

    print("=" * 90)

    # Per-model detail
    for row in results:
        print(f"\n{row['model']}:")
        for exp in experiments:
            f1 = row.get(f"{exp}_best_f1", "N/A")
            acc = row.get(f"{exp}_best_acc", "N/A")
            prec = row.get(f"{exp}_best_precision", "N/A")
            rec = row.get(f"{exp}_best_recall", "N/A")
            ep = row.get(f"{exp}_best_f1_epoch", "N/A")
            n_ep = row.get(f"{exp}_epochs_trained", "N/A")
            print(f"  {exp}: F1={f1}, Acc={acc}, Prec={prec}, Rec={rec}, "
                  f"best@epoch={ep}, trained={n_ep} epochs")


def main():
    parser = argparse.ArgumentParser(description="Compare Big-Vul integration impact")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--experiments", nargs="+",
                        default=["exp_b_juliet19", "exp_c_combined"],
                        help="Experiment names to compare (directory names under outputs/)")
    args = parser.parse_args()

    config = load_config(args.config)
    results = compare(config, args.experiments)

    print_comparison(results, args.experiments)

    # Save to JSON
    output_path = "outputs/bigvul_impact_report.json"
    os.makedirs("outputs", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
