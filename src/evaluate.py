"""Evaluation and error analysis for the CWE classifier.

Generates:
- Overall metrics (accuracy, macro precision/recall/F1)
- Per-class classification report
- Top confused CWE pairs
"""

import argparse
import json
import os
import sys

import truststore
truststore.inject_into_ssl()

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import get_dataloaders
from src.model import CWEClassifier
from src.utils import load_config, setup_logging, get_device, load_label_map


def predict_all(model, dataloader, device):
    """Run inference on all samples, return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return all_preds, all_labels


def get_confused_pairs(y_true, y_pred, label_names, top_k=20):
    """Find the most frequently confused CWE pairs."""
    cm = confusion_matrix(y_true, y_pred)
    pairs = []

    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                pairs.append({
                    "actual": label_names[i],
                    "predicted": label_names[j],
                    "count": int(cm[i][j]),
                })

    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Evaluate CWE classifier")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (default: best.pt)")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging()
    device = get_device()

    # Load label map (int_label → CWE name)
    label_map = load_label_map(config["label_map_path"])
    inv_label_map = {v: k for k, v in label_map.items()}
    label_names = [f"CWE-{inv_label_map[i]}" for i in range(len(label_map))]

    # Load model
    checkpoint_path = args.checkpoint or os.path.join(config["checkpoint_dir"], "best.pt")
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    model = CWEClassifier(
        model_name=config["model_name"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    logger.info(f"Checkpoint epoch: {ckpt['epoch']}, F1: {ckpt.get('test_f1', 'N/A')}")

    # Load test data
    _, test_loader, _ = get_dataloaders(config)

    # Predict
    all_preds, all_labels = predict_all(model, test_loader, device)

    # Overall metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }

    logger.info("--- Overall Metrics ---")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Per-class report — only include labels that appear in test set
    present_labels = sorted(set(all_labels) | set(all_preds))
    present_names = [label_names[i] for i in present_labels]
    report = classification_report(
        all_labels, all_preds,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )

    # Confused pairs
    confused = get_confused_pairs(all_labels, all_preds, label_names)

    # Save outputs
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    pd.DataFrame(confused).to_csv("outputs/confusion_pairs.csv", index=False)

    logger.info(f"Saved metrics to outputs/metrics.json")
    logger.info(f"Saved classification report to outputs/classification_report.txt")
    logger.info(f"Saved confusion pairs to outputs/confusion_pairs.csv")

    # Print summary
    print("\n--- Classification Report (abbreviated) ---")
    print(report[:2000])

    print("\n--- Top 10 Confused Pairs ---")
    for pair in confused[:10]:
        print(f"  {pair['actual']} -> {pair['predicted']}: {pair['count']}")


if __name__ == "__main__":
    main()
