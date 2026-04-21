"""Evaluate only Gemma 4 E2B IT and update model_comparison.json."""
import gc
import json
import os
import sys
import time
import traceback

import truststore
truststore.inject_into_ssl()

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score,
)

from src.gemma_predict import load_zero_shot_model, predict_code_zero_shot
from src.utils import load_config, load_label_map


def macro_fpr(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fprs = []
    for i in range(num_classes):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fprs.append(fp / (fp + tn) if fp + tn > 0 else 0.0)
    return np.mean(fprs)


MODEL_ID = "google/gemma-4-E2B-it"
DISPLAY_NAME = "Gemma 4 E2B IT (Zero-shot)"
N_SAMPLES = 50
NUM_BEAMS = 1  # greedy — beam search OOMs on 8GB VRAM

try:
    config = load_config("config.yaml")
    num_classes = config["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map(config["label_map_path"])

    # Load test data
    test_df = pd.read_parquet(config["test_path"], columns=["code", "cwe_id"])
    test_df = test_df.sample(N_SAMPLES, random_state=42).reset_index(drop=True)

    print(f"Evaluating {DISPLAY_NAME} on {N_SAMPLES} samples with {NUM_BEAMS} beams...", flush=True)
    zs_model, zs_tokenizer = load_zero_shot_model(MODEL_ID)

    y_pred, y_true = [], []
    latencies = []
    for idx, (_, row) in enumerate(test_df.iterrows()):
        print(f"\n[{idx+1}/{N_SAMPLES}] Starting sample...", flush=True)
        t0 = time.perf_counter()
        try:
            preds = predict_code_zero_shot(
                row["code"], zs_model, zs_tokenizer, MODEL_ID, label_map,
                top_k=1, num_beams=NUM_BEAMS,
            )
        except Exception as e:
            print(f"  Error on sample {idx+1}: {e}", flush=True)
            traceback.print_exc()
            preds = [{"cwe": "CWE-0", "confidence": 0.0}]
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)
        predicted_cwe = preds[0]["cwe"].replace("CWE-", "")
        true_cwe = str(row["cwe_id"])
        y_pred.append(label_map.get(predicted_cwe, 0))
        y_true.append(label_map.get(true_cwe, 0))
        print(f"  Done sample {idx+1}/{N_SAMPLES} in {elapsed/1000:.1f}s | pred={preds[0]['cwe']} true=CWE-{true_cwe}", flush=True)
        # Free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    fpr = macro_fpr(y_true, y_pred, num_classes)
    avg_lat = np.mean(latencies)

    n_total = pd.read_parquet(config["test_path"], columns=["cwe_id"]).shape[0]

    print(f"\nResults: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} MCC={mcc:.4f} Lat={avg_lat:.1f}ms", flush=True)

    # Update model_comparison.json
    output_path = os.path.join("outputs", "model_comparison.json")
    with open(output_path) as f:
        results = json.load(f)

    new_entry = {
        "Model": DISPLAY_NAME,
        "model_id": MODEL_ID,
        "inference_only": True,
        "eval_samples": f"sampled {N_SAMPLES}/{n_total}",
        "Parameters": "N/A",
        "Size (MB)": "N/A",
        "Best Epoch": "N/A",
        "Training Time (s)": "N/A",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "MCC": mcc,
        "FPR": fpr,
        "Latency (ms)": round(avg_lat, 2),
    }

    # Replace existing Gemma 4 entry or append
    for i, r in enumerate(results):
        if r.get("model_id") == MODEL_ID:
            results[i] = new_entry
            break
    else:
        results.append(new_entry)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Updated {output_path}", flush=True)

except Exception as e:
    print(f"\nFATAL ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
