"""Inference script for predicting CWE categories on new code files.

Usage:
    python -m src.predict path/to/code.c
    python -m src.predict path/to/directory/ --top-k 3
"""

import argparse
import os
import sys

import truststore
truststore.inject_into_ssl()

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocess import preprocess
from src.model import CWEClassifier, CWEBiLSTM, load_qlora_for_inference
from src.utils import load_config, get_device, load_label_map


def load_model(config: dict, device: torch.device):
    """Load trained model from best checkpoint."""
    model_variant = config["model_name"].split("/")[-1]
    checkpoint_path = os.path.join(config["checkpoint_dir"], model_variant, "best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. Train the model first.")

    model = CWEClassifier(
        model_name=config["model_name"],
        num_classes=config["num_classes"],
        dropout=0.0,  # No dropout during inference
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def predict_code(
    code: str,
    model: CWEClassifier,
    tokenizer,
    label_map: dict,
    device: torch.device,
    max_length: int = 256,
    top_k: int = 5,
) -> list[dict]:
    """Predict CWE category for a raw code string.

    Returns:
        List of dicts with 'cwe', 'confidence' keys, sorted descending.
    """
    code = preprocess(code)

    encoding = tokenizer(
        code,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    inv_map = {v: k for k, v in label_map.items()}

    top_probs, top_indices = probs.topk(top_k)
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        cwe_id = inv_map.get(idx, "unknown")
        results.append({
            "cwe": f"CWE-{cwe_id}",
            "confidence": round(prob, 4),
        })

    return results


def predict_file(
    filepath: str,
    model: CWEClassifier,
    tokenizer,
    label_map: dict,
    device: torch.device,
    max_length: int = 256,
    top_k: int = 5,
) -> list[dict]:
    """Predict CWE category for a single source code file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        code = f.read()
    return predict_code(code, model, tokenizer, label_map, device, max_length, top_k)


def load_qlora_model(config: dict):
    """Load QLoRA-trained model from best adapter checkpoint.

    Returns:
        (model, tokenizer) tuple.
    """
    qlora_model_entry = None
    for m in config.get("available_models", []):
        if m.get("qlora", False):
            qlora_model_entry = m
            break
    if qlora_model_entry is None:
        raise ValueError("No model with 'qlora: true' found in available_models")

    model_id = qlora_model_entry["model_id"]
    variant = model_id.split("/")[-1] + "-qlora"
    adapter_path = os.path.join(config["checkpoint_dir"], variant, "best_adapter")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"No QLoRA adapter found at {adapter_path}. "
            "Train with: python -m src.train_qlora"
        )

    model, tokenizer = load_qlora_for_inference(
        model_id=model_id,
        num_classes=config["num_classes"],
        adapter_path=adapter_path,
    )
    return model, tokenizer


def predict_code_qlora(
    code: str,
    model,
    tokenizer,
    label_map: dict,
    max_length: int = 256,
    top_k: int = 5,
) -> list[dict]:
    """Predict CWE category using a QLoRA sequence classification model.

    Returns:
        List of dicts with 'cwe', 'confidence' keys, sorted descending.
    """
    code = preprocess(code)

    encoding = tokenizer(
        code,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

    inv_map = {v: k for k, v in label_map.items()}

    top_probs, top_indices = probs.topk(top_k)
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        cwe_id = inv_map.get(idx, "unknown")
        results.append({
            "cwe": f"CWE-{cwe_id}",
            "confidence": round(prob, 4),
        })

    return results


def load_dl_model(config: dict, device: torch.device):
    """Load a trained DL model (BiLSTM) from best checkpoint.

    Returns:
        (model, tokenizer) tuple.
    """
    dl_config = config.get("dl", {})
    tokenizer_name = dl_config.get("tokenizer_name", "Salesforce/codet5-small")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model_variant = config["model_name"]  # e.g. "bilstm-attention"
    checkpoint_path = os.path.join(config["checkpoint_dir"], model_variant, "best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. Train the model first.")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = CWEBiLSTM(
        vocab_size=tokenizer.vocab_size,
        num_classes=config["num_classes"],
        embedding_dim=dl_config.get("embedding_dim", 128),
        hidden_dim=dl_config.get("hidden_dim", 256),
        num_layers=dl_config.get("num_layers", 2),
        dropout=0.0,  # No dropout during inference
        pad_idx=tokenizer.pad_token_id or 0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer


def predict_code_dl(
    code: str,
    model: CWEBiLSTM,
    tokenizer,
    label_map: dict,
    device: torch.device,
    max_length: int = 256,
    top_k: int = 5,
) -> list[dict]:
    """Predict CWE category using a DL model (BiLSTM).

    Returns:
        List of dicts with 'cwe', 'confidence' keys, sorted descending.
    """
    code = preprocess(code)

    encoding = tokenizer(
        code,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    inv_map = {v: k for k, v in label_map.items()}

    top_probs, top_indices = probs.topk(top_k)
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        cwe_id = inv_map.get(idx, "unknown")
        results.append({
            "cwe": f"CWE-{cwe_id}",
            "confidence": round(prob, 4),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Predict CWE category for source code files")
    parser.add_argument("path", help="Path to a .c/.cpp file or directory of files")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to show")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    # Load model and tokenizer
    model = load_model(config, device)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    label_map = load_label_map(config["label_map_path"])

    # Collect files
    if os.path.isfile(args.path):
        files = [args.path]
    elif os.path.isdir(args.path):
        files = [
            os.path.join(args.path, f)
            for f in sorted(os.listdir(args.path))
            if f.endswith((".c", ".cpp"))
        ]
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        sys.exit(1)

    if not files:
        print(f"No .c/.cpp files found in {args.path}")
        sys.exit(1)

    # Risk prioritizer
    from src.risk import HealthcareRiskPrioritizer
    try:
        prioritizer = HealthcareRiskPrioritizer()
    except FileNotFoundError:
        prioritizer = None

    # Predict
    for filepath in files:
        print(f"\n{'='*60}")
        print(f"File: {filepath}")
        print(f"{'='*60}")

        results = predict_file(
            filepath, model, tokenizer, label_map, device,
            max_length=config["max_length"],
            top_k=args.top_k,
        )

        for i, r in enumerate(results, 1):
            bar = "#" * int(r["confidence"] * 40)
            print(f"  {i}. {r['cwe']:>10}  {r['confidence']:.4f}  {bar}")

        # Healthcare risk assessment for top prediction
        if prioritizer and results:
            top = results[0]
            assessment = prioritizer.assess(top["cwe"], top["confidence"])
            print(f"\n  Healthcare Risk Assessment (top prediction):")
            print(f"    Priority:     {assessment.priority_level}")
            print(f"    Risk Score:   {assessment.risk_score:.4f}")
            print(f"    Threats:      {', '.join(assessment.threat_types)}")
            print(f"    Domains:      {', '.join(assessment.control_domains)}")


if __name__ == "__main__":
    main()
