"""QLoRA training loop for CWE classification using decoder models.

Fine-tunes CodeGemma (or similar causal LMs) with 4-bit quantization and LoRA
adapters for 118-class CWE classification via AutoModelForSequenceClassification.

Supports CUDA acceleration, mixed precision (bf16), gradient checkpointing,
early stopping, checkpoint saving/resumption, and TensorBoard logging.
"""

import argparse
import gc
import json
import os
import sys
import time

import truststore
truststore.inject_into_ssl()

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import load_qlora_classifier
from src.utils import load_config, setup_logging, set_seed, count_parameters


def _get_qlora_dataloaders(config: dict, tokenizer):
    """Create train/test DataLoaders using the QLoRA model's tokenizer."""
    import pandas as pd
    from src.data.dataset import CWEDataset
    from torch.utils.data import DataLoader

    qlora_cfg = config["qlora"]

    train_df = pd.read_parquet(config["train_path"])
    test_df = pd.read_parquet(config["test_path"])

    train_dataset = CWEDataset(
        codes=train_df["code"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=config["max_length"],
    )
    test_dataset = CWEDataset(
        codes=test_df["code"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=config["max_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=qlora_cfg.get("per_device_batch_size", 2),
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=qlora_cfg.get("per_device_batch_size", 2),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
    )

    return train_loader, test_loader


def evaluate(model, dataloader, device):
    """Run evaluation and return loss, accuracy, macro-F1."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * labels.size(0)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    avg_loss = total_loss / n
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, macro_f1


def train(config: dict):
    logger = setup_logging(log_file=os.path.join(config["log_dir"], "train_qlora.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    set_seed(config["seed"])

    qlora_cfg = config["qlora"]

    # Find the QLoRA model entry in available_models
    qlora_model_entry = None
    for m in config.get("available_models", []):
        if m.get("qlora", False):
            qlora_model_entry = m
            break
    if qlora_model_entry is None:
        raise ValueError("No model with 'qlora: true' found in available_models")

    model_id = qlora_model_entry["model_id"]
    logger.info(f"QLoRA model: {model_id}")

    # Load model with QLoRA
    logger.info("Loading model with QLoRA...")
    model, tokenizer = load_qlora_classifier(
        model_id=model_id,
        num_classes=config["num_classes"],
        qlora_config=qlora_cfg,
    )

    params = count_parameters(model)
    logger.info(
        f"Model parameters — total: {params['total']:,}, "
        f"trainable: {params['trainable']:,} "
        f"({params['trainable'] / params['total'] * 100:.2f}%)"
    )
    model.print_trainable_parameters()

    # Data
    logger.info("Loading data...")
    train_loader, test_loader = _get_qlora_dataloaders(config, tokenizer)
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Optimizer — only trainable parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=qlora_cfg.get("learning_rate", 2e-4),
        weight_decay=config.get("weight_decay", 0.01),
    )

    # Try 8-bit paged optimizer if bitsandbytes supports it
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=qlora_cfg.get("learning_rate", 2e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )
        logger.info("Using PagedAdamW8bit optimizer")
    except (ImportError, AttributeError):
        logger.info("Using standard AdamW optimizer")

    # Mixed precision (bf16 preferred for QLoRA)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Mixed precision: {use_amp} (dtype={amp_dtype})")

    # TensorBoard
    writer = SummaryWriter(log_dir=config["log_dir"])

    # Checkpoint directory
    model_variant = model_id.split("/")[-1] + "-qlora"
    model_checkpoint_dir = os.path.join(config["checkpoint_dir"], model_variant)
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    # Training config
    epochs = qlora_cfg.get("epochs", 5)
    accum_steps = qlora_cfg.get("gradient_accumulation_steps", 4)
    log_every = config.get("log_every_n_steps", 100)
    patience = qlora_cfg.get("patience", 2)

    # Resume from checkpoint
    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0

    state_path = os.path.join(model_checkpoint_dir, "latest.pt")
    if os.path.exists(state_path):
        logger.info(f"Resuming from {state_path}")
        ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)

        # Restore adapter weights
        adapter_path = os.path.join(model_checkpoint_dir, "latest_adapter")
        if os.path.exists(adapter_path):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model.base_model.model, adapter_path)
            logger.info(f"Restored adapter from {adapter_path}")

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info(f"Resumed at epoch {start_epoch}, best F1: {best_f1:.4f}")

    # Training loop
    training_start = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        optimizer.zero_grad()

        for step, batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if use_amp:
                with autocast("cuda", dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / accum_steps
                loss.backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accum_steps
                loss.backward()

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accum_steps
            global_step = epoch * len(train_loader) + step

            if (step + 1) % log_every == 0:
                avg = epoch_loss / (step + 1)
                writer.add_scalar("qlora/train_loss", avg, global_step)
                progress.set_postfix(loss=f"{avg:.4f}")

        # Flush remaining gradients
        if len(train_loader) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_time = time.time() - epoch_start
        total_training_time = time.time() - training_start
        train_avg_loss = epoch_loss / len(train_loader)

        # Evaluate
        logger.info("Evaluating on test set...")
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, device)

        logger.info(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_avg_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Test Macro-F1: {test_f1:.4f} | "
            f"Time: {epoch_time:.0f}s"
        )

        writer.add_scalar("qlora/test_loss", test_loss, epoch)
        writer.add_scalar("qlora/test_accuracy", test_acc, epoch)
        writer.add_scalar("qlora/test_macro_f1", test_f1, epoch)

        # Save checkpoint — adapter weights via PEFT, training state as .pt
        model.save_pretrained(os.path.join(model_checkpoint_dir, "latest_adapter"))

        ckpt_data = {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": max(best_f1, test_f1),
            "patience_counter": patience_counter,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "config": config,
            "training_time": total_training_time,
            "model_id": model_id,
        }
        torch.save(ckpt_data, os.path.join(model_checkpoint_dir, "latest.pt"))

        # Best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            ckpt_data["best_f1"] = best_f1
            model.save_pretrained(os.path.join(model_checkpoint_dir, "best_adapter"))
            torch.save(ckpt_data, os.path.join(model_checkpoint_dir, "best.pt"))
            logger.info(f"New best model saved (Macro-F1: {best_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break

        # Free memory between epochs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()
    logger.info(f"Training complete. Best Macro-F1: {best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train CWE classifier with QLoRA")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
