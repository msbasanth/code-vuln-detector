"""Training loop for the CWE vulnerability classifier.

Supports CUDA acceleration, mixed precision (fp16), early stopping,
checkpoint saving/resumption, and TensorBoard logging.
"""

import argparse
import json
import os
import sys
import time

import truststore
truststore.inject_into_ssl()

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import get_dataloaders
from src.model import CWEClassifier
from src.utils import load_config, setup_logging, set_seed, get_device, count_parameters


def evaluate(model, dataloader, device):
    """Run evaluation and return loss, accuracy, macro-F1."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    avg_loss = total_loss / n
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, macro_f1


def train(config: dict):
    logger = setup_logging(log_file=os.path.join(config["log_dir"], "train.log"))
    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(config["seed"])

    # Data
    logger.info("Loading data...")
    train_loader, test_loader, tokenizer = get_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Model
    num_classes = config["num_classes"]
    model = CWEClassifier(
        model_name=config["model_name"],
        num_classes=num_classes,
        dropout=config["dropout"],
    ).to(device)

    params = count_parameters(model)
    logger.info(f"Model parameters — total: {params['total']:,}, trainable: {params['trainable']:,}")

    # Optimizer & loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    # Mixed precision
    use_fp16 = config.get("fp16", False) and device.type == "cuda"
    scaler = GradScaler("cuda") if use_fp16 else None
    logger.info(f"Mixed precision (fp16): {use_fp16}")

    # TensorBoard
    writer = SummaryWriter(log_dir=config["log_dir"])

    # Checkpoint directory (model-specific)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    model_variant = config["model_name"].split("/")[-1]
    model_checkpoint_dir = os.path.join(config["checkpoint_dir"], model_variant)
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0

    checkpoint_path = os.path.join(model_checkpoint_dir, "latest.pt")
    if os.path.exists(checkpoint_path):
        logger.info(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        logger.info(f"Resumed at epoch {start_epoch}, best F1: {best_f1:.4f}")

    # Training loop
    accum_steps = config.get("gradient_accumulation_steps", 1)
    log_every = config.get("log_every_n_steps", 100)
    training_start = time.time()

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{config['epochs']}",
        )

        optimizer.zero_grad()

        for step, batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if use_fp16:
                with autocast("cuda"):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    loss = loss / accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / accum_steps

                loss.backward()

                if (step + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item() * accum_steps
            global_step = epoch * len(train_loader) + step

            if (step + 1) % log_every == 0:
                avg = epoch_loss / (step + 1)
                writer.add_scalar("train/loss", avg, global_step)
                progress.set_postfix(loss=f"{avg:.4f}")

        epoch_time = time.time() - epoch_start
        total_training_time = time.time() - training_start
        train_avg_loss = epoch_loss / len(train_loader)

        # Evaluate
        logger.info("Evaluating on test set...")
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, device)

        logger.info(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_avg_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Test Macro-F1: {test_f1:.4f} | "
            f"Time: {epoch_time:.0f}s"
        )

        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/accuracy", test_acc, epoch)
        writer.add_scalar("test/macro_f1", test_f1, epoch)

        # Save checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": max(best_f1, test_f1),
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "config": config,
            "training_time": total_training_time,
        }
        if scaler:
            ckpt_data["scaler_state_dict"] = scaler.state_dict()

        torch.save(ckpt_data, os.path.join(model_checkpoint_dir, "latest.pt"))

        # Best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(ckpt_data, os.path.join(model_checkpoint_dir, "best.pt"))
            logger.info(f"New best model saved (Macro-F1: {best_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{config['patience']}")

        if patience_counter >= config["patience"]:
            logger.info("Early stopping triggered")
            break

    writer.close()
    logger.info(f"Training complete. Best Macro-F1: {best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train CWE classifier")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
