"""Quick script to compare checkpoint metrics across all trained models."""
import torch

models = ["codet5-small", "codet5-base", "codebert-base"]

print(f"{'Model':<16} {'Checkpoint':<8} {'Epoch':<6} {'Test Loss':<10} {'Test Acc':<10} {'Macro-F1':<10} {'Best F1':<10}")
print("-" * 80)

for m in models:
    for tag in ["best", "latest"]:
        path = f"outputs/checkpoints/{m}/{tag}.pt"
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            epoch = ckpt["epoch"] + 1
            test_loss = round(ckpt["test_loss"], 4)
            test_acc = round(ckpt["test_acc"], 4)
            test_f1 = round(ckpt["test_f1"], 4)
            best_f1 = round(ckpt["best_f1"], 4)
            print(f"{m:<16} {tag:<8} {epoch:<6} {test_loss:<10} {test_acc:<10} {test_f1:<10} {best_f1:<10}")
        except FileNotFoundError:
            print(f"{m:<16} {tag:<8} NOT FOUND")
    print()
