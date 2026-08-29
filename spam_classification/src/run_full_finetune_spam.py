"""
Full fine-tuning for spam classification.

Strategy: unfreeze EVERY parameter in the pretrained GPT-2 backbone (all
~124M weights) plus a fresh classification head, and update all of them via
backprop. This is usually the strongest of the three strategies in terms of
raw accuracy - the model can genuinely reshape its internal representations
for spam detection specifically - but it's also the slowest, most
memory-hungry (Adam keeps optimizer state for every one of the 124M
parameters), and the most at risk of overfitting given how small the spam
dataset is.

Run this in Colab, with DATA_DIR pointed at your Drive folder from the
dataset-prep notebook (e.g. "/content/drive/MyDrive/llm_finetune_project/data/spam").
"""

import os
import json
import time
import torch
import torch.nn as nn
import tiktoken
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from instantiate_model import instantiate_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/content/drive/MyDrive/llm_finetune_project/data/spam"
RESULTS_DIR = "/content/drive/MyDrive/llm_finetune_project/results/spam"
CHECKPOINT_DIR = "/content/drive/MyDrive/llm_finetune_project/checkpoints/spam"

MAX_LENGTH = 128
BATCH_SIZE = 16  # smaller than frozen/LoRA: full fine-tuning also stores
                 # Adam's optimizer state for every one of the 124M params
NUM_EPOCHS = 4
LEARNING_RATE = 5e-5  # small LR: nudging already-good pretrained weights,
                       # not training fresh ones from near-zero
NUM_CLASSES = 2
PAD_TOKEN_ID = 50256  # GPT-2's <|endoftext|> token, used for right-padding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Dataset (identical to the other two scripts - same data, same tokenization)
# ---------------------------------------------------------------------------
class SpamDataset(Dataset):
    """
    Tokenizes each text, truncates/right-pads to a fixed MAX_LENGTH using
    GPT-2's <|endoftext|> token as padding. Since GPT-2 is causal, the
    representation at the LAST position of a right-padded sequence still
    correctly reflects the entire real input, so logits[:, -1, :] is safe
    to use as the classification logits without tracking per-example lengths.
    """

    def __init__(self, csv_path, tokenizer, max_length=MAX_LENGTH, pad_token_id=PAD_TOKEN_ID):
        df = pd.read_csv(csv_path)
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.texts[idx], allowed_special={"<|endoftext|>"})
        ids = ids[: self.max_length]
        pad_len = self.max_length - len(ids)
        ids = ids + [self.pad_token_id] * pad_len
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def build_full_finetune_model():
    model, cfg = instantiate_model("gpt2-small (124M)", load_weights=True)

    # No freezing here - every existing parameter stays trainable (this is
    # already the default state right after loading weights).

    # Replace the vocab-sized output head with a fresh classification head.
    # Since nothing was frozen first, this new head is trainable exactly
    # like everything else - there's no special ordering concern here,
    # unlike in the LoRA script.
    model.out_head = nn.Linear(cfg["emb_dim"], NUM_CLASSES)

    return model, cfg


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_full_checkpoint(model, path):
    """
    Saves the ENTIRE model state dict - every parameter changed during full
    fine-tuning, so (unlike the frozen/LoRA scripts) there's no smaller
    subset to save. This file will be large (~500MB for GPT-2-small) -
    worth reporting that size directly alongside the LoRA/frozen checkpoint
    sizes in your results, since the size difference is itself part of the
    comparison story.
    """
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"Saved full model checkpoint to {path} ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)

        logits = model(input_ids)[:, -1, :]
        loss = nn.functional.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids)[:, -1, :]
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = correct / total
    return accuracy, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    train_ds = SpamDataset(os.path.join(DATA_DIR, "train.csv"), tokenizer)
    val_ds = SpamDataset(os.path.join(DATA_DIR, "validation.csv"), tokenizer)
    test_ds = SpamDataset(os.path.join(DATA_DIR, "test.csv"), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model, cfg = build_full_finetune_model()
    model.to(DEVICE)

    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_acc, _, _ = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | train loss {train_loss:.4f} | val acc {val_acc:.4f}")
    train_time = time.time() - start_time

    test_acc, test_preds, test_labels = evaluate(model, test_loader, DEVICE)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Training time: {train_time:.1f}s")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "full_finetune_spam.pt")
    save_full_checkpoint(model, checkpoint_path)
    checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "full_finetune_metrics.json"), "w") as f:
        json.dump({
            "strategy": "full_finetune",
            "task": "spam",
            "test_accuracy": test_acc,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100 * trainable / total,
            "train_time_seconds": train_time,
            "num_epochs": NUM_EPOCHS,
            "checkpoint_path": checkpoint_path,
            "checkpoint_size_mb": checkpoint_size_mb,
            "test_predictions": test_preds,
            "test_labels": test_labels,
        }, f, indent=2)

    print(f"\nSaved metrics to {os.path.join(RESULTS_DIR, 'full_finetune_metrics.json')}")


if __name__ == "__main__":
    main()
