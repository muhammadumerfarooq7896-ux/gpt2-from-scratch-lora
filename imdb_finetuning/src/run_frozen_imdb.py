

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
DATA_DIR = "/content/drive/MyDrive/llm_finetune_project/data/imdb"
RESULTS_DIR = "/content/drive/MyDrive/llm_finetune_project/results/imdb"
CHECKPOINT_DIR = "/content/drive/MyDrive/llm_finetune_project/checkpoints/imdb"

MAX_LENGTH = 128  
BATCH_SIZE = 32
NUM_EPOCHS = 4
LEARNING_RATE = 1e-3
NUM_CLASSES = 2  # positive / negative
PAD_TOKEN_ID = 50256  # GPT-2's <|endoftext|> token, used for right-padding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class IMDBDataset(Dataset):
    """
    Tokenizes each review, truncates to MAX_LENGTH (reviews are often much
    longer than that) and right-pads shorter ones with GPT-2's <|endoftext|>
    token. Since GPT-2 is causal, the representation at the LAST position of
    a right-padded sequence still correctly reflects everything before it,
    so logits[:, -1, :] is safe to use as the classification logits.
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
def build_frozen_model():
    model, cfg = instantiate_model("gpt2-small (124M)", load_weights=True)

    for param in model.parameters():
        param.requires_grad = False

    model.out_head = nn.Linear(cfg["emb_dim"], NUM_CLASSES)

    return model, cfg


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_head_checkpoint(model, path):
    trainable_state_dict = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(trainable_state_dict, path)

    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"Saved head-only checkpoint to {path} ({size_mb:.4f} MB, "
          f"{len(trainable_state_dict)} tensors)")


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

    train_ds = IMDBDataset(os.path.join(DATA_DIR, "train.csv"), tokenizer)
    val_ds = IMDBDataset(os.path.join(DATA_DIR, "validation.csv"), tokenizer)
    test_ds = IMDBDataset(os.path.join(DATA_DIR, "test.csv"), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model, cfg = build_frozen_model()
    model.to(DEVICE)

    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

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
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "frozen_imdb.pt")
    save_head_checkpoint(model, checkpoint_path)
    checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "frozen_metrics.json"), "w") as f:
        json.dump({
            "strategy": "frozen",
            "task": "imdb",
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

    print(f"\nSaved metrics to {os.path.join(RESULTS_DIR, 'frozen_metrics.json')}")


if __name__ == "__main__":
    main()
