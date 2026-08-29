"""
Evaluation, comparison charts, and error analysis across all runs.

Reads the results/*.json files saved by each run_*.py / train_tfidf_*.py script
(both tasks, all four strategies), and produces:

  1. A grouped bar chart comparing test accuracy across strategies, per task.
  2. A grouped bar chart comparing checkpoint size (log scale) across strategies.
  3. A confusion matrix for every (task, strategy) run, saved as one grid image
     per task.
  4. A printed classification report (precision/recall/F1 per class) for every run.
  5. For each run, a handful of actual misclassified examples pulled back out of
     the original test CSVs (using the fact that test_predictions/test_labels
     were saved in the same order as the un-shuffled test_loader).

Run this in Colab AFTER all 8 fine-tuning/TF-IDF runs have completed and saved
their results/*.json files. Needs matplotlib, seaborn, scikit-learn, pandas.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/content/drive/MyDrive/llm_finetune_project"
TASKS = ["spam", "imdb"]
STRATEGIES = ["frozen", "lora", "full_finetune", "tfidf"]
STRATEGY_LABELS = {  # nicer labels for chart legends
    "frozen": "Frozen",
    "lora": "LoRA",
    "full_finetune": "Full fine-tune",
    "tfidf": "TF-IDF + LogReg",
}
CLASS_NAMES = {
    "spam": ["ham", "spam"],
    "imdb": ["negative", "positive"],
}

METRICS_FILENAME = {
    "frozen": "frozen_metrics.json",
    "lora": "lora_metrics.json",
    "full_finetune": "full_finetune_metrics.json",
    "tfidf": "tfidf_metrics.json",
}

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_all_results():
    """Returns nested dict: results[task][strategy] = metrics dict (or None if missing)."""
    results = {task: {} for task in TASKS}
    for task in TASKS:
        for strategy in STRATEGIES:
            path = os.path.join(PROJECT_ROOT, "results", task, METRICS_FILENAME[strategy])
            if os.path.exists(path):
                with open(path) as f:
                    results[task][strategy] = json.load(f)
            else:
                print(f"WARNING: missing {path} - skipping {task}/{strategy}")
                results[task][strategy] = None
    return results


# ---------------------------------------------------------------------------
# Chart 1: accuracy comparison
# ---------------------------------------------------------------------------
def plot_accuracy_comparison(results, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(STRATEGIES))
    width = 0.35

    for i, task in enumerate(TASKS):
        accuracies = [
            results[task][s]["test_accuracy"] * 100 if results[task][s] else 0
            for s in STRATEGIES
        ]
        bars = ax.bar(x + i * width, accuracies, width, label=task.upper())
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{acc:.1f}%", ha="center", fontsize=9)

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy by Strategy and Task")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES])
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved accuracy comparison chart to {save_path}")


# ---------------------------------------------------------------------------
# Chart 2: checkpoint size comparison (efficiency)
# ---------------------------------------------------------------------------
def plot_checkpoint_size_comparison(results, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(STRATEGIES))
    width = 0.35

    for i, task in enumerate(TASKS):
        sizes = [
            results[task][s]["checkpoint_size_mb"] if results[task][s] else 0
            for s in STRATEGIES
        ]
        bars = ax.bar(x + i * width, sizes, width, label=task.upper())
        for bar, size in zip(bars, sizes):
            label = f"{size:.2f}MB" if size >= 1 else f"{size * 1000:.1f}KB"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     label, ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Checkpoint Size (MB, log scale)")
    ax.set_yscale("log")
    ax.set_title("Checkpoint Size by Strategy and Task (log scale)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES])
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved checkpoint size comparison chart to {save_path}")


# ---------------------------------------------------------------------------
# Chart 3: confusion matrices (grid, one per task)
# ---------------------------------------------------------------------------
def plot_confusion_matrix_grid(results, task, save_path):
    class_names = CLASS_NAMES[task]
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(5 * len(STRATEGIES), 4.5))

    for ax, strategy in zip(axes, STRATEGIES):
        metrics = results[task][strategy]
        if metrics is None:
            ax.set_title(f"{STRATEGY_LABELS[strategy]}\n(missing)")
            ax.axis("off")
            continue

        preds = metrics["test_predictions"]
        labels = metrics["test_labels"]
        cm = confusion_matrix(labels, preds)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names, cbar=False)
        ax.set_title(f"{STRATEGY_LABELS[strategy]}\nacc={metrics['test_accuracy']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle(f"Confusion Matrices — {task.upper()}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved {task} confusion matrix grid to {save_path}")


# ---------------------------------------------------------------------------
# Classification reports (printed, not charted)
# ---------------------------------------------------------------------------
def print_classification_reports(results):
    for task in TASKS:
        print(f"\n{'=' * 60}\n{task.upper()}\n{'=' * 60}")
        for strategy in STRATEGIES:
            metrics = results[task][strategy]
            if metrics is None:
                continue
            print(f"\n--- {STRATEGY_LABELS[strategy]} ---")
            print(classification_report(
                metrics["test_labels"], metrics["test_predictions"],
                target_names=CLASS_NAMES[task], digits=3
            ))


# ---------------------------------------------------------------------------
# Misclassified examples — pulls the actual text back from the test CSV,
# using the fact that test_predictions/test_labels are saved in the same
# order as the un-shuffled test_loader (shuffle=False in every run script).
# ---------------------------------------------------------------------------
def show_misclassified_examples(results, task, strategy, n=5):
    metrics = results[task][strategy]
    if metrics is None:
        print(f"No results for {task}/{strategy}")
        return

    test_csv_path = os.path.join(PROJECT_ROOT, "data", task, "test.csv")
    test_df = pd.read_csv(test_csv_path)

    preds = metrics["test_predictions"]
    labels = metrics["test_labels"]

    if len(preds) != len(test_df):
        print(f"WARNING: prediction count ({len(preds)}) doesn't match test.csv "
              f"row count ({len(test_df)}) for {task}/{strategy} - "
              "TF-IDF folds train+val together for fitting but still evaluates "
              "on the same test.csv, so this should normally match. If it "
              "doesn't, don't trust the index alignment below.")

    class_names = CLASS_NAMES[task]
    print(f"\n--- Misclassified examples: {task}/{STRATEGY_LABELS[strategy]} ---")

    misclassified_idxs = [i for i, (p, l) in enumerate(zip(preds, labels)) if p != l]
    print(f"Total misclassified: {len(misclassified_idxs)} / {len(labels)}")

    for idx in misclassified_idxs[:n]:
        text = test_df.iloc[idx]["text"]
        text_preview = text[:200] + ("..." if len(text) > 200 else "")
        print(f"\n  True: {class_names[labels[idx]]} | Predicted: {class_names[preds[idx]]}")
        print(f"  Text: {text_preview}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = load_all_results()

    plot_accuracy_comparison(results, os.path.join(OUTPUT_DIR, "accuracy_comparison.png"))
    plot_checkpoint_size_comparison(results, os.path.join(OUTPUT_DIR, "checkpoint_size_comparison.png"))

    for task in TASKS:
        plot_confusion_matrix_grid(results, task, os.path.join(OUTPUT_DIR, f"confusion_matrices_{task}.png"))

    print_classification_reports(results)


    for task in TASKS:
        valid_strategies = [s for s in STRATEGIES if results[task][s] is not None]
        weakest_strategy = min(valid_strategies, key=lambda s: results[task][s]["test_accuracy"])
        show_misclassified_examples(results, task, weakest_strategy, n=5)

    print(f"\nAll charts saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
