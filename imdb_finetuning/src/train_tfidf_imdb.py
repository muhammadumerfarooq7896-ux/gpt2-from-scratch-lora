"""
TF-IDF + Logistic Regression baseline for IMDB sentiment classification.

Identical strategy to train_tfidf_spam.py: no GPT-2 involved, a classical
lexical baseline for comparison against the three transformer strategies.

Run this anywhere - it's fast and doesn't need a GPU. In Colab, just make
sure DATA_DIR points at your Drive dataset folder.
"""

import os
import json
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/content/drive/MyDrive/llm_finetune_project/data/imdb"
RESULTS_DIR = "/content/drive/MyDrive/llm_finetune_project/results/imdb"
CHECKPOINT_DIR = "/content/drive/MyDrive/llm_finetune_project/checkpoints/imdb"

MAX_FEATURES = 5000  # vocabulary size cap for the TF-IDF vectorizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    fit_texts = pd.concat([train_df["text"], val_df["text"]]).astype(str).tolist()
    fit_labels = pd.concat([train_df["label"], val_df["label"]]).tolist()

    test_texts = test_df["text"].astype(str).tolist()
    test_labels = test_df["label"].tolist()

    start_time = time.time()

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
    X_train = vectorizer.fit_transform(fit_texts)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, fit_labels)

    train_time = time.time() - start_time

    X_test = vectorizer.transform(test_texts)
    test_preds = clf.predict(X_test).tolist()
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"Vocabulary size (TF-IDF features): {len(vectorizer.vocabulary_):,}")
    print(f"Logistic regression coefficients: {clf.coef_.size:,}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Training time: {train_time:.2f}s")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "tfidf_imdb.joblib")
    joblib.dump({"vectorizer": vectorizer, "classifier": clf}, checkpoint_path)
    checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)
    print(f"Saved TF-IDF + LogisticRegression checkpoint to {checkpoint_path} "
          f"({checkpoint_size_mb:.4f} MB)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "tfidf_metrics.json"), "w") as f:
        json.dump({
            "strategy": "tfidf",
            "task": "imdb",
            "test_accuracy": test_acc,
            "trainable_params": int(clf.coef_.size + clf.intercept_.size),
            "vocab_size": len(vectorizer.vocabulary_),
            "train_time_seconds": train_time,
            "checkpoint_path": checkpoint_path,
            "checkpoint_size_mb": checkpoint_size_mb,
            "test_predictions": test_preds,
            "test_labels": test_labels,
        }, f, indent=2)

    print(f"\nSaved metrics to {os.path.join(RESULTS_DIR, 'tfidf_metrics.json')}")


if __name__ == "__main__":
    main()
