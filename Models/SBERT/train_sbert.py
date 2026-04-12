import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from _constants import (
    sbert_train_vectors_path,
    sbert_output_dir,
    sbert_checkpoints_dir,
    sbert_predictions_dir,
    sbert_metrics_dir,
)
from Models.SBERT.config_sbert_model import (
    TARGET_LABEL,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    RANDOM_SEED,
    VAL_RATIO,
    CLASSIFIER_TYPE,
    HIDDEN_DIM,
    DROPOUT,
    POOLING_STRATEGY,
    NORMALIZE_INPUTS,
    SAVE_CHECKPOINTS,
    SAVE_PREDICTIONS,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
)


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    os.makedirs(sbert_output_dir, exist_ok=True)
    os.makedirs(sbert_checkpoints_dir, exist_ok=True)
    os.makedirs(sbert_predictions_dir, exist_ok=True)
    os.makedirs(sbert_metrics_dir, exist_ok=True)


def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------------
# Aggregation
# -------------------------------------------------------------------
def aggregate_embeddings_by_celebrity(
    rows: List[dict],
    target_label: str,
    pooling: str = "mean",
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Converts chunk-level SBERT rows into celebrity-level embeddings.

    Returns:
        X: np.ndarray of shape [num_celebrities, embedding_dim]
        y_raw: raw labels
        celebrity_ids: ids aligned with X/y
    """
    grouped_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    grouped_labels: Dict[str, str] = {}

    for row in rows:
        celebrity_id = str(row["celebrity_id"])
        emb = np.asarray(row["embedding"], dtype=np.float32)

        grouped_embeddings[celebrity_id].append(emb)
        grouped_labels[celebrity_id] = row[target_label]

    celebrity_ids = sorted(grouped_embeddings.keys())

    features = []
    labels = []

    for cid in celebrity_ids:
        embs = np.stack(grouped_embeddings[cid], axis=0)

        if pooling == "mean":
            pooled = embs.mean(axis=0)
        elif pooling == "max":
            pooled = embs.max(axis=0)
        else:
            raise ValueError(f"Unsupported pooling strategy: {pooling}")

        features.append(pooled)
        labels.append(grouped_labels[cid])

    X = np.stack(features, axis=0).astype(np.float32)
    return X, labels, celebrity_ids


# -------------------------------------------------------------------
# Label encoding
# -------------------------------------------------------------------
def build_label_mapping(target_label: str, y_raw: List[str]) -> Dict[str, int]:
    if target_label == "gender":
        # Keep explicit ordering for stability
        ordered = ["male", "female"]
        present = [label for label in ordered if label in set(y_raw)]
        return {label: idx for idx, label in enumerate(present)}

    if target_label == "occupation":
        ordered = ["sports", "performer", "creator", "politics"]
        present = [label for label in ordered if label in set(y_raw)]
        return {label: idx for idx, label in enumerate(present)}

    if target_label == "birthyear":
        years = sorted({int(y) for y in y_raw})
        return {str(year): idx for idx, year in enumerate(years)}

    raise ValueError(f"Unsupported target label: {target_label}")


def encode_labels(y_raw: List[str], label_to_id: Dict[str, int]) -> np.ndarray:
    return np.asarray([label_to_id[str(y)] for y in y_raw], dtype=np.int64)


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------
class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    input_dim: int,
    num_labels: int,
    classifier_type: str,
    hidden_dim: int,
    dropout: float,
) -> nn.Module:
    if classifier_type == "linear":
        return LinearClassifier(input_dim=input_dim, num_labels=num_labels)
    if classifier_type == "mlp":
        return MLPClassifier(
            input_dim=input_dim,
            num_labels=num_labels,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported CLASSIFIER_TYPE: {classifier_type}")


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
@dataclass
class RunArtifacts:
    metrics: dict
    predictions: List[dict]


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    return X_train_norm, X_val_norm


def train_one_target(target_label: str) -> RunArtifacts:
    print(f"\n========== TARGET: {target_label} ==========")

    rows = load_json(sbert_train_vectors_path)
    X, y_raw, celebrity_ids = aggregate_embeddings_by_celebrity(
        rows=rows,
        target_label=target_label,
        pooling=POOLING_STRATEGY,
    )

    label_to_id = build_label_mapping(target_label, y_raw)
    id_to_label = {v: k for k, v in label_to_id.items()}
    y = encode_labels(y_raw, label_to_id)

    print(f"[{target_label}] Num samples: {len(X)}")
    print(f"[{target_label}] Input dim: {X.shape[1]}")
    print(f"[{target_label}] Num labels: {len(label_to_id)}")
    print(f"[{target_label}] Labels: {label_to_id}")

    indices = np.arange(len(X))
    stratify_y = y if len(np.unique(y)) > 1 else None

    train_idx, val_idx = train_test_split(
        indices,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=stratify_y,
    )

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    ids_val = [celebrity_ids[i] for i in val_idx]

    if NORMALIZE_INPUTS:
        X_train, X_val = normalize_features(X_train, X_val)

    train_ds = EmbeddingDataset(X_train, y_train)
    val_ds = EmbeddingDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        input_dim=X.shape[1],
        num_labels=len(label_to_id),
        classifier_type=CLASSIFIER_TYPE,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_X)
                loss = criterion(logits, batch_y)

                preds = torch.argmax(logits, dim=1)

                val_losses.append(loss.item())
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(batch_y.cpu().numpy().tolist())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        print(
            f"[{target_label}] "
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss={np.mean(train_losses):.4f} | "
            f"val_loss={np.mean(val_losses):.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1 + EARLY_STOPPING_MIN_DELTA:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"[{target_label}] Early stopping triggered at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best model state captured.")

    model.load_state_dict(best_state)
    model.eval()

    # Final validation predictions with best model
    val_probs = []
    val_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            val_probs.extend(probs.cpu().numpy().tolist())
            val_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch_y.numpy().tolist())

    report = classification_report(
        all_labels,
        val_preds,
        labels=list(range(len(id_to_label))),
        target_names=[id_to_label[i] for i in range(len(id_to_label))],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "target_label": target_label,
        "num_samples": int(len(X)),
        "num_train": int(len(X_train)),
        "num_val": int(len(X_val)),
        "input_dim": int(X.shape[1]),
        "num_labels": int(len(label_to_id)),
        "label_to_id": label_to_id,
        "best_val_macro_f1": float(best_val_f1),
        "best_val_accuracy": float(accuracy_score(all_labels, val_preds)),
        "classification_report": report,
        "pooling_strategy": POOLING_STRATEGY,
        "classifier_type": CLASSIFIER_TYPE,
        "normalize_inputs": NORMALIZE_INPUTS,
    }

    predictions = []
    for cid, gold_id, pred_id, probs in zip(ids_val, all_labels, val_preds, val_probs):
        predictions.append(
            {
                "celebrity_id": cid,
                "gold_label": id_to_label[gold_id],
                "pred_label": id_to_label[pred_id],
                "probabilities": probs,
            }
        )

    if SAVE_CHECKPOINTS:
        checkpoint_path = os.path.join(sbert_checkpoints_dir, f"{target_label}_best.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_to_id": label_to_id,
                "input_dim": X.shape[1],
                "classifier_type": CLASSIFIER_TYPE,
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT,
                "pooling_strategy": POOLING_STRATEGY,
                "normalize_inputs": NORMALIZE_INPUTS,
            },
            checkpoint_path,
        )
        print(f"[{target_label}] Saved checkpoint to: {checkpoint_path}")

    metrics_path = os.path.join(sbert_metrics_dir, f"{target_label}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[{target_label}] Saved metrics to: {metrics_path}")

    if SAVE_PREDICTIONS:
        predictions_path = os.path.join(sbert_predictions_dir, f"{target_label}_val_predictions.json")
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"[{target_label}] Saved predictions to: {predictions_path}")

    return RunArtifacts(metrics=metrics, predictions=predictions)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def resolve_targets(arg_target: str) -> List[str]:
    if arg_target == "all":
        return ["occupation", "gender", "birthyear"]
    return [arg_target]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SBERT MVP classifier")
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear", "all"],
        default=TARGET_LABEL,
        help="Target label to train",
    )
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    ensure_dirs()

    targets = resolve_targets(args.target)
    for target in targets:
        train_one_target(target)


if __name__ == "__main__":
    main()