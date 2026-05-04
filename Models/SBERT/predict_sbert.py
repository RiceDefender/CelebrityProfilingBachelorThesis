import argparse
import json
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from _constants import (
    sbert_train_vectors_path,
    sbert_test_vectors_path,
    sbert_checkpoints_dir,
    sbert_predictions_dir,
)

from Models.SBERT.config_sbert_model import (
    TARGET_LABEL,
    BATCH_SIZE,
    RANDOM_SEED,
    VAL_RATIO,
)

from Models.SBERT.train_sbert import (
    aggregate_embeddings_by_celebrity,
    build_model,
    normalize_features,
    set_seed,
    ensure_dirs,
    load_json,
)


def resolve_targets(arg_target: str) -> List[str]:
    if arg_target == "all":
        return ["occupation", "gender", "birthyear"]
    return [arg_target]


def load_checkpoint(target_label: str, device):
    path = os.path.join(sbert_checkpoints_dir, f"{target_label}_best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location=device)


def compute_train_normalization(target_label: str, pooling: str):
    """
    Reconstructs the same train split from train_sbert.py,
    so test embeddings are normalized with the same train mean/std.
    """
    rows = load_json(sbert_train_vectors_path)

    X, y_raw, _ = aggregate_embeddings_by_celebrity(
        rows=rows,
        target_label=target_label,
        pooling=pooling,
    )

    label_to_id = {label: idx for idx, label in enumerate(sorted(set(y_raw)))}
    if target_label == "gender":
        label_to_id = {"male": 0, "female": 1}
    elif target_label == "occupation":
        label_to_id = {
            "sports": 0,
            "performer": 1,
            "creator": 2,
            "politics": 3,
        }
    elif target_label == "birthyear":
        label_to_id = {
            "1994": 0,
            "1985": 1,
            "1975": 2,
            "1963": 3,
            "1947": 4,
        }

    y = np.asarray([label_to_id[str(v)] for v in y_raw], dtype=np.int64)

    from sklearn.model_selection import train_test_split

    indices = np.arange(len(X))
    stratify_y = y if len(np.unique(y)) > 1 else None

    train_idx, _ = train_test_split(
        indices,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=stratify_y,
    )

    X_train = X[train_idx]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    return mean, std


def predict_one_target(target_label: str):
    print(f"\n========== PREDICT TARGET: {target_label} ==========")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(target_label, device)

    label_to_id = checkpoint["label_to_id"]
    id_to_label = {v: k for k, v in label_to_id.items()}

    pooling = checkpoint.get("pooling_strategy", "mean")
    normalize_inputs = checkpoint.get("normalize_inputs", False)

    rows = load_json(sbert_test_vectors_path)

    X_test, y_raw_test, celebrity_ids = aggregate_embeddings_by_celebrity(
        rows=rows,
        target_label=target_label,
        pooling=pooling,
    )

    if normalize_inputs:
        mean, std = compute_train_normalization(target_label, pooling)
        X_test = (X_test - mean) / std

    model = build_model(
        input_dim=checkpoint["input_dim"],
        num_labels=len(label_to_id),
        classifier_type=checkpoint["classifier_type"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    predictions = []

    for cid, pred_id, probs in zip(celebrity_ids, all_preds, all_probs):
        item = {
            "celebrity_id": cid,
            "pred_label": id_to_label[pred_id],
            "probabilities": probs,
        }

        # Falls Testdaten Labels enthalten, speichern wir sie zur späteren Evaluation mit.
        if y_raw_test is not None:
            item["true_label"] = y_raw_test[len(predictions)]

        predictions.append(item)

    out_path = os.path.join(
        sbert_predictions_dir,
        f"{target_label}_test_predictions.json",
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"[{target_label}] Saved test predictions to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict test data with saved SBERT checkpoints")
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear", "all"],
        default=TARGET_LABEL,
    )

    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    ensure_dirs()

    for target in resolve_targets(args.target):
        predict_one_target(target)


if __name__ == "__main__":
    main()