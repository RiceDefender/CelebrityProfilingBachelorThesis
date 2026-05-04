import argparse
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from _constants import (
    sbert_train_vectors_path,
    sbert_v2_output_dir,
    sbert_v2_checkpoints_dir,
    sbert_v2_predictions_dir,
    sbert_v2_metrics_dir,
)

from Models.SBERT.config_sbert_model import (
    TARGET_LABEL,
    RANDOM_SEED,
    VAL_RATIO,
    V2_MAX_ITER,
    V2_C,
    V2_SOLVER,
    V2_VOTING_STRATEGY,
    V2_NORMALIZE_INPUTS,
    V2_CLASS_WEIGHT_BY_TARGET,
    V2_USE_SAMPLE_WEIGHTS
)

BIRTHYEAR_BUCKETS = ["1994", "1985", "1975", "1963", "1947"]

LABEL_ORDERS = {
    "occupation": ["sports", "performer", "creator", "politics"],
    "gender": ["male", "female"],
    "birthyear": BIRTHYEAR_BUCKETS,
}

def build_sample_weights(y_train, id_to_label, target_label):
    config = V2_CLASS_WEIGHT_BY_TARGET.get(target_label, None)

    if config is None:
        return None

    if config == "balanced":
        from sklearn.utils.class_weight import compute_sample_weight
        return compute_sample_weight(class_weight="balanced", y=y_train)

    weights = []
    for label_id in y_train:
        label = id_to_label[int(label_id)]
        weights.append(float(config.get(label, 1.0)))

    return np.asarray(weights, dtype=np.float32)

def ensure_dirs():
    os.makedirs(sbert_v2_output_dir, exist_ok=True)
    os.makedirs(sbert_v2_checkpoints_dir, exist_ok=True)
    os.makedirs(sbert_v2_predictions_dir, exist_ok=True)
    os.makedirs(sbert_v2_metrics_dir, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_birthyear_to_bucket(year) -> str:
    year = int(year)
    bucket_years = [int(y) for y in BIRTHYEAR_BUCKETS]
    nearest = min(bucket_years, key=lambda b: abs(year - b))
    return str(nearest)


def build_label_mapping(target_label: str, labels: List[str]) -> Dict[str, int]:
    ordered = LABEL_ORDERS[target_label]
    present = [label for label in ordered if label in set(labels)]
    return {label: idx for idx, label in enumerate(present)}


def load_chunk_level_data(target_label: str):
    rows = load_json(sbert_train_vectors_path)

    X = []
    y_raw = []
    celebrity_ids = []

    for row in rows:
        label = row[target_label]
        if target_label == "birthyear":
            label = map_birthyear_to_bucket(label)

        X.append(np.asarray(row["embedding"], dtype=np.float32))
        y_raw.append(str(label))
        celebrity_ids.append(str(row["celebrity_id"]))

    X = np.stack(X, axis=0).astype(np.float32)
    return X, y_raw, celebrity_ids


def aggregate_soft_vote(
    celebrity_ids: List[str],
    true_labels: List[str],
    probs: np.ndarray,
    id_to_label: Dict[int, str],
):
    grouped_probs = defaultdict(list)
    grouped_true = {}

    for cid, gold, prob in zip(celebrity_ids, true_labels, probs):
        grouped_probs[cid].append(prob)
        grouped_true[cid] = gold

    predictions = []
    y_true = []
    y_pred = []

    for cid in sorted(grouped_probs.keys()):
        avg_prob = np.mean(np.stack(grouped_probs[cid], axis=0), axis=0)
        pred_id = int(np.argmax(avg_prob))
        pred_label = id_to_label[pred_id]
        true_label = grouped_true[cid]

        predictions.append({
            "celebrity_id": cid,
            "true_label": true_label,
            "pred_label": pred_label,
            "probabilities": avg_prob.tolist(),
            "num_chunks": len(grouped_probs[cid]),
        })

        y_true.append(true_label)
        y_pred.append(pred_label)

    return predictions, y_true, y_pred


def aggregate_hard_vote(
    celebrity_ids: List[str],
    true_labels: List[str],
    pred_ids: np.ndarray,
    id_to_label: Dict[int, str],
    num_labels: int,
):
    grouped_preds = defaultdict(list)
    grouped_true = {}

    for cid, gold, pred_id in zip(celebrity_ids, true_labels, pred_ids):
        grouped_preds[cid].append(int(pred_id))
        grouped_true[cid] = gold

    predictions = []
    y_true = []
    y_pred = []

    for cid in sorted(grouped_preds.keys()):
        counts = np.bincount(grouped_preds[cid], minlength=num_labels)
        pred_id = int(np.argmax(counts))
        pred_label = id_to_label[pred_id]
        true_label = grouped_true[cid]

        vote_distribution = counts / counts.sum()

        predictions.append({
            "celebrity_id": cid,
            "true_label": true_label,
            "pred_label": pred_label,
            "probabilities": vote_distribution.tolist(),
            "num_chunks": len(grouped_preds[cid]),
        })

        y_true.append(true_label)
        y_pred.append(pred_label)

    return predictions, y_true, y_pred


def train_one_target(target_label: str):
    print(f"\n========== SBERT V2 TARGET: {target_label} ==========")

    X, y_raw, celebrity_ids = load_chunk_level_data(target_label)

    label_to_id = build_label_mapping(target_label, y_raw)
    id_to_label = {v: k for k, v in label_to_id.items()}
    y = np.asarray([label_to_id[label] for label in y_raw], dtype=np.int64)

    unique_celebrities = sorted(set(celebrity_ids))
    celebrity_to_label = {}

    for cid, label in zip(celebrity_ids, y_raw):
        celebrity_to_label[cid] = label

    celeb_labels = [celebrity_to_label[cid] for cid in unique_celebrities]
    celeb_y = np.asarray([label_to_id[label] for label in celeb_labels], dtype=np.int64)

    train_celeb_ids, val_celeb_ids = train_test_split(
        unique_celebrities,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=celeb_y,
    )

    train_celeb_ids = set(train_celeb_ids)
    val_celeb_ids = set(val_celeb_ids)

    train_mask = np.asarray([cid in train_celeb_ids for cid in celebrity_ids])
    val_mask = np.asarray([cid in val_celeb_ids for cid in celebrity_ids])

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    val_ids = [cid for cid, keep in zip(celebrity_ids, val_mask) if keep]
    val_true_labels = [label for label, keep in zip(y_raw, val_mask) if keep]

    scaler = None
    if V2_NORMALIZE_INPUTS:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    class_weight_config = V2_CLASS_WEIGHT_BY_TARGET.get(target_label, None)

    clf = LogisticRegression(
        C=V2_C,
        max_iter=V2_MAX_ITER,
        solver=V2_SOLVER,
        class_weight=class_weight_config if class_weight_config == "balanced" else None,
        n_jobs=None,
        random_state=RANDOM_SEED,
    )

    sample_weights = None

    if V2_USE_SAMPLE_WEIGHTS:
        sample_weights = build_sample_weights(
            y_train=y_train,
            id_to_label=id_to_label,
            target_label=target_label,
        )

    clf.fit(X_train, y_train, sample_weight=sample_weights)

    chunk_preds = clf.predict(X_val)
    chunk_probs = clf.predict_proba(X_val)

    if V2_VOTING_STRATEGY == "soft":
        predictions, celeb_y_true, celeb_y_pred = aggregate_soft_vote(
            celebrity_ids=val_ids,
            true_labels=val_true_labels,
            probs=chunk_probs,
            id_to_label=id_to_label,
        )
    elif V2_VOTING_STRATEGY == "hard":
        predictions, celeb_y_true, celeb_y_pred = aggregate_hard_vote(
            celebrity_ids=val_ids,
            true_labels=val_true_labels,
            pred_ids=chunk_preds,
            id_to_label=id_to_label,
            num_labels=len(label_to_id),
        )
    else:
        raise ValueError(f"Unsupported V2_VOTING_STRATEGY: {V2_VOTING_STRATEGY}")

    labels = LABEL_ORDERS[target_label]

    report = classification_report(
        celeb_y_true,
        celeb_y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "target_label": target_label,
        "version": "sbert_v2",
        "classifier_type": "logistic_regression",
        "voting_strategy": V2_VOTING_STRATEGY,
        "num_chunk_samples": int(len(X)),
        "num_train_chunks": int(len(X_train)),
        "num_val_chunks": int(len(X_val)),
        "num_train_celebrities": int(len(train_celeb_ids)),
        "num_val_celebrities": int(len(val_celeb_ids)),
        "input_dim": int(X.shape[1]),
        "num_labels": int(len(label_to_id)),
        "label_to_id": label_to_id,
        "normalize_inputs": V2_NORMALIZE_INPUTS,
        "val_accuracy": float(accuracy_score(celeb_y_true, celeb_y_pred)),
        "val_macro_f1": float(
            f1_score(celeb_y_true, celeb_y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "classification_report": report,
    }

    checkpoint = {
        "classifier": clf,
        "scaler": scaler,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "target_label": target_label,
        "voting_strategy": V2_VOTING_STRATEGY,
        "normalize_inputs": V2_NORMALIZE_INPUTS,
    }

    checkpoint_path = os.path.join(
        sbert_v2_checkpoints_dir,
        f"{target_label}_logreg_voting.pkl",
    )

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    metrics_path = os.path.join(
        sbert_v2_metrics_dir,
        f"{target_label}_metrics.json",
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    predictions_path = os.path.join(
        sbert_v2_predictions_dir,
        f"{target_label}_val_predictions.json",
    )

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved checkpoint: {checkpoint_path}")
    print(f"[OK] Saved metrics: {metrics_path}")
    print(f"[OK] Saved predictions: {predictions_path}")
    print(f"[RESULT] {target_label} val_acc={metrics['val_accuracy']:.4f} val_macro_f1={metrics['val_macro_f1']:.4f}")


def resolve_targets(target: str):
    if target == "all":
        return ["occupation", "gender", "birthyear"]
    return [target]


def main():
    parser = argparse.ArgumentParser(description="Train SBERT V2 Logistic Regression with chunk voting")
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear", "all"],
        default=TARGET_LABEL,
    )
    args = parser.parse_args()

    ensure_dirs()

    for target in resolve_targets(args.target):
        train_one_target(target)


if __name__ == "__main__":
    main()