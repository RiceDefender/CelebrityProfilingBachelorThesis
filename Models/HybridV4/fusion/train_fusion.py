import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from _constants import (
    hybrid_v4_bertweet_probs_dir,
    hybrid_v4_feature_predictions_dir,
    hybrid_v4_fusion_models_dir,
    hybrid_v4_fusion_predictions_dir,
    hybrid_v4_fusion_metrics_dir,
)

from Models.HybridV4.feature.config_features import LABEL_ORDERS, RANDOM_SEED


TARGETS = ["occupation"]


def ensure_dirs():
    os.makedirs(hybrid_v4_fusion_models_dir, exist_ok=True)
    os.makedirs(hybrid_v4_fusion_predictions_dir, exist_ok=True)
    os.makedirs(hybrid_v4_fusion_metrics_dir, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def rows_by_id(rows: List[dict]) -> Dict[str, dict]:
    return {str(row["celebrity_id"]): row for row in rows}


def load_bertweet_rows(target: str, split: str) -> Dict[str, dict]:
    path = os.path.join(
        hybrid_v4_bertweet_probs_dir,
        f"{target}_{split}_bertweet_probs.json",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing BERTweet probs: {path}")

    return rows_by_id(load_json(path))


def load_sparse_feature_rows(target: str, split: str) -> Dict[str, dict]:
    path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{target}_{split}_feature_probs.json",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing sparse feature probs: {path}")

    return rows_by_id(load_json(path))


def load_svd_feature_rows(target: str, split: str) -> Dict[str, dict]:
    path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{target}_{split}_svd_feature_probs.json",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing SVD feature probs: {path}")

    return rows_by_id(load_json(path))


def load_gate_rows(gate: str, split: str) -> Dict[str, dict]:
    path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{gate}_{split}_feature_probs.json",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing gate probs: {path}")

    return rows_by_id(load_json(path))


def entropy(probs: List[float]) -> float:
    arr = np.asarray(probs, dtype=np.float64)
    arr = np.clip(arr, 1e-12, 1.0)
    return float(-np.sum(arr * np.log(arr)))


def margin(probs: List[float]) -> float:
    arr = np.sort(np.asarray(probs, dtype=np.float64))[::-1]
    if len(arr) < 2:
        return 0.0
    return float(arr[0] - arr[1])


def max_prob(probs: List[float]) -> float:
    return float(np.max(np.asarray(probs, dtype=np.float64)))


def get_binary_prob(row: dict, positive_label: str) -> float:
    labels = row["labels"]
    probs = row["feature_probabilities"]

    if positive_label not in labels:
        raise ValueError(
            f"Positive label {positive_label} not found in labels={labels}"
        )

    idx = labels.index(positive_label)
    return float(probs[idx])


def build_dataset(target: str, split: str) -> Tuple[np.ndarray, np.ndarray, List[dict], List[str]]:
    labels = LABEL_ORDERS[target]
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    bertweet = load_bertweet_rows(target, split)
    sparse = load_sparse_feature_rows(target, split)
    svd = load_svd_feature_rows(target, split)

    creator_binary = load_gate_rows("creator_binary", split)
    creator_vs_performer = load_gate_rows("creator_vs_performer", split)

    common_ids = set(bertweet.keys()) & set(sparse.keys()) & set(svd.keys())

    # creator_vs_performer exists only for true creator/performer rows.
    # For other occupation classes we fill neutral values.
    # creator_binary exists for all rows.
    common_ids = common_ids & set(creator_binary.keys())

    x_rows = []
    y_rows = []
    meta_rows = []

    for cid in sorted(common_ids, key=lambda x: int(x) if x.isdigit() else x):
        b = bertweet[cid]
        sp = sparse[cid]
        sv = svd[cid]
        cb = creator_binary[cid]
        cvp = creator_vs_performer.get(cid)

        true_label = b["true_label"]

        if true_label != sp["true_label"] or true_label != sv["true_label"]:
            raise ValueError(
                f"True-label mismatch for celebrity_id={cid}: "
                f"bertweet={true_label}, sparse={sp['true_label']}, svd={sv['true_label']}"
            )

        b_v3 = b["bertweet_v3_probabilities"]
        b_v34 = b["bertweet_v34_probabilities"]
        sp_probs = sp["feature_probabilities"]
        sv_probs = sv["svd_feature_probabilities"]

        features = []

        # Main model probabilities
        features.extend(b_v3)
        features.extend(b_v34)
        features.extend(sp_probs)
        features.extend(sv_probs)

        # Confidence features
        for probs in [b_v3, b_v34, sp_probs, sv_probs]:
            features.append(max_prob(probs))
            features.append(margin(probs))
            features.append(entropy(probs))

        # Creator gates
        cb_creator_prob = get_binary_prob(cb, "creator")
        features.append(cb_creator_prob)

        if cvp is not None:
            cvp_creator_prob = get_binary_prob(cvp, "creator")
            cvp_available = 1.0
        else:
            cvp_creator_prob = 0.5
            cvp_available = 0.0

        features.append(cvp_creator_prob)
        features.append(cvp_available)

        x_rows.append(features)
        y_rows.append(label_to_id[true_label])

        meta_rows.append({
            "celebrity_id": cid,
            "target": target,
            "split": split,
            "true_label": true_label,
            "labels": labels,
            "bertweet_v3_pred_label": b["bertweet_v3_pred_label"],
            "bertweet_v34_pred_label": b["bertweet_v34_pred_label"],
            "sparse_feature_pred_label": sp["feature_pred_label"],
            "svd_feature_pred_label": sv["svd_feature_pred_label"],
            "creator_binary_creator_prob": cb_creator_prob,
            "creator_vs_performer_creator_prob": cvp_creator_prob,
            "creator_vs_performer_available": cvp_available,
        })

    if not x_rows:
        raise ValueError(f"No common rows found for target={target}, split={split}")

    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int64)

    return x, y, meta_rows, labels


def evaluate(y_true, y_pred, labels: List[str]):
    id_to_label = {idx: label for idx, label in enumerate(labels)}

    y_true_labels = [id_to_label[int(i)] for i in y_true]
    y_pred_labels = [id_to_label[int(i)] for i in y_pred]

    acc = accuracy_score(y_true_labels, y_pred_labels)
    macro_f1 = f1_score(
        y_true_labels,
        y_pred_labels,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true_labels,
        y_pred_labels,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "classification_report": report,
    }


def train_fusion_for_target(target: str):
    print(f"\n========== HybridV4 Fusion: target={target} ==========")

    labels = LABEL_ORDERS[target]

    x_val, y_val, val_meta, _ = build_dataset(target, "val")
    x_test, y_test, test_meta, _ = build_dataset(target, "test")

    print(f"[INFO] Val shape:  {x_val.shape}")
    print(f"[INFO] Test shape: {x_test.shape}")
    print(f"[INFO] Labels:     {labels}")

    scaler = StandardScaler()
    x_val_scaled = scaler.fit_transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    clf = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        solver="liblinear",
        verbose=0,
    )

    clf.fit(x_val_scaled, y_val)

    test_probs = clf.predict_proba(x_test_scaled)
    test_pred_ids = np.argmax(test_probs, axis=1)

    metrics = evaluate(y_test, test_pred_ids, labels)
    metrics.update({
        "target": target,
        "split": "test",
        "model": "hybrid_v4_late_fusion_logreg",
        "num_val_rows": int(x_val.shape[0]),
        "num_test_rows": int(x_test.shape[0]),
        "num_features": int(x_val.shape[1]),
        "feature_sources": [
            "bertweet_v3_probabilities",
            "bertweet_v34_probabilities",
            "sparse_feature_probabilities",
            "svd_feature_probabilities",
            "confidence_features",
            "creator_binary_probability",
            "creator_vs_performer_probability",
        ],
    })

    id_to_label = {idx: label for idx, label in enumerate(labels)}

    predictions = []
    for meta, probs, pred_id in zip(test_meta, test_probs, test_pred_ids):
        row = dict(meta)
        row.update({
            "fusion_probabilities": probs.tolist(),
            "fusion_pred_label": id_to_label[int(pred_id)],
            "fusion_model": "hybrid_v4_late_fusion_logreg",
        })
        predictions.append(row)

    model_payload = {
        "target": target,
        "labels": labels,
        "scaler": scaler,
        "classifier": clf,
        "feature_sources": metrics["feature_sources"],
    }

    model_path = os.path.join(
        hybrid_v4_fusion_models_dir,
        f"{target}_fusion_model.pkl",
    )
    pred_path = os.path.join(
        hybrid_v4_fusion_predictions_dir,
        f"{target}_test_fusion_predictions.json",
    )
    metrics_path = os.path.join(
        hybrid_v4_fusion_metrics_dir,
        f"{target}_test_fusion_metrics.json",
    )

    save_pickle(model_payload, model_path)
    save_json(predictions, pred_path)
    save_json(metrics, metrics_path)

    print(
        f"[RESULT] {target} fusion "
        f"test_acc={metrics['accuracy']:.4f} "
        f"test_macro_f1={metrics['macro_f1']:.4f}"
    )
    print(f"[OK] Saved model:      {model_path}")
    print(f"[OK] Saved predictions:{pred_path}")
    print(f"[OK] Saved metrics:    {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train HybridV4 late-fusion model."
    )
    parser.add_argument(
        "--target",
        choices=["occupation"],
        default="occupation",
    )

    args = parser.parse_args()

    ensure_dirs()
    train_fusion_for_target(args.target)


if __name__ == "__main__":
    main()