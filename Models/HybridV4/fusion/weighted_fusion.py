import argparse
import json
import os
import sys
from itertools import product
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

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
    hybrid_v4_fusion_predictions_dir,
    hybrid_v4_fusion_metrics_dir,
)

from Models.HybridV4.feature.config_features import LABEL_ORDERS


def ensure_dirs():
    os.makedirs(hybrid_v4_fusion_predictions_dir, exist_ok=True)
    os.makedirs(hybrid_v4_fusion_metrics_dir, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def rows_by_id(rows: List[dict]) -> Dict[str, dict]:
    return {str(row["celebrity_id"]): row for row in rows}


def load_rows(target: str, split: str):
    bertweet_path = os.path.join(
        hybrid_v4_bertweet_probs_dir,
        f"{target}_{split}_bertweet_probs.json",
    )
    sparse_path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{target}_{split}_feature_probs.json",
    )
    svd_path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{target}_{split}_svd_feature_probs.json",
    )

    bertweet = rows_by_id(load_json(bertweet_path))
    sparse = rows_by_id(load_json(sparse_path))
    svd = rows_by_id(load_json(svd_path))

    common_ids = sorted(
        set(bertweet.keys()) & set(sparse.keys()) & set(svd.keys()),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    if not common_ids:
        raise ValueError("No common IDs found.")

    return bertweet, sparse, svd, common_ids


def load_gate_rows(gate: str, split: str) -> Dict[str, dict]:
    path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{gate}_{split}_feature_probs.json",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing gate probs: {path}")

    return rows_by_id(load_json(path))


def get_binary_prob(row: dict, positive_label: str) -> float:
    labels = row["labels"]
    probs = row["feature_probabilities"]

    if positive_label not in labels:
        raise ValueError(
            f"Positive label {positive_label} not found in labels={labels}"
        )

    idx = labels.index(positive_label)
    return float(probs[idx])


def evaluate_predictions(y_true_labels, y_pred_labels, labels):
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


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    s = float(np.sum(weights))
    if s <= 0:
        raise ValueError("Weight sum must be > 0.")
    return weights / s


def run_weighted_fusion(
        target: str,
        split: str,
        weights,
        suffix: str,
        creator_boost_alpha: float = 0.0,
):
    labels = LABEL_ORDERS[target]

    bertweet, sparse, svd, common_ids = load_rows(target, split)
    weights = normalize_weights(weights)
    creator_binary = None
    creator_vs_performer = None

    if target == "occupation" and creator_boost_alpha > 0:
        creator_binary = load_gate_rows("creator_binary", split)
        creator_vs_performer = load_gate_rows("creator_vs_performer", split)

    predictions = []
    y_true = []
    y_pred = []

    for cid in common_ids:
        b = bertweet[cid]
        sp = sparse[cid]
        sv = svd[cid]

        true_label = b["true_label"]

        if true_label != sp["true_label"] or true_label != sv["true_label"]:
            raise ValueError(
                f"True-label mismatch for cid={cid}: "
                f"bertweet={true_label}, sparse={sp['true_label']}, svd={sv['true_label']}"
            )

        p_v3 = np.asarray(b["bertweet_v3_probabilities"], dtype=np.float64)
        p_v34 = np.asarray(b["bertweet_v34_probabilities"], dtype=np.float64)
        p_sparse = np.asarray(sp["feature_probabilities"], dtype=np.float64)
        p_svd = np.asarray(sv["svd_feature_probabilities"], dtype=np.float64)

        final_probs = (
                weights[0] * p_v3
                + weights[1] * p_v34
                + weights[2] * p_sparse
                + weights[3] * p_svd
        )

        creator_binary_prob = None
        creator_vs_performer_prob = None
        creator_signal = None

        if target == "occupation" and creator_boost_alpha > 0:
            creator_idx = labels.index("creator")

            cb_row = creator_binary.get(cid) if creator_binary is not None else None
            cvp_row = creator_vs_performer.get(cid) if creator_vs_performer is not None else None

            if cb_row is not None:
                creator_binary_prob = get_binary_prob(cb_row, "creator")
            else:
                creator_binary_prob = 0.0

            if cvp_row is not None:
                creator_vs_performer_prob = get_binary_prob(cvp_row, "creator")
                creator_signal = 0.5 * creator_binary_prob + 0.5 * creator_vs_performer_prob
            else:
                # If the creator_vs_performer model has no row, the true occupation was
                # neither creator nor performer in its own dataset. At inference time we
                # do not know this, so we use only the general creator_binary signal.
                creator_vs_performer_prob = 0.5
                creator_signal = creator_binary_prob

            final_probs[creator_idx] += creator_boost_alpha * creator_signal

        final_probs = final_probs / np.sum(final_probs)

        pred_idx = int(np.argmax(final_probs))
        pred_label = labels[pred_idx]

        y_true.append(true_label)
        y_pred.append(pred_label)

        predictions.append({
            "celebrity_id": cid,
            "target": target,
            "split": split,
            "true_label": true_label,
            "labels": labels,
            "weights": {
                "bertweet_v3": float(weights[0]),
                "bertweet_v34": float(weights[1]),
                "sparse_feature": float(weights[2]),
                "svd_feature": float(weights[3]),
            },
            "creator_boost_alpha": float(creator_boost_alpha),
            "creator_binary_creator_prob": creator_binary_prob,
            "creator_vs_performer_creator_prob": creator_vs_performer_prob,
            "creator_signal": creator_signal,
            "bertweet_v3_pred_label": b["bertweet_v3_pred_label"],
            "bertweet_v34_pred_label": b["bertweet_v34_pred_label"],
            "sparse_feature_pred_label": sp["feature_pred_label"],
            "svd_feature_pred_label": sv["svd_feature_pred_label"],
            "fusion_probabilities": final_probs.tolist(),
            "fusion_pred_label": pred_label,
            "fusion_model": "hybrid_v4_weighted_average_creator_boost"
                if creator_boost_alpha > 0
                else "hybrid_v4_weighted_average",
        })

    metrics = evaluate_predictions(y_true, y_pred, labels)
    metrics.update({
        "target": target,
        "split": split,
        "model": "hybrid_v4_weighted_average_creator_boost"
        if creator_boost_alpha > 0
        else "hybrid_v4_weighted_average",
        "num_rows": len(common_ids),
        "creator_boost_alpha": float(creator_boost_alpha),
        "weights": {
            "bertweet_v3": float(weights[0]),
            "bertweet_v34": float(weights[1]),
            "sparse_feature": float(weights[2]),
            "svd_feature": float(weights[3]),
        },
    })

    pred_path = os.path.join(
        hybrid_v4_fusion_predictions_dir,
        f"{target}_{split}_weighted_fusion_predictions_{suffix}.json",
    )
    metrics_path = os.path.join(
        hybrid_v4_fusion_metrics_dir,
        f"{target}_{split}_weighted_fusion_metrics_{suffix}.json",
    )

    save_json(predictions, pred_path)
    save_json(metrics, metrics_path)

    print(
        f"[RESULT] {target} {split} weighted fusion "
        f"weights={weights.tolist()} "
        f"acc={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print(f"[OK] Saved predictions: {pred_path}")
    print(f"[OK] Saved metrics:     {metrics_path}")

    return metrics


def grid_search_val_then_test(target: str):
    labels = LABEL_ORDERS[target]

    candidates = []

    # Simple coarse grid.
    values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for w in product(values, repeat=4):
        if sum(w) == 0:
            continue

        weights = normalize_weights(w)

        try:
            metrics = run_weighted_fusion(
                target=target,
                split="val",
                weights=weights,
                suffix="grid_tmp",
            )
        except Exception:
            continue

        candidates.append({
            "weights": weights.tolist(),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
        })

    if not candidates:
        raise ValueError("No valid weight candidates found on val.")

    candidates.sort(
        key=lambda x: (x["val_macro_f1"], x["val_accuracy"]),
        reverse=True,
    )

    best = candidates[0]

    print("\n========== Best validation weights ==========")
    print(json.dumps(best, indent=2))

    # Save grid summary
    summary_path = os.path.join(
        hybrid_v4_fusion_metrics_dir,
        f"{target}_weighted_fusion_grid_search_summary.json",
    )
    save_json(candidates[:50], summary_path)

    # Evaluate best on test
    run_weighted_fusion(
        target=target,
        split="test",
        weights=best["weights"],
        suffix="best_val_weights",
    )


def main():
    parser = argparse.ArgumentParser(
        description="HybridV4 weighted-average late fusion."
    )
    parser.add_argument(
        "--target",
        choices=["occupation"],
        default="occupation",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )
    parser.add_argument(
        "--weights",
        nargs=4,
        type=float,
        default=[0.5, 0.3, 0.2, 0.0],
        help="Weights for: bertweet_v3 bertweet_v34 sparse_feature svd_feature",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Grid-search weights on val and evaluate best on test.",
    )

    parser.add_argument(
        "--creator-boost-alpha",
        type=float,
        default=0.0,
        help="Boost creator probability using creator gate signals. 0 disables it.",
    )

    args = parser.parse_args()

    ensure_dirs()

    if args.grid_search:
        grid_search_val_then_test(args.target)
    else:
        suffix = "_".join(str(w).replace(".", "p") for w in args.weights)

        if args.creator_boost_alpha > 0:
            alpha_suffix = str(args.creator_boost_alpha).replace(".", "p")
            suffix = f"{suffix}_creatorboost_{alpha_suffix}"
        run_weighted_fusion(
            target=args.target,
            split=args.split,
            weights=args.weights,
            suffix=suffix,
            creator_boost_alpha=args.creator_boost_alpha,
        )


if __name__ == "__main__":
    main()
