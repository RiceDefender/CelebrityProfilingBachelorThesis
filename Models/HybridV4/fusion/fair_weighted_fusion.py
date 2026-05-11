import argparse
import json
import os
import sys
from itertools import product
from typing import Dict, List, Optional

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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def rows_by_id(rows: List[dict]) -> Dict[str, dict]:
    return {str(row["celebrity_id"]): row for row in rows}


def split_filename(split: str) -> str:
    if split == "fusion_val":
        return "fusion_val"
    if split == "test":
        return "test"
    raise ValueError(f"Unsupported split: {split}")


def load_rows(target: str, split: str):
    split_name = split_filename(split)

    v3_path = os.path.join(
        hybrid_v4_bertweet_probs_dir,
        f"{target}_{split_name}_bertweet_v3_probs.json",
    )
    v34_path = os.path.join(
        hybrid_v4_bertweet_probs_dir,
        f"{target}_{split_name}_bertweet_v34_probs.json",
    )
    sparse_path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{target}_{split_name}_feature_probs.json",
    )

    v3 = rows_by_id(load_json(v3_path))
    v34 = rows_by_id(load_json(v34_path))
    sparse = rows_by_id(load_json(sparse_path))

    common_ids = sorted(
        set(v3.keys()) & set(v34.keys()) & set(sparse.keys()),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    if not common_ids:
        raise ValueError(f"No common IDs for target={target}, split={split}")

    return v3, v34, sparse, common_ids


def load_gate_rows(gate: str, split: str) -> Optional[Dict[str, dict]]:
    split_name = split_filename(split)
    path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{gate}_{split_name}_feature_probs.json",
    )

    if not os.path.exists(path):
        return None

    return rows_by_id(load_json(path))


def get_binary_prob(row: dict, positive_label: str) -> float:
    labels = row["labels"]
    probs = row["feature_probabilities"]

    if positive_label not in labels:
        raise ValueError(f"{positive_label} not found in labels={labels}")

    return float(probs[labels.index(positive_label)])


def evaluate_predictions(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
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
    total = float(np.sum(weights))

    if total <= 0:
        raise ValueError("Weight sum must be > 0.")

    return weights / total


def run_fusion(
    target: str,
    split: str,
    weights,
    creator_boost_alpha: float = 0.0,
    save_outputs: bool = False,
    suffix: str = "tmp",
):
    labels = LABEL_ORDERS[target]
    weights = normalize_weights(weights)

    v3, v34, sparse, common_ids = load_rows(target, split)

    creator_binary = None
    creator_vs_performer = None

    if target == "occupation" and creator_boost_alpha > 0:
        creator_binary = load_gate_rows("creator_binary", split)
        creator_vs_performer = load_gate_rows("creator_vs_performer", split)

    y_true = []
    y_pred = []
    predictions = []

    for cid in common_ids:
        r_v3 = v3[cid]
        r_v34 = v34[cid]
        r_sparse = sparse[cid]

        true_label = r_v3["true_label"]

        if true_label != r_v34["true_label"] or true_label != r_sparse["true_label"]:
            raise ValueError(
                f"Label mismatch for cid={cid}: "
                f"v3={true_label}, v34={r_v34['true_label']}, sparse={r_sparse['true_label']}"
            )

        p_v3 = np.asarray(r_v3["bertweet_v3_probabilities"], dtype=np.float64)
        p_v34 = np.asarray(r_v34["bertweet_v34_probabilities"], dtype=np.float64)
        p_sparse = np.asarray(r_sparse["feature_probabilities"], dtype=np.float64)

        final_probs = (
            weights[0] * p_v3
            + weights[1] * p_v34
            + weights[2] * p_sparse
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
            },
            "creator_boost_alpha": float(creator_boost_alpha),
            "creator_binary_creator_prob": creator_binary_prob,
            "creator_vs_performer_creator_prob": creator_vs_performer_prob,
            "creator_signal": creator_signal,
            "bertweet_v3_pred_label": r_v3["bertweet_v3_pred_label"],
            "bertweet_v34_pred_label": r_v34["bertweet_v34_pred_label"],
            "sparse_feature_pred_label": r_sparse["feature_pred_label"],
            "fusion_probabilities": final_probs.tolist(),
            "fusion_pred_label": pred_label,
            "fusion_model": "hybrid_v4_fair_weighted_average",
        })

    metrics = evaluate_predictions(y_true, y_pred, labels)
    metrics.update({
        "target": target,
        "split": split,
        "model": "hybrid_v4_fair_weighted_average",
        "num_rows": len(common_ids),
        "weights": {
            "bertweet_v3": float(weights[0]),
            "bertweet_v34": float(weights[1]),
            "sparse_feature": float(weights[2]),
        },
        "creator_boost_alpha": float(creator_boost_alpha),
    })

    if save_outputs:
        pred_path = os.path.join(
            hybrid_v4_fusion_predictions_dir,
            f"{target}_{split}_fair_weighted_fusion_predictions_{suffix}.json",
        )
        metrics_path = os.path.join(
            hybrid_v4_fusion_metrics_dir,
            f"{target}_{split}_fair_weighted_fusion_metrics_{suffix}.json",
        )

        save_json(predictions, pred_path)
        save_json(metrics, metrics_path)

        print(f"[OK] Saved predictions: {pred_path}")
        print(f"[OK] Saved metrics:     {metrics_path}")

    return metrics, predictions


def grid_search_on_fusion_val(target: str, fixed_creator_boost_alpha: Optional[float] = None):
    weight_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    if fixed_creator_boost_alpha is not None:
        if target != "occupation" and fixed_creator_boost_alpha != 0.0:
            raise ValueError("--creator-boost-alpha is only meaningful for target=occupation.")
        alpha_values = [float(fixed_creator_boost_alpha)]
    elif target == "occupation":
        alpha_values = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15]
    else:
        alpha_values = [0.0]

    candidates = []

    for weights in product(weight_values, repeat=3):
        if sum(weights) <= 0:
            continue

        weights_norm = normalize_weights(weights)

        for alpha in alpha_values:
            metrics, _ = run_fusion(
                target=target,
                split="fusion_val",
                weights=weights_norm,
                creator_boost_alpha=alpha,
                save_outputs=False,
            )

            candidates.append({
                "weights": weights_norm.tolist(),
                "creator_boost_alpha": float(alpha),
                "fusion_val_accuracy": metrics["accuracy"],
                "fusion_val_macro_f1": metrics["macro_f1"],
            })

    candidates.sort(
        key=lambda row: (row["fusion_val_macro_f1"], row["fusion_val_accuracy"]),
        reverse=True,
    )

    best = candidates[0]

    summary_path = os.path.join(
        hybrid_v4_fusion_metrics_dir,
        f"{target}_fair_weighted_fusion_grid_search_summary.json",
    )
    save_json(candidates, summary_path)

    print("\n========== Best on fusion_val ==========")
    print(json.dumps(best, indent=2))
    print(f"[OK] Saved grid summary: {summary_path}")

    suffix = (
        f"best_val_"
        f"w_{best['weights'][0]:.2f}_{best['weights'][1]:.2f}_{best['weights'][2]:.2f}_"
        f"alpha_{best['creator_boost_alpha']:.3f}"
    )
    suffix = suffix.replace(".", "p")

    test_metrics, _ = run_fusion(
        target=target,
        split="test",
        weights=best["weights"],
        creator_boost_alpha=best["creator_boost_alpha"],
        save_outputs=True,
        suffix=suffix,
    )

    final_report = {
        "target": target,
        "selection_split": "fusion_val",
        "evaluation_split": "test",
        "best_selection": best,
        "test_metrics": test_metrics,
        "note": "Weights and creator boost selected only on fusion_val, then evaluated on official test.",
    }

    final_path = os.path.join(
        hybrid_v4_fusion_metrics_dir,
        f"{target}_fair_weighted_fusion_final_report.json",
    )
    save_json(final_report, final_path)

    print("\n========== Final test result ==========")
    print(
        f"[RESULT] {target} fair fusion "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}"
    )
    print(f"[OK] Saved final report: {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fair HybridV4 weighted fusion: select on fusion_val, evaluate on test."
    )
    parser.add_argument("--target", choices=["occupation", "gender", "birthyear"], default="occupation")
    parser.add_argument(
        "--creator-boost-alpha",
        type=float,
        default=None,
        help="Optional fixed creator boost alpha. If omitted, alpha is selected by validation grid search.",
    )
    args = parser.parse_args()

    ensure_dirs()
    grid_search_on_fusion_val(
        target=args.target,
        fixed_creator_boost_alpha=args.creator_boost_alpha,
    )


if __name__ == "__main__":
    main()