import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from _constants import (
    bertweet_v3_predictions_dir,
    bertweet_v3_test_metrics_dir,
)

from Models.BERTweet.config_bertweet_model import (
    LABEL_ORDERS,
    V31_THRESHOLD_GRID_LOW,
    V31_THRESHOLD_GRID_HIGH,
)


LABELS = LABEL_ORDERS["occupation"]


def ensure_dirs():
    os.makedirs(bertweet_v3_test_metrics_dir, exist_ok=True)

    plots_dir = os.path.join(bertweet_v3_test_metrics_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    return plots_dir


def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def by_id(predictions: List[dict]) -> Dict[str, dict]:
    return {str(row["celebrity_id"]): row for row in predictions}


def p_creator(row: dict) -> float:
    """
    creator_binary label order:
    ["not_creator", "creator"]
    """
    return float(row["probabilities"][1])


def pred_3class(row: dict) -> str:
    return str(row["pred_label"])


def pred_4class(row: dict) -> str:
    return str(row["pred_label"])


def apply_gating(
    occupation_4class: List[dict],
    creator_binary: List[dict],
    occupation_3class: List[dict],
    low_threshold: float,
    high_threshold: float,
) -> List[dict]:
    occ4 = by_id(occupation_4class)
    bin2 = by_id(creator_binary)
    occ3 = by_id(occupation_3class)

    common_ids = sorted(set(occ4.keys()) & set(bin2.keys()) & set(occ3.keys()))

    gated_predictions = []

    for cid in common_ids:
        row4 = occ4[cid]
        row2 = bin2[cid]
        row3 = occ3[cid]

        pc = p_creator(row2)

        pred4 = pred_4class(row4)
        pred3 = pred_3class(row3)

        if pc >= high_threshold:
            final_pred = "creator"
            gate_decision = "creator_high"

        elif pc <= low_threshold:
            final_pred = pred3
            gate_decision = "not_creator_low_use_3class"

        else:
            final_pred = pred4
            gate_decision = "uncertain_use_4class"

        gated_predictions.append({
            "celebrity_id": cid,
            "true_label": row4["true_label"],
            "pred_label": final_pred,

            "p_creator": pc,
            "gate_decision": gate_decision,
            "threshold_low": low_threshold,
            "threshold_high": high_threshold,

            "pred_4class": pred4,
            "pred_3class": pred3,
            "pred_creator_binary": row2["pred_label"],

            "probabilities_4class": row4.get("probabilities"),
            "probabilities_3class": row3.get("probabilities"),
            "probabilities_creator_binary": row2.get("probabilities"),

            "version": "bertweet_v3_1",
            "model_name": "vinai/bertweet-base",
            "voting_strategy": "creator_gated",
        })

    return gated_predictions


def compute_metrics(predictions: List[dict]) -> Dict:
    y_true = [row["true_label"] for row in predictions]
    y_pred = [row["pred_label"] for row in predictions]

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    return {
        "num_celebrities": len(predictions),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
    }


def gate_distribution(predictions: List[dict]) -> Dict[str, int]:
    counts = {}

    for row in predictions:
        key = row["gate_decision"]
        counts[key] = counts.get(key, 0) + 1

    return counts


def threshold_search(
    occupation_4class_val: List[dict],
    creator_binary_val: List[dict],
    occupation_3class_val: List[dict],
) -> Tuple[dict, List[dict]]:
    results = []

    for low in V31_THRESHOLD_GRID_LOW:
        for high in V31_THRESHOLD_GRID_HIGH:
            if low >= high:
                continue

            gated = apply_gating(
                occupation_4class=occupation_4class_val,
                creator_binary=creator_binary_val,
                occupation_3class=occupation_3class_val,
                low_threshold=low,
                high_threshold=high,
            )

            metrics = compute_metrics(gated)

            creator_report = metrics["classification_report"].get("creator", {})

            results.append({
                "low_threshold": low,
                "high_threshold": high,
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "creator_precision": creator_report.get("precision", 0.0),
                "creator_recall": creator_report.get("recall", 0.0),
                "creator_f1": creator_report.get("f1-score", 0.0),
                "gate_distribution": gate_distribution(gated),
            })

    # Primary criterion: Macro-F1
    # Tie-breaker: creator F1
    best = sorted(
        results,
        key=lambda r: (r["val_macro_f1"], r["creator_f1"]),
        reverse=True,
    )[0]

    return best, results


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_normalized_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, output_path: str):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm_norm, vmin=0, vmax=1)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BERTweet V3.1 creator-gated occupation")
    parser.add_argument(
        "--use-fixed-thresholds",
        action="store_true",
        help="Use thresholds passed by CLI instead of Val threshold search.",
    )
    parser.add_argument("--low", type=float, default=0.35)
    parser.add_argument("--high", type=float, default=0.60)

    args = parser.parse_args()

    plots_dir = ensure_dirs()

    # Existing V3 4-class predictions
    occupation_4class_val_path = os.path.join(
        bertweet_v3_predictions_dir,
        "occupation_val_all_predictions.json",
    )
    occupation_4class_test_path = os.path.join(
        bertweet_v3_predictions_dir,
        "occupation_test_all_predictions.json",
    )

    # New V3.1 auxiliary predictions
    creator_binary_val_path = os.path.join(
        bertweet_v3_predictions_dir,
        "creator_binary_val_all_predictions.json",
    )
    creator_binary_test_path = os.path.join(
        bertweet_v3_predictions_dir,
        "creator_binary_test_predictions.json",
    )
    occupation_3class_val_path = os.path.join(
        bertweet_v3_predictions_dir,
        "occupation_3class_val_all_predictions.json",
    )
    occupation_3class_test_path = os.path.join(
        bertweet_v3_predictions_dir,
        "occupation_3class_test_all_predictions.json",
    )

    occupation_4class_val = load_json(occupation_4class_val_path)
    occupation_4class_test = load_json(occupation_4class_test_path)

    creator_binary_val = load_json(creator_binary_val_path)
    creator_binary_test = load_json(creator_binary_test_path)

    occupation_3class_val = load_json(occupation_3class_val_path)
    occupation_3class_test = load_json(occupation_3class_test_path)

    def print_id_overlap(name, predictions):
        ids = set(str(row["celebrity_id"]) for row in predictions)
        print(f"[DEBUG] {name}: {len(ids)} ids")
        print(f"[DEBUG] {name} first 10:", sorted(ids)[:10])
        return ids

    ids_occ4_val = print_id_overlap("occupation_4class_val", occupation_4class_val)
    ids_bin_val = print_id_overlap("creator_binary_val", creator_binary_val)
    ids_3_val = print_id_overlap("occupation_3class_val", occupation_3class_val)

    print("[DEBUG] occ4 ∩ bin:", len(ids_occ4_val & ids_bin_val))
    print("[DEBUG] occ4 ∩ 3class:", len(ids_occ4_val & ids_3_val))
    print("[DEBUG] bin ∩ 3class:", len(ids_bin_val & ids_3_val))
    print("[DEBUG] all three:", len(ids_occ4_val & ids_bin_val & ids_3_val))

    if args.use_fixed_thresholds:
        best = {
            "low_threshold": args.low,
            "high_threshold": args.high,
            "source": "fixed_cli",
        }
        search_results = []
    else:
        best, search_results = threshold_search(
            occupation_4class_val=occupation_4class_val,
            creator_binary_val=creator_binary_val,
            occupation_3class_val=occupation_3class_val,
        )
        best["source"] = "val_threshold_search"

    low = float(best["low_threshold"])
    high = float(best["high_threshold"])

    print("\n========== BEST V3.1 THRESHOLDS ==========")
    print(f"[INFO] low_threshold:  {low}")
    print(f"[INFO] high_threshold: {high}")
    if "val_macro_f1" in best:
        print(f"[INFO] val_macro_f1:   {best['val_macro_f1']:.4f}")
    if "creator_f1" in best:
        print(f"[INFO] val creator F1: {best['creator_f1']:.4f}")

    val_gated = apply_gating(
        occupation_4class=occupation_4class_val,
        creator_binary=creator_binary_val,
        occupation_3class=occupation_3class_val,
        low_threshold=low,
        high_threshold=high,
    )

    test_gated = apply_gating(
        occupation_4class=occupation_4class_test,
        creator_binary=creator_binary_test,
        occupation_3class=occupation_3class_test,
        low_threshold=low,
        high_threshold=high,
    )

    val_metrics = compute_metrics(val_gated)
    test_metrics = compute_metrics(test_gated)

    val_metrics.update({
        "version": "bertweet_v3_1",
        "split": "val",
        "low_threshold": low,
        "high_threshold": high,
        "gate_distribution": gate_distribution(val_gated),
        "best_threshold_record": best,
    })

    test_metrics.update({
        "version": "bertweet_v3_1",
        "split": "test",
        "low_threshold": low,
        "high_threshold": high,
        "gate_distribution": gate_distribution(test_gated),
        "best_threshold_record": best,
    })

    threshold_search_path = os.path.join(
        bertweet_v3_test_metrics_dir,
        "occupation_v31_threshold_search.json",
    )
    val_metrics_path = os.path.join(
        bertweet_v3_test_metrics_dir,
        "occupation_v31_val_metrics.json",
    )
    test_metrics_path = os.path.join(
        bertweet_v3_test_metrics_dir,
        "occupation_v31_test_metrics.json",
    )
    val_predictions_path = os.path.join(
        bertweet_v3_predictions_dir,
        "occupation_v31_val_predictions.json",
    )
    test_predictions_path = os.path.join(
        bertweet_v3_predictions_dir,
        "occupation_v31_test_predictions.json",
    )

    save_json(search_results, threshold_search_path)
    save_json(val_metrics, val_metrics_path)
    save_json(test_metrics, test_metrics_path)
    save_json(val_gated, val_predictions_path)
    save_json(test_gated, test_predictions_path)

    test_cm = np.array(test_metrics["confusion_matrix"])

    cm_path = os.path.join(
        plots_dir,
        "occupation_v31_test_confusion_matrix.png",
    )
    cm_norm_path = os.path.join(
        plots_dir,
        "occupation_v31_test_confusion_matrix_normalized.png",
    )

    plot_confusion_matrix(
        cm=test_cm,
        labels=LABELS,
        title="BERTweet V3.1 Occupation Test Confusion Matrix",
        output_path=cm_path,
    )

    plot_normalized_confusion_matrix(
        cm=test_cm,
        labels=LABELS,
        title="BERTweet V3.1 Occupation Test Confusion Matrix Normalized",
        output_path=cm_norm_path,
    )

    print("\n========== V3.1 RESULTS ==========")
    print(f"[VAL]  acc={val_metrics['accuracy']:.4f} macro_f1={val_metrics['macro_f1']:.4f}")
    print(f"[TEST] acc={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"[TEST] gate_distribution={test_metrics['gate_distribution']}")
    print(f"[OK] Saved threshold search: {threshold_search_path}")
    print(f"[OK] Saved val metrics:      {val_metrics_path}")
    print(f"[OK] Saved test metrics:     {test_metrics_path}")
    print(f"[OK] Saved val predictions:  {val_predictions_path}")
    print(f"[OK] Saved test predictions: {test_predictions_path}")
    print(f"[OK] Saved plot:            {cm_path}")
    print(f"[OK] Saved plot:            {cm_norm_path}")


if __name__ == "__main__":
    main()