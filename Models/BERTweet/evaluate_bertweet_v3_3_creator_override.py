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
)

LABELS = ["sports", "performer", "creator", "politics"]

# occupation label order:
# ["sports", "performer", "creator", "politics"]
CREATOR_INDEX_4CLASS = 2

# creator_binary label order:
# ["not_creator", "creator"]
CREATOR_INDEX_BINARY = 1


THRESHOLD_GRID = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]

MIN_P4_CREATOR_GRID = [
    0.00,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]


def ensure_dirs():
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3_3")
    predictions_dir = os.path.join(output_dir, "predictions")
    metrics_dir = os.path.join(output_dir, "metrics")
    plots_dir = os.path.join(output_dir, "plots")

    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    return output_dir, predictions_dir, metrics_dir, plots_dir


def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def by_id(rows: List[dict]) -> Dict[str, dict]:
    return {str(row["celebrity_id"]): row for row in rows}


def resolve_existing_path(candidates: List[str], label: str) -> str:
    for path in candidates:
        if os.path.exists(path):
            print(f"[INFO] {label}: {path}")
            return path

    print(f"[ERROR] Could not find {label}. Tried:")
    for path in candidates:
        print(f"  - {path}")
    raise FileNotFoundError(label)


def p_creator_binary(row: dict) -> float:
    return float(row["probabilities"][CREATOR_INDEX_BINARY])


def p_creator_4class(row: dict) -> float:
    return float(row["probabilities"][CREATOR_INDEX_4CLASS])


def apply_creator_override(
    occupation_4class: List[dict],
    creator_binary: List[dict],
    threshold: float,
    min_p4_creator: float,
) -> List[dict]:
    occ4 = by_id(occupation_4class)
    bin2 = by_id(creator_binary)

    common_ids = sorted(set(occ4.keys()) & set(bin2.keys()))

    predictions = []

    for cid in common_ids:
        row4 = occ4[cid]
        row2 = bin2[cid]

        pred4 = str(row4["pred_label"])

        pc_bin = p_creator_binary(row2)
        pc_4 = p_creator_4class(row4)

        if pc_bin >= threshold and pc_4 >= min_p4_creator:
            final_pred = "creator"
            decision = "creator_override"
        else:
            final_pred = pred4
            decision = "use_4class"

        predictions.append({
            "celebrity_id": cid,
            "true_label": row4["true_label"],
            "pred_label": final_pred,

            "pred_4class": pred4,
            "pred_creator_binary": row2["pred_label"],

            "p_creator_binary": pc_bin,
            "p_creator_4class": pc_4,

            "threshold": threshold,
            "min_p4_creator": min_p4_creator,
            "decision": decision,

            "probabilities_4class": row4.get("probabilities"),
            "probabilities_creator_binary": row2.get("probabilities"),

            "version": "bertweet_v3_3",
            "strategy": "creator_binary_override_4class",
        })

    return predictions


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


def decision_distribution(predictions: List[dict]) -> Dict[str, int]:
    counts = {}

    for row in predictions:
        key = row["decision"]
        counts[key] = counts.get(key, 0) + 1

    return counts


def threshold_search(
    occupation_4class_val: List[dict],
    creator_binary_val: List[dict],
) -> Tuple[dict, List[dict]]:
    results = []

    for threshold in THRESHOLD_GRID:
        for min_p4_creator in MIN_P4_CREATOR_GRID:
            preds = apply_creator_override(
                occupation_4class=occupation_4class_val,
                creator_binary=creator_binary_val,
                threshold=threshold,
                min_p4_creator=min_p4_creator,
            )

            metrics = compute_metrics(preds)
            creator_report = metrics["classification_report"].get("creator", {})

            results.append({
                "threshold": threshold,
                "min_p4_creator": min_p4_creator,
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "val_weighted_f1": metrics["weighted_f1"],
                "creator_precision": creator_report.get("precision", 0.0),
                "creator_recall": creator_report.get("recall", 0.0),
                "creator_f1": creator_report.get("f1-score", 0.0),
                "decision_distribution": decision_distribution(preds),
            })

    # Hauptziel: Macro-F1.
    # Tie-breaker: Creator-F1.
    best = sorted(
        results,
        key=lambda r: (r["val_macro_f1"], r["creator_f1"]),
        reverse=True,
    )[0]

    return best, results


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    output_path: str,
    normalized: bool = False,
):
    if normalized:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
    else:
        cm_plot = cm

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_plot)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if normalized:
                text = f"{cm_plot[i, j]:.2f}\n({cm[i, j]})"
            else:
                text = str(cm[i, j])

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_id_debug(name: str, rows: List[dict]) -> set:
    ids = set(str(row["celebrity_id"]) for row in rows)
    print(f"[DEBUG] {name}: {len(ids)} ids")
    print(f"[DEBUG] {name} first 10: {sorted(ids)[:10]}")
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BERTweet V3.3 Creator Override"
    )
    parser.add_argument(
        "--use-fixed-thresholds",
        action="store_true",
        help="Use CLI thresholds instead of validation threshold search.",
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-p4-creator", type=float, default=0.00)

    args = parser.parse_args()

    _, predictions_dir, metrics_dir, plots_dir = ensure_dirs()

    occupation_4class_val_path = resolve_existing_path(
        [
            os.path.join(bertweet_v3_predictions_dir, "occupation_val_all_predictions.json"),
            os.path.join(bertweet_v3_predictions_dir, "occupation_val_predictions.json"),
        ],
        "val 4class occupation predictions",
    )

    occupation_4class_test_path = resolve_existing_path(
        [
            os.path.join(bertweet_v3_predictions_dir, "occupation_test_all_predictions.json"),
            os.path.join(bertweet_v3_predictions_dir, "occupation_test_predictions.json"),
        ],
        "test 4class occupation predictions",
    )

    creator_binary_val_path = resolve_existing_path(
        [
            os.path.join(bertweet_v3_predictions_dir, "creator_binary_val_all_predictions.json"),
            os.path.join(bertweet_v3_predictions_dir, "creator_binary_val_predictions.json"),
        ],
        "val creator binary predictions",
    )

    creator_binary_test_path = resolve_existing_path(
        [
            os.path.join(bertweet_v3_predictions_dir, "creator_binary_test_predictions.json"),
            os.path.join(bertweet_v3_predictions_dir, "creator_binary_test_all_predictions.json"),
        ],
        "test creator binary predictions",
    )

    occupation_4class_val = load_json(occupation_4class_val_path)
    occupation_4class_test = load_json(occupation_4class_test_path)
    creator_binary_val = load_json(creator_binary_val_path)
    creator_binary_test = load_json(creator_binary_test_path)

    ids_occ4_val = print_id_debug("occupation_4class_val", occupation_4class_val)
    ids_bin_val = print_id_debug("creator_binary_val", creator_binary_val)
    print(f"[DEBUG] val overlap: {len(ids_occ4_val & ids_bin_val)}")

    ids_occ4_test = print_id_debug("occupation_4class_test", occupation_4class_test)
    ids_bin_test = print_id_debug("creator_binary_test", creator_binary_test)
    print(f"[DEBUG] test overlap: {len(ids_occ4_test & ids_bin_test)}")

    if args.use_fixed_thresholds:
        best = {
            "threshold": args.threshold,
            "min_p4_creator": args.min_p4_creator,
            "source": "fixed_cli",
        }
        search_results = []
    else:
        best, search_results = threshold_search(
            occupation_4class_val=occupation_4class_val,
            creator_binary_val=creator_binary_val,
        )
        best["source"] = "val_threshold_search"

    threshold = float(best["threshold"])
    min_p4_creator = float(best["min_p4_creator"])

    print("\n========== BEST V3.3 THRESHOLDS ==========")
    print(f"[INFO] threshold:      {threshold}")
    print(f"[INFO] min_p4_creator: {min_p4_creator}")
    if "val_macro_f1" in best:
        print(f"[INFO] val_macro_f1:   {best['val_macro_f1']:.4f}")
    if "creator_f1" in best:
        print(f"[INFO] val creator F1: {best['creator_f1']:.4f}")

    val_preds = apply_creator_override(
        occupation_4class=occupation_4class_val,
        creator_binary=creator_binary_val,
        threshold=threshold,
        min_p4_creator=min_p4_creator,
    )

    test_preds = apply_creator_override(
        occupation_4class=occupation_4class_test,
        creator_binary=creator_binary_test,
        threshold=threshold,
        min_p4_creator=min_p4_creator,
    )

    val_metrics = compute_metrics(val_preds)
    test_metrics = compute_metrics(test_preds)

    val_metrics.update({
        "version": "bertweet_v3_3",
        "split": "val",
        "threshold": threshold,
        "min_p4_creator": min_p4_creator,
        "decision_distribution": decision_distribution(val_preds),
        "best_threshold_record": best,
    })

    test_metrics.update({
        "version": "bertweet_v3_3",
        "split": "test",
        "threshold": threshold,
        "min_p4_creator": min_p4_creator,
        "decision_distribution": decision_distribution(test_preds),
        "best_threshold_record": best,
    })

    save_json(
        search_results,
        os.path.join(metrics_dir, "occupation_v33_threshold_search.json"),
    )
    save_json(
        val_metrics,
        os.path.join(metrics_dir, "occupation_v33_val_metrics.json"),
    )
    save_json(
        test_metrics,
        os.path.join(metrics_dir, "occupation_v33_test_metrics.json"),
    )
    save_json(
        val_preds,
        os.path.join(predictions_dir, "occupation_v33_val_predictions.json"),
    )
    save_json(
        test_preds,
        os.path.join(predictions_dir, "occupation_v33_test_predictions.json"),
    )

    test_cm = np.array(test_metrics["confusion_matrix"])

    plot_confusion_matrix(
        cm=test_cm,
        labels=LABELS,
        title="BERTweet V3.3 Occupation Test Confusion Matrix",
        output_path=os.path.join(plots_dir, "occupation_v33_test_confusion_matrix.png"),
        normalized=False,
    )

    plot_confusion_matrix(
        cm=test_cm,
        labels=LABELS,
        title="BERTweet V3.3 Occupation Test Confusion Matrix Normalized",
        output_path=os.path.join(plots_dir, "occupation_v33_test_confusion_matrix_normalized.png"),
        normalized=True,
    )

    print("\n========== V3.3 RESULTS ==========")
    print(f"[VAL]  acc={val_metrics['accuracy']:.4f} macro_f1={val_metrics['macro_f1']:.4f}")
    print(f"[TEST] acc={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"[TEST] decision_distribution={test_metrics['decision_distribution']}")

    creator_report = test_metrics["classification_report"]["creator"]
    print(
        "[TEST] creator "
        f"precision={creator_report['precision']:.4f} "
        f"recall={creator_report['recall']:.4f} "
        f"f1={creator_report['f1-score']:.4f}"
    )

    print(f"[OK] Saved metrics:     {metrics_dir}")
    print(f"[OK] Saved predictions: {predictions_dir}")
    print(f"[OK] Saved plots:       {plots_dir}")


if __name__ == "__main__":
    main()