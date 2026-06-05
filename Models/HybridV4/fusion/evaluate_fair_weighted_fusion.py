import argparse
import json
import os
import sys
from typing import List

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
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from _constants import (
    hybrid_v4_fusion_predictions_dir,
    hybrid_v4_fusion_metrics_dir,
)

from Models.HybridV4.feature.config_features import LABEL_ORDERS


def ensure_dirs():
    os.makedirs(hybrid_v4_fusion_metrics_dir, exist_ok=True)

    plots_dir = os.path.join(hybrid_v4_fusion_metrics_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    return plots_dir


def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def find_latest_prediction_file(target: str, split: str) -> str:
    prefix = f"{target}_{split}_fair_weighted_fusion_predictions_"
    suffix = ".json"

    candidates = [
        os.path.join(hybrid_v4_fusion_predictions_dir, filename)
        for filename in os.listdir(hybrid_v4_fusion_predictions_dir)
        if filename.startswith(prefix) and filename.endswith(suffix)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No prediction file found for target={target}, split={split} "
            f"in {hybrid_v4_fusion_predictions_dir}"
        )

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def extract_suffix_from_prediction_path(path: str, target: str, split: str) -> str:
    filename = os.path.basename(path)
    prefix = f"{target}_{split}_fair_weighted_fusion_predictions_"
    suffix = ".json"

    if filename.startswith(prefix) and filename.endswith(suffix):
        return filename[len(prefix):-len(suffix)]

    return "evaluated"


def plot_confusion_matrix(cm, labels: List[str], title: str, output_path: str, normalized=False):
    matrix = cm.astype(float)

    if normalized:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix),
            where=row_sums != 0,
        )

    fig, ax = plt.subplots(
        figsize=(max(8, len(labels) * 1.2), max(6, len(labels) * 0.8))
    )

    im = ax.imshow(
        matrix,
        vmin=0 if normalized else None,
        vmax=1 if normalized else None,
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{matrix[i, j]:.2f}" if normalized else str(int(cm[i, j]))
            ax.text(j, i, text, ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate_prediction_file(target: str, split: str, predictions_path: str = None):
    plots_dir = ensure_dirs()

    labels = LABEL_ORDERS[target]

    if predictions_path is None:
        predictions_path = find_latest_prediction_file(target, split)

    predictions = load_json(predictions_path)

    y_true = [row["true_label"] for row in predictions]
    y_pred = [row["fusion_pred_label"] for row in predictions]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        digits=4,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    suffix = extract_suffix_from_prediction_path(predictions_path, target, split)

    metrics = {
        "target": target,
        "split": split,
        "model": "hybrid_v4_fair_weighted_average",
        "prediction_file": predictions_path,
        "num_rows": len(predictions),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = os.path.join(
        hybrid_v4_fusion_metrics_dir,
        f"{target}_{split}_fair_weighted_fusion_evaluation_{suffix}.json",
    )

    cm_path = os.path.join(
        plots_dir,
        f"{target}_{split}_fair_weighted_fusion_confusion_matrix_{suffix}.png",
    )

    cm_norm_path = os.path.join(
        plots_dir,
        f"{target}_{split}_fair_weighted_fusion_confusion_matrix_normalized_{suffix}.png",
    )

    save_json(metrics, metrics_path)

    plot_confusion_matrix(
        cm,
        labels,
        f"HybridV4 Fair Fusion {target} {split} Confusion Matrix",
        cm_path,
        normalized=False,
    )

    plot_confusion_matrix(
        cm,
        labels,
        f"HybridV4 Fair Fusion {target} {split} Confusion Matrix Normalized",
        cm_norm_path,
        normalized=True,
    )

    print("\n========== Classification report ==========")
    print(report_text)

    print("\n========== Confusion matrix ==========")
    print(f"labels: {labels}")
    print(cm)

    print("\n========== Result ==========")
    print(
        f"[RESULT] {target} {split} "
        f"acc={acc:.4f} macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f}"
    )

    print(f"[OK] Used predictions:                    {predictions_path}")
    print(f"[OK] Saved metrics:                       {metrics_path}")
    print(f"[OK] Saved confusion matrix:              {cm_path}")
    print(f"[OK] Saved normalized confusion matrix:   {cm_norm_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HybridV4 fair weighted fusion predictions."
    )

    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear"],
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["test", "fusion_val"],
        default="test",
    )

    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional path to a specific fair weighted fusion predictions JSON file. "
             "If omitted, the latest matching file is used.",
    )

    args = parser.parse_args()

    evaluate_prediction_file(
        target=args.target,
        split=args.split,
        predictions_path=args.predictions,
    )


if __name__ == "__main__":
    main()