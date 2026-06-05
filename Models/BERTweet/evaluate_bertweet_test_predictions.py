import argparse
import json
import os
import sys
from typing import Dict, List

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

from Models.BERTweet.config_bertweet_model import LABEL_ORDERS


TARGETS = ["occupation", "gender", "birthyear"]


def ensure_dirs():
    os.makedirs(bertweet_v3_test_metrics_dir, exist_ok=True)

    plots_dir = os.path.join(bertweet_v3_test_metrics_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    return plots_dir


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    output_path: str,
):
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
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_normalized_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    output_path: str,
):
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
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_class_f1(
    report: Dict,
    labels: List[str],
    title: str,
    output_path: str,
):
    f1_values = [report[label]["f1-score"] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(labels, f1_values)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("F1-score")

    for idx, value in enumerate(f1_values):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate_one_target(target: str, plots_dir: str):
    print(f"\n========== Evaluating BERTweet V3 TEST TARGET: {target} ==========")

    labels = LABEL_ORDERS[target]

    predictions_path = os.path.join(
        bertweet_v3_predictions_dir,
        f"{target}_test_predictions.json",
    )

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(
            f"Missing predictions file: {predictions_path}. "
            f"Run predict_bertweet.py first."
        )

    predictions = load_json(predictions_path)

    y_true = [row["true_label"] for row in predictions]
    y_pred = [row["pred_label"] for row in predictions]

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

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "target_label": target,
        "version": "bertweet_v3",
        "split": "test",
        "num_test_celebrities": len(predictions),
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = os.path.join(
        bertweet_v3_test_metrics_dir,
        f"{target}_test_metrics.json",
    )

    cm_path = os.path.join(
        plots_dir,
        f"{target}_test_confusion_matrix.png",
    )

    cm_norm_path = os.path.join(
        plots_dir,
        f"{target}_test_confusion_matrix_normalized.png",
    )

    f1_path = os.path.join(
        plots_dir,
        f"{target}_test_per_class_f1.png",
    )

    save_json(metrics, metrics_path)

    plot_confusion_matrix(
        cm=cm,
        labels=labels,
        title=f"BERTweet V3 {target} Test Confusion Matrix",
        output_path=cm_path,
    )

    plot_normalized_confusion_matrix(
        cm=cm,
        labels=labels,
        title=f"BERTweet V3 {target} Test Confusion Matrix Normalized",
        output_path=cm_norm_path,
    )

    plot_per_class_f1(
        report=report,
        labels=labels,
        title=f"BERTweet V3 {target} Test Per-Class F1",
        output_path=f1_path,
    )

    print(f"[RESULT] {target} test_acc={acc:.4f} test_macro_f1={macro_f1:.4f}")
    print(f"[OK] Saved metrics: {metrics_path}")
    print(f"[OK] Saved plot:    {cm_path}")
    print(f"[OK] Saved plot:    {cm_norm_path}")
    print(f"[OK] Saved plot:    {f1_path}")


def resolve_targets(target: str):
    if target == "all":
        return TARGETS
    return [target]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BERTweet V3 test predictions"
    )
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear", "all"],
        default="all",
    )

    args = parser.parse_args()

    plots_dir = ensure_dirs()

    for target in resolve_targets(args.target):
        evaluate_one_target(target, plots_dir)


if __name__ == "__main__":
    main()