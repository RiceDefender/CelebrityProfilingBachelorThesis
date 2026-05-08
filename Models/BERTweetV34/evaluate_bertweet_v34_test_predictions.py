import argparse
import json
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from _constants import bertweet_v34_predictions_dir, bertweet_v34_test_metrics_dir, bertweet_v34_age_bins_path
from Models.BERTweetV34.config_bertweet_v34_model import LABEL_ORDERS, VERSION
from Models.BERTweetV34.age_bins_v34 import load_age_bins, age_bin_display_name

TARGETS = ["occupation", "gender", "birthyear"]


def ensure_dirs():
    os.makedirs(bertweet_v34_test_metrics_dir, exist_ok=True)
    plots_dir = os.path.join(bertweet_v34_test_metrics_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def display_labels_for_target(target: str, labels: List[str]) -> List[str]:
    if target != "birthyear_8range":
        return labels
    try:
        bins = load_age_bins(bertweet_v34_age_bins_path)
        return [age_bin_display_name(label, bins) for label in labels]
    except FileNotFoundError:
        return labels


def plot_confusion_matrix(cm, labels, title, output_path, normalized=False):
    matrix = cm.astype(float)
    if normalized:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), max(6, len(labels) * 0.8)))
    im = ax.imshow(matrix, vmin=0 if normalized else None, vmax=1 if normalized else None)
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
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_class_f1(report, labels, display_labels, title, output_path):
    f1_values = [report[label]["f1-score"] for label in labels]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(display_labels, f1_values)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("F1-score")
    ax.tick_params(axis="x", rotation=45)
    for idx, value in enumerate(f1_values):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate_one_target(target: str, plots_dir: str):
    print(f"\n========== Evaluating BERTweet V3.4 TEST TARGET: {target} ==========")
    labels = LABEL_ORDERS[target]
    display_labels = display_labels_for_target(target, labels)
    predictions_path = os.path.join(bertweet_v34_predictions_dir, f"{target}_test_predictions.json")
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}. Run predict_bertweet_v34.py first.")
    predictions = load_json(predictions_path)
    y_true = [row["true_label"] for row in predictions]
    y_pred = [row["pred_label"] for row in predictions]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "target_label": target,
        "version": VERSION,
        "split": "test",
        "num_test_celebrities": len(predictions),
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
        "labels": labels,
        "display_labels": display_labels,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    if target == "birthyear_8range":
        try:
            metrics["age_bins"] = load_age_bins(bertweet_v34_age_bins_path)
        except FileNotFoundError:
            pass
    metrics_path = os.path.join(bertweet_v34_test_metrics_dir, f"{target}_test_metrics.json")
    save_json(metrics, metrics_path)
    plot_confusion_matrix(cm, display_labels, f"BERTweet V3.4 {target} Test Confusion Matrix", os.path.join(plots_dir, f"{target}_test_confusion_matrix.png"))
    plot_confusion_matrix(cm, display_labels, f"BERTweet V3.4 {target} Test Confusion Matrix Normalized", os.path.join(plots_dir, f"{target}_test_confusion_matrix_normalized.png"), normalized=True)
    plot_per_class_f1(report, labels, display_labels, f"BERTweet V3.4 {target} Test Per-Class F1", os.path.join(plots_dir, f"{target}_test_per_class_f1.png"))
    print(f"[RESULT] {target} test_acc={acc:.4f} test_macro_f1={macro_f1:.4f}")
    print(f"[OK] Saved metrics: {metrics_path}")


def resolve_targets(target):
    if target == "all":
        return TARGETS
    if target == "all_with_age8":
        return TARGETS + ["birthyear_8range"]
    return [target]


def main():
    parser = argparse.ArgumentParser(description="Evaluate BERTweet V3.4 test predictions")
    parser.add_argument("--target", choices=["occupation", "gender", "birthyear", "birthyear_8range", "all", "all_with_age8"], default="all")
    args = parser.parse_args()
    plots_dir = ensure_dirs()
    for target in resolve_targets(args.target):
        evaluate_one_target(target, plots_dir)


if __name__ == "__main__":
    main()
