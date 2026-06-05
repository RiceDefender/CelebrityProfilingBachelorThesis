import argparse
import json
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from _constants import (
    sbert_v2_predictions_dir,
    sbert_v2_test_metrics_dir,
)

from Models.SBERT.config_sbert_model import TARGET_LABEL


TARGETS = ["occupation", "gender", "birthyear"]

LABEL_ORDERS = {
    "occupation": ["sports", "performer", "creator", "politics"],
    "gender": ["male", "female"],
    "birthyear": ["1994", "1985", "1975", "1963", "1947"],
}


def resolve_targets(target: str) -> List[str]:
    if target == "all":
        return TARGETS
    return [target]


def ensure_dirs():
    base = Path(sbert_v2_test_metrics_dir)
    plots_dir = base / "plots"
    tables_dir = base / "tables"

    base.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    return base, plots_dir, tables_dir


def load_predictions(target: str) -> pd.DataFrame:
    path = Path(sbert_v2_predictions_dir) / f"{target}_test_predictions.json"

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    df = pd.DataFrame(rows)

    if "true_label" not in df.columns or "pred_label" not in df.columns:
        raise ValueError(
            f"{path} braucht true_label und pred_label. "
            f"Gefundene Spalten: {list(df.columns)}"
        )

    df["true_label"] = df["true_label"].astype(str)
    df["pred_label"] = df["pred_label"].astype(str)

    return df


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels,
    title,
    output_path,
    normalize=False,
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
        cm_df = pd.DataFrame(cm, index=labels, columns=labels).fillna(0.0)
        fmt = ".2f"
    else:
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        fmt = "d"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 16},
    )

    plt.title(title, fontsize=24, pad=14)
    plt.xlabel("Predicted label", fontsize=18)
    plt.ylabel("True label", fontsize=18)
    plt.xticks(rotation=35, ha="right", fontsize=15)
    plt.yticks(rotation=0, fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return cm_df


def evaluate_target(target: str, base_dir: Path, plots_dir: Path, tables_dir: Path):
    labels = LABEL_ORDERS[target]
    df = load_predictions(target)

    df = df[
        df["true_label"].isin(labels)
        & df["pred_label"].isin(labels)
    ].copy()

    if df.empty:
        raise ValueError(f"No valid rows for target: {target}")

    y_true = df["true_label"]
    y_pred = df["pred_label"]

    prefix = f"sbert_v2_{target}"

    # predictions csv
    df.to_csv(tables_dir / f"{prefix}_test_predictions.csv", index=False)

    # classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(tables_dir / f"{prefix}_classification_report.csv")

    # confusion matrices csv
    cm_abs = confusion_matrix(y_true, y_pred, labels=labels)
    cm_abs_df = pd.DataFrame(cm_abs, index=labels, columns=labels)
    cm_abs_df.to_csv(tables_dir / f"{prefix}_confusion_matrix_absolute.csv")

    row_sums = cm_abs.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_abs.astype(float),
        row_sums,
        out=np.zeros_like(cm_abs, dtype=float),
        where=row_sums != 0,
    )
    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels).fillna(0.0)
    cm_norm_df.to_csv(tables_dir / f"{prefix}_confusion_matrix_normalized.csv")

    # plots
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        title=f"SBERT V2 Test {target.capitalize()} Confusion Matrix",
        output_path=plots_dir / f"{prefix}_confusion_matrix.png",
        normalize=False,
    )

    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        title=f"SBERT V2 Test {target.capitalize()} Confusion Matrix (Normalized)",
        output_path=plots_dir / f"{prefix}_confusion_matrix_normalized.png",
        normalize=True,
    )

    metrics = {
        "target_label": target,
        "version": "sbert_v2",
        "classifier_type": "logistic_regression",
        "voting_strategy": (
            df["voting_strategy"].iloc[0]
            if "voting_strategy" in df.columns
            else "unknown"
        ),
        "num_samples": int(len(df)),
        "num_test": int(len(df)),
        "num_labels": int(len(labels)),
        "label_to_id": {label: idx for idx, label in enumerate(labels)},
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "test_weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "classification_report": report,
    }

    metrics_path = base_dir / f"{target}_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(
        f"[OK] {target}: "
        f"acc={metrics['test_accuracy']:.4f} "
        f"macro_f1={metrics['test_macro_f1']:.4f}"
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SBERT V2 test predictions and plot confusion matrices"
    )
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear", "all"],
        default=TARGET_LABEL,
    )

    args = parser.parse_args()

    base_dir, plots_dir, tables_dir = ensure_dirs()

    summary_rows = []

    for target in resolve_targets(args.target):
        metrics = evaluate_target(target, base_dir, plots_dir, tables_dir)

        summary_rows.append({
            "model": "SBERT V2",
            "target": target,
            "num_samples": metrics["num_samples"],
            "accuracy": metrics["test_accuracy"],
            "macro_f1": metrics["test_macro_f1"],
            "weighted_f1": metrics["test_weighted_f1"],
            "classifier_type": metrics["classifier_type"],
            "voting_strategy": metrics["voting_strategy"],
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = base_dir / "sbert_v2_test_summary_metrics.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[OK] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()