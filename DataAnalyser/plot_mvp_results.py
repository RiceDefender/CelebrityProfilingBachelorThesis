import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from _constants import (
    bert_output_dir,
    sbert_output_dir,
    comparison_plots_dir,
    comparison_tables_dir,
)

import numpy as np


TARGETS = ["occupation", "gender", "birthyear"]

LABEL_ORDERS = {
    "occupation": ["sports", "performer", "creator", "politics"],
    "gender": ["male", "female"],
    "birthyear": ["1994", "1985", "1975", "1963", "1947"],
}

MODELS = {
    "bert_mvp": {
        "display_name": "BERT MVP",
        "output_dir": bert_output_dir,
    },
    "sbert_mvp": {
        "display_name": "SBERT MVP",
        "output_dir": sbert_output_dir,
    },
}


def ensure_dirs():
    os.makedirs(comparison_plots_dir, exist_ok=True)
    os.makedirs(comparison_tables_dir, exist_ok=True)


def find_prediction_file(model_output_dir: str, target: str) -> Path:
    """
    Supports both structures:

    BERT:
    outputs/bert_mvp/occupation/predictions/occupation_test_celeb_predictions.json

    SBERT:
    outputs/sbert_mvp/predictions/occupation_val_predictions.json
    """

    base = Path(model_output_dir)

    candidates = [
        base / target / "predictions" / f"{target}_test_celeb_predictions.json",
        base / target / "predictions" / f"{target}_val_celeb_predictions.json",
        base / target / "predictions" / f"{target}_test_predictions.json",
        base / target / "predictions" / f"{target}_val_predictions.json",
        base / "predictions" / f"{target}_test_celeb_predictions.json",
        base / "predictions" / f"{target}_val_celeb_predictions.json",
        base / "predictions" / f"{target}_test_predictions.json",
        base / "predictions" / f"{target}_val_predictions.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"No prediction file found for {base} / {target}")


def load_predictions(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    df = pd.DataFrame(rows)

    true_candidates = [
        "true_label",
        "label",
        "gold_label",
        "y_true",
        "true",
        "target",
    ]

    pred_candidates = [
        "pred_label",
        "prediction",
        "pred",
        "y_pred",
        "predicted_label",
        "predicted",
    ]

    true_col = next((col for col in true_candidates if col in df.columns), None)
    pred_col = next((col for col in pred_candidates if col in df.columns), None)

    if true_col is None or pred_col is None:
        raise ValueError(
            f"{path} has unsupported prediction format. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.rename(columns={
        true_col: "true_label",
        pred_col: "pred_label",
    })

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
        cm = cm.astype(float) / row_sums
        cm = pd.DataFrame(cm, index=labels, columns=labels).fillna(0.0)
        fmt = ".2f"
    else:
        cm = pd.DataFrame(cm, index=labels, columns=labels)
        fmt = "d"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
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

    return cm


def save_reports(model_key, model_name, target, df, labels):
    y_true = df["true_label"]
    y_pred = df["pred_label"]

    prefix = f"{model_key}_{target}"

    # Predictions CSV
    pred_csv_path = Path(comparison_tables_dir) / f"{prefix}_predictions.csv"
    df.to_csv(pred_csv_path, index=False)

    # Classification report CSV
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_csv_path = Path(comparison_tables_dir) / f"{prefix}_classification_report.csv"
    report_df.to_csv(report_csv_path)

    # Confusion matrices CSV
    cm_abs = confusion_matrix(y_true, y_pred, labels=labels)
    cm_abs_df = pd.DataFrame(cm_abs, index=labels, columns=labels)

    cm_abs_csv_path = Path(comparison_tables_dir) / f"{prefix}_confusion_matrix_absolute.csv"
    cm_abs_df.to_csv(cm_abs_csv_path)

    row_sums = cm_abs.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_abs.astype(float),
        row_sums,
        out=np.zeros_like(cm_abs, dtype=float),
        where=row_sums != 0,
    )
    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels).fillna(0.0)

    cm_norm_csv_path = Path(comparison_tables_dir) / f"{prefix}_confusion_matrix_normalized.csv"
    cm_norm_df.to_csv(cm_norm_csv_path)

    # Summary row
    summary = {
        "model": model_name,
        "model_key": model_key,
        "target": target,
        "num_samples": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "prediction_file_rows": len(df),
    }

    return summary


def main():
    ensure_dirs()

    summary_rows = []

    for model_key, config in MODELS.items():
        model_name = config["display_name"]
        model_output_dir = config["output_dir"]

        for target in TARGETS:
            labels = LABEL_ORDERS[target]

            try:
                prediction_file = find_prediction_file(model_output_dir, target)
            except FileNotFoundError as e:
                print(f"[SKIP] {e}")
                continue

            print(f"[INFO] Loading {model_key}/{target}: {prediction_file}")

            df = load_predictions(prediction_file)

            # keep only known labels for clean paper-style plots
            df = df[
                df["true_label"].isin(labels)
                & df["pred_label"].isin(labels)
            ].copy()

            if df.empty:
                print(f"[SKIP] No valid rows for {model_key}/{target}")
                continue

            prefix = f"{model_key}_{target}"

            # CSV outputs
            summary = save_reports(model_key, model_name, target, df, labels)
            summary_rows.append(summary)

            # Plot absolute confusion matrix
            plot_confusion_matrix(
                y_true=df["true_label"],
                y_pred=df["pred_label"],
                labels=labels,
                title=f"{model_name} {target.capitalize()} Confusion Matrix",
                output_path=Path(comparison_plots_dir) / f"{prefix}_confusion_matrix.png",
                normalize=False,
            )

            # Plot normalized confusion matrix
            plot_confusion_matrix(
                y_true=df["true_label"],
                y_pred=df["pred_label"],
                labels=labels,
                title=f"{model_name} {target.capitalize()} Confusion Matrix (Normalized)",
                output_path=Path(comparison_plots_dir) / f"{prefix}_confusion_matrix_normalized.png",
                normalize=True,
            )

            print(f"[OK] Created plots and tables for {model_key}/{target}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = Path(comparison_tables_dir) / "mvp_summary_metrics.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[OK] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()