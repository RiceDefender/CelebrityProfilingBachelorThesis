import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import _constants as C


# --------------------------------------------------
# Config
# --------------------------------------------------
AGE_BUCKET_LABELS = [1994, 1985, 1975, 1963, 1947]


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_data() -> pd.DataFrame:
    path = os.path.join(C.comparison_tables_dir, "merged_predictions.csv")
    df = pd.read_csv(path)

    df["true_birthyear"] = pd.to_numeric(df["true_birthyear"], errors="coerce")
    df["pred_birthyear"] = pd.to_numeric(df["pred_birthyear"], errors="coerce")
    return df


def ensure_output_dirs() -> None:
    os.makedirs(C.comparison_plots_dir, exist_ok=True)
    os.makedirs(C.comparison_tables_dir, exist_ok=True)


def nearest_bucket(year: int, bucket_labels: List[int]) -> int:
    """
    Map a year to the nearest representative bucket center.
    Example: 1987 -> 1985, 1972 -> 1975, etc.
    """
    return min(bucket_labels, key=lambda b: abs(year - b))


def add_age_bucket_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["true_birthyear_bucket"] = out["true_birthyear"].apply(
        lambda y: nearest_bucket(int(y), AGE_BUCKET_LABELS)
    )
    out["pred_birthyear_bucket"] = out["pred_birthyear"].apply(
        lambda y: nearest_bucket(int(y), AGE_BUCKET_LABELS)
    )
    return out


def normalize_cm(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, where=row_sums != 0)


# --------------------------------------------------
# Plotting
# --------------------------------------------------
def plot_age_bucket_confusion_matrix(df: pd.DataFrame) -> None:
    cm = confusion_matrix(
        df["true_birthyear_bucket"],
        df["pred_birthyear_bucket"],
        labels=AGE_BUCKET_LABELS,
    )
    cm_norm = normalize_cm(cm)

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=AGE_BUCKET_LABELS,
        yticklabels=AGE_BUCKET_LABELS,
        cbar=False,
    )
    plt.title("Birthyear Confusion Matrix (Normalized, Paper Style)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(
        os.path.join(C.comparison_plots_dir, "birthyear_confusion_matrix_normalized.png"),
        dpi=200,
    )
    plt.close()


def plot_age_bucket_distribution(df: pd.DataFrame) -> None:
    true_counts = (
        df["true_birthyear_bucket"]
        .value_counts()
        .reindex(AGE_BUCKET_LABELS, fill_value=0)
    )
    pred_counts = (
        df["pred_birthyear_bucket"]
        .value_counts()
        .reindex(AGE_BUCKET_LABELS, fill_value=0)
    )

    plot_df = pd.DataFrame({
        "True": true_counts,
        "Predicted": pred_counts,
    })

    ax = plot_df.plot(kind="bar", figsize=(8, 4.5))
    ax.set_title("True vs Predicted Birthyear Bucket Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(C.comparison_plots_dir, "birthyear_bucket_distribution_true_vs_pred.png"),
        dpi=200,
    )
    plt.close()


# --------------------------------------------------
# Tables
# --------------------------------------------------
def save_age_bucket_tables(df: pd.DataFrame) -> None:
    cm = confusion_matrix(
        df["true_birthyear_bucket"],
        df["pred_birthyear_bucket"],
        labels=AGE_BUCKET_LABELS,
    )
    cm_norm = normalize_cm(cm)

    cm_abs_df = pd.DataFrame(cm, index=AGE_BUCKET_LABELS, columns=AGE_BUCKET_LABELS)
    cm_norm_df = pd.DataFrame(cm_norm, index=AGE_BUCKET_LABELS, columns=AGE_BUCKET_LABELS)

    cm_abs_df.to_csv(
        os.path.join(C.comparison_tables_dir, "birthyear_bucket_confusion_matrix_absolute.csv")
    )
    cm_norm_df.to_csv(
        os.path.join(C.comparison_tables_dir, "birthyear_bucket_confusion_matrix_normalized.csv")
    )

    bucket_summary = pd.DataFrame({
        "true_count": df["true_birthyear_bucket"].value_counts().reindex(AGE_BUCKET_LABELS, fill_value=0),
        "pred_count": df["pred_birthyear_bucket"].value_counts().reindex(AGE_BUCKET_LABELS, fill_value=0),
    })
    bucket_summary.index.name = "birthyear_bucket"
    bucket_summary.to_csv(
        os.path.join(C.comparison_tables_dir, "birthyear_bucket_distribution.csv")
    )


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:
    ensure_output_dirs()

    df = load_data()
    df = add_age_bucket_columns(df)

    # save enriched merged file too
    df.to_csv(
        os.path.join(C.comparison_tables_dir, "merged_predictions_with_age_buckets.csv"),
        index=False,
    )

    save_age_bucket_tables(df)
    plot_age_bucket_confusion_matrix(df)
    plot_age_bucket_distribution(df)

    print("[DONE] Age paper-style plots and tables created.")


if __name__ == "__main__":
    main()