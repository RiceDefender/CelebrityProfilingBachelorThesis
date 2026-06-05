import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import _constants as C


# -----------------------------
# CONFIG (Paper style)
# -----------------------------
GENDER_LABELS = ["male", "female"]
OCCUPATION_LABELS = ["creator", "performer", "politics", "sports"]


# -----------------------------
# Load merged data
# -----------------------------
def load_data():
    path = os.path.join(C.comparison_tables_dir, "merged_predictions.csv")
    return pd.read_csv(path)


# -----------------------------
# Normalize confusion matrix
# -----------------------------
def normalize_cm(cm):
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, where=row_sums != 0)


# -----------------------------
# Plot heatmap (paper style)
# -----------------------------
def plot_heatmap(cm, labels, title, filename):
    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False
    )

    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    plt.savefig(os.path.join(C.comparison_plots_dir, filename), dpi=200)
    plt.close()


# -----------------------------
# Gender matrix
# -----------------------------
def plot_gender(df):
    cm = confusion_matrix(
        df["true_gender"],
        df["pred_gender"],
        labels=GENDER_LABELS
    )

    cm_norm = normalize_cm(cm)

    plot_heatmap(
        cm_norm,
        GENDER_LABELS,
        "Gender Confusion Matrix (Normalized)",
        "gender_confusion_matrix_normalized.png"
    )


# -----------------------------
# Occupation matrix
# -----------------------------
def plot_occupation(df):
    cm = confusion_matrix(
        df["true_occupation"],
        df["pred_occupation"],
        labels=OCCUPATION_LABELS
    )

    cm_norm = normalize_cm(cm)

    plot_heatmap(
        cm_norm,
        OCCUPATION_LABELS,
        "Occupation Confusion Matrix (Normalized)",
        "occupation_confusion_matrix_normalized.png"
    )


# -----------------------------
# Main
# -----------------------------
def main():
    df = load_data()

    plot_gender(df)
    plot_occupation(df)

    print("[DONE] Paper-style plots created.")


if __name__ == "__main__":
    main()