import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FILES = {
    "val": PROJECT_ROOT / "outputs" / "bertweet_v3_2" / "predictions" / "occupation_val_predictions.json",
    "test": PROJECT_ROOT / "outputs" / "bertweet_v3_2" / "predictions" / "occupation_test_predictions.json",
}

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bertweet_v3_2" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["sports", "performer", "creator", "politics"]


def load_predictions(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    y_true = [row["true_label"] for row in rows]
    y_pred = [row["pred_label"] for row in rows]

    return y_true, y_pred


def plot_normalized_confusion(split: str, y_true, y_pred):
    cm_raw = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_norm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm_norm)

    ax.set_title(f"BERTweet V3.2 Occupation {split.upper()} - Normalized Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=35, ha="right")
    ax.set_yticklabels(LABELS)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}\n({cm_raw[i, j]})",
                ha="center",
                va="center",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized share per true class")

    fig.tight_layout()

    out_path = OUTPUT_DIR / f"occupation_{split}_confusion_matrix_normalized.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved normalized confusion matrix: {out_path}")


def main():
    for split, path in PREDICTION_FILES.items():
        print(f"[INFO] Loading {split}: {path}")
        y_true, y_pred = load_predictions(path)
        plot_normalized_confusion(split, y_true, y_pred)


if __name__ == "__main__":
    main()