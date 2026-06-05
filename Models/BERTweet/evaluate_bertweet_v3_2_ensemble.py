import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABELS_4CLASS = ["sports", "performer", "creator", "politics"]
LABELS_3CLASS = ["sports", "performer", "politics"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_existing_path(candidates: List[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            print(f"[INFO] {label}: {path}")
            return path

    print(f"[ERROR] Could not find {label}. Tried:")
    for path in candidates:
        print(f"  - {path}")
    raise FileNotFoundError(label)


def build_by_id(rows: List[dict]) -> Dict[str, dict]:
    return {str(row["celebrity_id"]): row for row in rows}


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    total = float(np.sum(probs))
    if total <= 0:
        return np.ones_like(probs) / len(probs)
    return probs / total


def ensemble_probs(
    p4: np.ndarray,
    p3: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    p4 order: sports, performer, creator, politics
    p3 order: sports, performer, politics

    V3.2 logic:
    - creator probability remains from the 4-class model
    - remaining probability mass is distributed over sports/performer/politics
      using a soft blend of 4-class and 3-class probabilities
    """

    p4 = normalize_probs(np.asarray(p4, dtype=np.float64))
    p3 = normalize_probs(np.asarray(p3, dtype=np.float64))

    p4_sports = p4[0]
    p4_performer = p4[1]
    p4_creator = p4[2]
    p4_politics = p4[3]

    p4_non_creator = np.array([p4_sports, p4_performer, p4_politics], dtype=np.float64)
    p4_non_creator = normalize_probs(p4_non_creator)

    p3_non_creator = normalize_probs(p3)

    mixed_non_creator = alpha * p4_non_creator + (1.0 - alpha) * p3_non_creator
    mixed_non_creator = normalize_probs(mixed_non_creator)

    non_creator_mass = 1.0 - p4_creator

    final = np.zeros(4, dtype=np.float64)
    final[0] = non_creator_mass * mixed_non_creator[0]  # sports
    final[1] = non_creator_mass * mixed_non_creator[1]  # performer
    final[2] = p4_creator                            # creator
    final[3] = non_creator_mass * mixed_non_creator[2]  # politics

    return normalize_probs(final)


def evaluate_split(
    split_name: str,
    rows_4class: List[dict],
    rows_3class: List[dict],
    alpha: float,
) -> Tuple[dict, List[dict]]:
    by_id_4 = build_by_id(rows_4class)
    by_id_3 = build_by_id(rows_3class)

    y_true = []
    y_pred = []
    output_rows = []

    missing_3class = 0

    for cid, row4 in by_id_4.items():
        true_label = row4.get("true_label")
        p4 = np.asarray(row4["probabilities"], dtype=np.float64)

        if cid in by_id_3:
            p3 = np.asarray(by_id_3[cid]["probabilities"], dtype=np.float64)
            final_probs = ensemble_probs(p4, p3, alpha)
            ensemble_source = "4class_plus_3class"
        else:
            final_probs = normalize_probs(p4)
            missing_3class += 1
            ensemble_source = "4class_fallback"

        pred_label = LABELS_4CLASS[int(np.argmax(final_probs))]

        if true_label is not None:
            y_true.append(true_label)
            y_pred.append(pred_label)

        output_rows.append({
            "celebrity_id": cid,
            "true_label": true_label,
            "pred_label": pred_label,
            "probabilities": final_probs.tolist(),
            "labels": LABELS_4CLASS,
            "alpha": alpha,
            "version": "bertweet_v3_2",
            "split": split_name,
            "ensemble_source": ensemble_source,
            "p4_probabilities": p4.tolist(),
            "p3_probabilities": by_id_3[cid]["probabilities"] if cid in by_id_3 else None,
        })

    metrics = {
        "version": "bertweet_v3_2",
        "split": split_name,
        "alpha": alpha,
        "num_celebrities": len(output_rows),
        "missing_3class_predictions": missing_3class,
        "labels": LABELS_4CLASS,
    }

    if y_true:
        metrics.update({
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            "classification_report": classification_report(
                y_true,
                y_pred,
                labels=LABELS_4CLASS,
                output_dict=True,
                zero_division=0,
            ),
        })

    return metrics, output_rows


def plot_confusion_matrix(y_true, y_pred, labels, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def get_true_pred(rows: List[dict]):
    y_true = []
    y_pred = []

    for row in rows:
        if row.get("true_label") is not None:
            y_true.append(row["true_label"])
            y_pred.append(row["pred_label"])

    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-grid", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--best-metric", type=str, default="macro_f1", choices=["macro_f1", "accuracy", "weighted_f1"])
    args = parser.parse_args()

    alpha_grid = [float(x.strip()) for x in args.alpha_grid.split(",") if x.strip()]

    output_dir = PROJECT_ROOT / "outputs" / "bertweet_v3_2"
    pred_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    plot_dir = output_dir / "plots"

    # ------------------------------------------------------------------
    # Auto-resolve input files
    # ------------------------------------------------------------------

    val_4class_path = resolve_existing_path([
        PROJECT_ROOT / "outputs" / "bertweet_v3_1" / "predictions" / "occupation_val_all_predictions.json",
        PROJECT_ROOT / "outputs" / "bertweet_v3" / "predictions" / "occupation_val_all_predictions.json",
        PROJECT_ROOT / "outputs" / "bertweet_v3" / "predictions" / "occupation_val_predictions.json",
    ], "val 4class predictions")

    val_3class_path = resolve_existing_path([
        PROJECT_ROOT / "outputs" / "bertweet_v3_1" / "predictions" / "occupation_3class_val_predictions.json",
        PROJECT_ROOT / "outputs" / "bertweet_v3" / "predictions" / "occupation_3class_val_predictions.json",
    ], "val 3class predictions")

    test_4class_path = resolve_existing_path([
        PROJECT_ROOT / "outputs" / "bertweet_v3_1" / "predictions" / "occupation_test_predictions.json",
        PROJECT_ROOT / "outputs" / "bertweet_v3" / "predictions" / "occupation_test_predictions.json",
    ], "test 4class predictions")

    test_3class_path = resolve_existing_path([
        PROJECT_ROOT / "outputs" / "bertweet_v3_1" / "predictions" / "occupation_3class_test_predictions.json",
        PROJECT_ROOT / "outputs" / "bertweet_v3_1" / "predictions" / "occupation_3class_test_all_predictions.json",
        PROJECT_ROOT / "outputs" / "bertweet_v3_1" / "predictions" / "occupation_test_3class_predictions.json",

        PROJECT_ROOT / "outputs" / "bertweet_v3" / "predictions" / "occupation_3class_test_predictions.json",
        PROJECT_ROOT / "outputs" / "bertweet_v3" / "predictions" / "occupation_3class_test_all_predictions.json",
        PROJECT_ROOT / "outputs" / "bertweet_v3" / "predictions" / "occupation_test_3class_predictions.json",
    ], "test 3class predictions")

    rows_val_4 = load_json(val_4class_path)
    rows_val_3 = load_json(val_3class_path)
    rows_test_4 = load_json(test_4class_path)
    rows_test_3 = load_json(test_3class_path)

    print(f"[INFO] Val 4class rows:  {len(rows_val_4)}")
    print(f"[INFO] Val 3class rows:  {len(rows_val_3)}")
    print(f"[INFO] Test 4class rows: {len(rows_test_4)}")
    print(f"[INFO] Test 3class rows: {len(rows_test_3)}")

    # ------------------------------------------------------------------
    # Grid search alpha on validation
    # ------------------------------------------------------------------

    alpha_results = []

    for alpha in alpha_grid:
        val_metrics, _ = evaluate_split(
            split_name="val",
            rows_4class=rows_val_4,
            rows_3class=rows_val_3,
            alpha=alpha,
        )

        alpha_results.append({
            "alpha": alpha,
            "accuracy": val_metrics.get("accuracy"),
            "macro_f1": val_metrics.get("macro_f1"),
            "weighted_f1": val_metrics.get("weighted_f1"),
            "missing_3class_predictions": val_metrics.get("missing_3class_predictions"),
        })

        print(
            f"[VAL] alpha={alpha:.2f} "
            f"acc={val_metrics.get('accuracy'):.4f} "
            f"macro_f1={val_metrics.get('macro_f1'):.4f}"
        )

    best_result = max(alpha_results, key=lambda x: x[args.best_metric])
    best_alpha = best_result["alpha"]

    print()
    print(f"[BEST] metric={args.best_metric}")
    print(f"[BEST] alpha={best_alpha:.2f}")
    print(f"[BEST] val_acc={best_result['accuracy']:.4f}")
    print(f"[BEST] val_macro_f1={best_result['macro_f1']:.4f}")

    save_json(alpha_results, metrics_dir / "occupation_alpha_search.json")

    # ------------------------------------------------------------------
    # Save final val and test predictions with best alpha
    # ------------------------------------------------------------------

    final_val_metrics, final_val_preds = evaluate_split(
        split_name="val",
        rows_4class=rows_val_4,
        rows_3class=rows_val_3,
        alpha=best_alpha,
    )

    final_test_metrics, final_test_preds = evaluate_split(
        split_name="test",
        rows_4class=rows_test_4,
        rows_3class=rows_test_3,
        alpha=best_alpha,
    )

    final_val_metrics["selected_by"] = args.best_metric
    final_test_metrics["selected_by"] = args.best_metric
    final_test_metrics["selected_alpha_from_val"] = best_alpha

    save_json(final_val_metrics, metrics_dir / "occupation_val_metrics.json")
    save_json(final_test_metrics, metrics_dir / "occupation_test_metrics.json")

    save_json(final_val_preds, pred_dir / "occupation_val_predictions.json")
    save_json(final_test_preds, pred_dir / "occupation_test_predictions.json")

    # ------------------------------------------------------------------
    # Confusion matrices
    # ------------------------------------------------------------------

    y_val_true, y_val_pred = get_true_pred(final_val_preds)
    y_test_true, y_test_pred = get_true_pred(final_test_preds)

    if y_val_true:
        plot_confusion_matrix(
            y_val_true,
            y_val_pred,
            LABELS_4CLASS,
            plot_dir / "occupation_val_confusion_matrix.png",
            f"BERTweet V3.2 Occupation Val Confusion Matrix (alpha={best_alpha:.2f})",
        )

    if y_test_true:
        plot_confusion_matrix(
            y_test_true,
            y_test_pred,
            LABELS_4CLASS,
            plot_dir / "occupation_test_confusion_matrix.png",
            f"BERTweet V3.2 Occupation Test Confusion Matrix (alpha={best_alpha:.2f})",
        )

    print()
    print("[OK] Saved alpha search:")
    print(metrics_dir / "occupation_alpha_search.json")
    print("[OK] Saved val metrics:")
    print(metrics_dir / "occupation_val_metrics.json")
    print("[OK] Saved test metrics:")
    print(metrics_dir / "occupation_test_metrics.json")
    print("[OK] Saved predictions:")
    print(pred_dir / "occupation_val_predictions.json")
    print(pred_dir / "occupation_test_predictions.json")
    print("[OK] Saved plots:")
    print(plot_dir)


if __name__ == "__main__":
    main()