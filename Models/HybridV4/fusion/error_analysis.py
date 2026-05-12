import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


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


def ensure_dirs():
    os.makedirs(hybrid_v4_fusion_metrics_dir, exist_ok=True)


def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(rows: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_prediction_file(target: str, predictions_path: Optional[str]) -> str:
    if predictions_path:
        return predictions_path

    candidates = [
        name for name in os.listdir(hybrid_v4_fusion_predictions_dir)
        if name.startswith(f"{target}_test_fair_weighted_fusion_predictions_")
        and name.endswith(".json")
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No fair fusion prediction file found for target={target} in "
            f"{hybrid_v4_fusion_predictions_dir}"
        )

    candidates = sorted(
        candidates,
        key=lambda name: os.path.getmtime(os.path.join(hybrid_v4_fusion_predictions_dir, name)),
        reverse=True,
    )

    selected = os.path.join(hybrid_v4_fusion_predictions_dir, candidates[0])

    print(f"[INFO] Using latest prediction file: {selected}")
    return selected


def prob_dict(row: dict) -> Dict[str, float]:
    labels = row["labels"]
    probs = row["fusion_probabilities"]
    return {label: float(prob) for label, prob in zip(labels, probs)}


def top_k_labels(row: dict, k: int = 3) -> List[str]:
    p = prob_dict(row)
    return [
        label
        for label, _ in sorted(p.items(), key=lambda item: item[1], reverse=True)[:k]
    ]


def confidence(row: dict) -> float:
    return float(max(row["fusion_probabilities"]))


def margin(row: dict) -> float:
    probs = sorted([float(p) for p in row["fusion_probabilities"]], reverse=True)

    if len(probs) < 2:
        return probs[0]

    return probs[0] - probs[1]


def entropy(row: dict) -> float:
    probs = np.asarray(row["fusion_probabilities"], dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def agreement_type(row: dict) -> str:
    pred = row["fusion_pred_label"]

    base_preds = [
        row.get("bertweet_v3_pred_label"),
        row.get("bertweet_v34_pred_label"),
        row.get("sparse_feature_pred_label"),
    ]

    non_null_base_preds = [p for p in base_preds if p is not None]

    if len(set(non_null_base_preds)) == 1:
        only = non_null_base_preds[0]
        if pred == only:
            return f"all_base_agree_on_{only}"
        return "fusion_overrode_all_base"

    if pred == row.get("bertweet_v3_pred_label"):
        return "fusion_follows_v3"

    if pred == row.get("bertweet_v34_pred_label"):
        return "fusion_follows_v34"

    if pred == row.get("sparse_feature_pred_label"):
        return "fusion_follows_sparse"

    return "fusion_other"


def flatten_row(row: dict) -> dict:
    probs = prob_dict(row)
    top_labels = top_k_labels(row, k=min(3, len(probs)))

    out = {
        "celebrity_id": row["celebrity_id"],
        "target": row.get("target"),
        "split": row.get("split"),
        "true_label": row["true_label"],
        "fusion_pred_label": row["fusion_pred_label"],
        "is_correct": row["true_label"] == row["fusion_pred_label"],
        "confidence": confidence(row),
        "margin": margin(row),
        "entropy": entropy(row),
        "agreement_type": agreement_type(row),
        "bertweet_v3_pred_label": row.get("bertweet_v3_pred_label"),
        "bertweet_v34_pred_label": row.get("bertweet_v34_pred_label"),
        "sparse_feature_pred_label": row.get("sparse_feature_pred_label"),
        "top1_label": top_labels[0] if len(top_labels) > 0 else None,
        "top2_label": top_labels[1] if len(top_labels) > 1 else None,
        "top3_label": top_labels[2] if len(top_labels) > 2 else None,
        "creator_signal": row.get("creator_signal"),
        "creator_binary_creator_prob": row.get("creator_binary_creator_prob"),
        "creator_vs_performer_creator_prob": row.get("creator_vs_performer_creator_prob"),
    }

    for label, prob in probs.items():
        out[f"prob_{label}"] = prob

    return out


def analyze_predictions(rows: List[dict], target: str) -> dict:
    labels = rows[0]["labels"]

    y_true = [row["true_label"] for row in rows]
    y_pred = [row["fusion_pred_label"] for row in rows]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    mistake_counter = Counter()
    high_conf_wrong = []
    low_margin_wrong = []
    agreement_counter = Counter()

    per_class_errors = defaultdict(Counter)

    for row in rows:
        true_label = row["true_label"]
        pred_label = row["fusion_pred_label"]

        agreement_counter[agreement_type(row)] += 1

        if true_label != pred_label:
            mistake_counter[(true_label, pred_label)] += 1
            per_class_errors[true_label][pred_label] += 1

            flat = flatten_row(row)
            high_conf_wrong.append(flat)
            low_margin_wrong.append(flat)

    high_conf_wrong.sort(key=lambda r: r["confidence"], reverse=True)
    low_margin_wrong.sort(key=lambda r: r["margin"])

    summary = {
        "target": target,
        "num_rows": len(rows),
        "labels": labels,
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": {
            "labels": labels,
            "counts": cm.tolist(),
            "normalized_by_true_label": (
                cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ).tolist(),
        },
        "mistakes_by_true_pred": [
            {
                "true_label": true_label,
                "pred_label": pred_label,
                "count": count,
            }
            for (true_label, pred_label), count in mistake_counter.most_common()
        ],
        "agreement": dict(agreement_counter.most_common()),
        "per_class_errors": {
            true_label: dict(counter.most_common())
            for true_label, counter in per_class_errors.items()
        },
    }

    return {
        "summary": summary,
        "all_rows_flat": [flatten_row(row) for row in rows],
        "wrong_rows_flat": [flatten_row(row) for row in rows if row["true_label"] != row["fusion_pred_label"]],
        "high_conf_wrong": high_conf_wrong,
        "low_margin_wrong": low_margin_wrong,
    }


def print_summary(summary: dict, top_n: int):
    print("\n========== Error analysis summary ==========")
    print(f"Target:      {summary['target']}")
    print(f"Rows:        {summary['num_rows']}")
    print(f"Accuracy:    {summary['accuracy']:.4f}")
    print(f"Macro F1:    {summary['macro_f1']:.4f}")
    print(f"Weighted F1: {summary['weighted_f1']:.4f}")

    print("\n========== Confusion matrix ==========")
    print(f"labels: {summary['labels']}")
    for row in summary["confusion_matrix"]["counts"]:
        print(row)

    print("\n========== Normalized confusion matrix ==========")
    for row in summary["confusion_matrix"]["normalized_by_true_label"]:
        print(["{:.2f}".format(x) for x in row])

    print("\n========== Top mistakes ==========")
    for item in summary["mistakes_by_true_pred"][:top_n]:
        print(
            f"{item['true_label']:12s} -> {item['pred_label']:12s}: "
            f"{item['count']}"
        )

    print("\n========== Agreement ==========")
    for key, value in summary["agreement"].items():
        print(f"{key:30s}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Error analysis for HybridV4 fair weighted fusion predictions."
    )
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear"],
        required=True,
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional path to prediction JSON. If omitted, latest fair fusion test file for target is used.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of high-confidence wrong examples to save/inspect.",
    )

    args = parser.parse_args()

    ensure_dirs()

    pred_path = get_prediction_file(args.target, args.predictions)
    rows = load_json(pred_path)

    result = analyze_predictions(rows, target=args.target)
    summary = result["summary"]

    base_name = os.path.splitext(os.path.basename(pred_path))[0]

    out_dir = os.path.join(
        hybrid_v4_fusion_metrics_dir,
        "error_analysis",
        args.target,
    )
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, f"{base_name}_error_summary.json")
    all_rows_path = os.path.join(out_dir, f"{base_name}_all_rows.csv")
    wrong_rows_path = os.path.join(out_dir, f"{base_name}_wrong_rows.csv")
    high_conf_path = os.path.join(out_dir, f"{base_name}_high_conf_wrong_top_{args.top_n}.csv")
    low_margin_path = os.path.join(out_dir, f"{base_name}_low_margin_wrong_top_{args.top_n}.csv")

    save_json(summary, summary_path)
    save_csv(result["all_rows_flat"], all_rows_path)
    save_csv(result["wrong_rows_flat"], wrong_rows_path)
    save_csv(result["high_conf_wrong"][:args.top_n], high_conf_path)
    save_csv(result["low_margin_wrong"][:args.top_n], low_margin_path)

    print_summary(summary, top_n=args.top_n)

    print("\n========== Saved files ==========")
    print(f"[OK] Summary:          {summary_path}")
    print(f"[OK] All rows:         {all_rows_path}")
    print(f"[OK] Wrong rows:       {wrong_rows_path}")
    print(f"[OK] High-conf wrong:  {high_conf_path}")
    print(f"[OK] Low-margin wrong: {low_margin_path}")


if __name__ == "__main__":
    main()