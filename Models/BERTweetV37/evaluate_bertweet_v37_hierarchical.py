import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Models.BERTweetV37.config_bertweet_v37 import MODEL_NAME


# -------------------------------------------------------------------
# V3.7 output dirs
# -------------------------------------------------------------------
BERTWEET_V37_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "bertweet_v37")
BERTWEET_V37_PREDICTIONS_DIR = os.path.join(BERTWEET_V37_OUTPUT_DIR, "predictions")
BERTWEET_V37_METRICS_DIR = os.path.join(BERTWEET_V37_OUTPUT_DIR, "metrics")

OCCUPATION_LABELS = ["sports", "performer", "creator", "politics"]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_dirs():
    os.makedirs(BERTWEET_V37_PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(BERTWEET_V37_METRICS_DIR, exist_ok=True)


def load_json(path: str):
    print(f"[INFO] Loading: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def by_celebrity(predictions: List[dict]) -> Dict[str, dict]:
    out = {}
    for item in predictions:
        cid = str(item["celebrity_id"])
        if cid in out:
            raise ValueError(f"Duplicate celebrity_id in predictions: {cid}")
        out[cid] = item
    return out


def get_prob(item: dict, label: str) -> float:
    if "probability_by_label" in item:
        return float(item["probability_by_label"][label])

    labels = item.get("labels")
    probs = item.get("probabilities")
    if labels is None or probs is None:
        raise ValueError("Prediction item must contain probability_by_label or labels+probabilities.")
    return float(probs[labels.index(label)])


def normalize_prob_dict(prob_dict: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(prob_dict.values()))
    if total <= 0:
        n = len(prob_dict)
        return {k: 1.0 / n for k in prob_dict}
    return {k: float(v) / total for k, v in prob_dict.items()}


def argmax_label(prob_dict: Dict[str, float]) -> str:
    return max(prob_dict.items(), key=lambda kv: kv[1])[0]


def combine_predictions(group3_item: dict, cp_item: dict, mode: str) -> dict:
    true_occupation = str(group3_item["true_occupation"])

    g3_probs = {
        "sports": get_prob(group3_item, "sports"),
        "entertainment": get_prob(group3_item, "entertainment"),
        "politics": get_prob(group3_item, "politics"),
    }

    cp_probs = {
        "creator": get_prob(cp_item, "creator"),
        "performer": get_prob(cp_item, "performer"),
    }

    group3_pred = str(group3_item["pred_label"])
    cp_pred = str(cp_item["pred_label"])

    if mode == "hard":
        if group3_pred == "sports":
            final_probs = {
                "sports": 1.0,
                "performer": 0.0,
                "creator": 0.0,
                "politics": 0.0,
            }
            final_pred = "sports"
        elif group3_pred == "politics":
            final_probs = {
                "sports": 0.0,
                "performer": 0.0,
                "creator": 0.0,
                "politics": 1.0,
            }
            final_pred = "politics"
        else:
            final_probs = {
                "sports": 0.0,
                "politics": 0.0,
                "creator": 1.0 if cp_pred == "creator" else 0.0,
                "performer": 1.0 if cp_pred == "performer" else 0.0,
            }
            final_pred = cp_pred

    elif mode == "soft":
        final_probs = {
            "sports": g3_probs["sports"],
            "politics": g3_probs["politics"],
            "creator": g3_probs["entertainment"] * cp_probs["creator"],
            "performer": g3_probs["entertainment"] * cp_probs["performer"],
        }
        final_probs = normalize_prob_dict(final_probs)
        final_pred = argmax_label(final_probs)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    sorted_probs = sorted(final_probs.items(), key=lambda kv: kv[1], reverse=True)
    top1_label, top1_prob = sorted_probs[0]
    top2_label, top2_prob = sorted_probs[1]

    return {
        "celebrity_id": str(group3_item["celebrity_id"]),
        "true_label": true_occupation,
        "pred_label": final_pred,
        "correct": final_pred == true_occupation,
        "probability_by_label": {label: float(final_probs[label]) for label in OCCUPATION_LABELS},
        "probabilities": [float(final_probs[label]) for label in OCCUPATION_LABELS],
        "labels": OCCUPATION_LABELS,
        "top1_label": top1_label,
        "top1_prob": float(top1_prob),
        "top2_label": top2_label,
        "top2_prob": float(top2_prob),
        "margin": float(top1_prob - top2_prob),
        "group3_pred": group3_pred,
        "group3_probabilities": g3_probs,
        "creator_performer_pred": cp_pred,
        "creator_performer_probabilities": cp_probs,
        "mode": mode,
        "version": "bertweet_v37",
        "model_name": MODEL_NAME,
    }


def evaluate_predictions(predictions: List[dict], mode: str) -> dict:
    y_true = [item["true_label"] for item in predictions]
    y_pred = [item["pred_label"] for item in predictions]

    cm = confusion_matrix(y_true, y_pred, labels=OCCUPATION_LABELS)
    report = classification_report(
        y_true,
        y_pred,
        labels=OCCUPATION_LABELS,
        output_dict=True,
        zero_division=0,
    )

    mistake_pairs = Counter(
        f"{t}->{p}" for t, p in zip(y_true, y_pred) if t != p
    )

    return {
        "target_label": "occupation",
        "version": "bertweet_v37",
        "mode": mode,
        "model_name": MODEL_NAME,
        "num_celebrities": int(len(predictions)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=OCCUPATION_LABELS, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": {
            "labels": OCCUPATION_LABELS,
            "matrix": cm.astype(int).tolist(),
        },
        "prediction_distribution": dict(Counter(y_pred)),
        "mistake_pairs": dict(mistake_pairs.most_common()),
    }


def evaluate_direct_file(path: str) -> dict:
    direct_predictions = load_json(path)
    y_true = [str(item.get("true_label", item.get("true_occupation"))) for item in direct_predictions]
    y_pred = [str(item["pred_label"]) for item in direct_predictions]

    cm = confusion_matrix(y_true, y_pred, labels=OCCUPATION_LABELS)
    return {
        "path": path,
        "num_celebrities": int(len(direct_predictions)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=OCCUPATION_LABELS, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=OCCUPATION_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": OCCUPATION_LABELS,
            "matrix": cm.astype(int).tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BERTweet V3.7 hierarchical occupation predictions")
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument(
        "--direct-predictions",
        default=None,
        help="Optional direct 4-class occupation predictions JSON for comparison.",
    )
    args = parser.parse_args()

    ensure_dirs()

    group3_path = os.path.join(
        BERTWEET_V37_PREDICTIONS_DIR,
        f"occupation_group3_{args.split}_predictions.json",
    )
    cp_path = os.path.join(
        BERTWEET_V37_PREDICTIONS_DIR,
        f"creator_performer_{args.split}_predictions.json",
    )

    group3 = by_celebrity(load_json(group3_path))
    cp = by_celebrity(load_json(cp_path))

    missing_cp = sorted(set(group3.keys()) - set(cp.keys()))
    missing_group3 = sorted(set(cp.keys()) - set(group3.keys()))
    if missing_cp or missing_group3:
        raise ValueError(
            f"Prediction celebrity IDs do not match. "
            f"missing_cp={len(missing_cp)}, missing_group3={len(missing_group3)}"
        )

    final_report = {
        "version": "bertweet_v37",
        "target_label": "occupation",
        "split": args.split,
        "model_name": MODEL_NAME,
        "input_files": {
            "occupation_group3": group3_path,
            "creator_performer": cp_path,
        },
        "results": {},
    }

    for mode in ["hard", "soft"]:
        combined = [
            combine_predictions(group3[cid], cp[cid], mode=mode)
            for cid in sorted(group3.keys())
        ]
        metrics = evaluate_predictions(combined, mode=mode)

        predictions_path = os.path.join(
            BERTWEET_V37_PREDICTIONS_DIR,
            f"occupation_v37_{mode}_hierarchical_{args.split}_predictions.json",
        )
        metrics_path = os.path.join(
            BERTWEET_V37_METRICS_DIR,
            f"occupation_v37_{mode}_hierarchical_{args.split}_metrics.json",
        )

        save_json(combined, predictions_path)
        save_json(metrics, metrics_path)

        final_report["results"][mode] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "predictions_path": predictions_path,
            "metrics_path": metrics_path,
            "per_class_f1": {
                label: metrics["classification_report"][label]["f1-score"]
                for label in OCCUPATION_LABELS
            },
            "mistake_pairs": metrics["mistake_pairs"],
        }

        print(
            f"[RESULT] v37_{mode}_hierarchical "
            f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
        )
        print(f"[OK] Saved predictions: {predictions_path}")
        print(f"[OK] Saved metrics:     {metrics_path}")

    if args.direct_predictions:
        direct = evaluate_direct_file(args.direct_predictions)
        final_report["direct_comparison"] = direct
        print(
            f"[RESULT] direct_comparison "
            f"acc={direct['accuracy']:.4f} macro_f1={direct['macro_f1']:.4f}"
        )

    final_report_path = os.path.join(
        BERTWEET_V37_METRICS_DIR,
        f"occupation_v37_hierarchical_{args.split}_final_report.json",
    )
    save_json(final_report, final_report_path)
    print(f"[OK] Saved final report: {final_report_path}")


if __name__ == "__main__":
    main()
