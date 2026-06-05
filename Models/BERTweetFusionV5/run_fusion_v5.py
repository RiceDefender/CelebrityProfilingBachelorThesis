import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Models.BERTweetFusionV5.config_fusion_v5 import (  # noqa: E402
    ANALYSIS_DIR,
    BASE_MODEL_WEIGHTS,
    GATING_THRESHOLDS,
    LABELS,
    METRICS_DIR,
    OUTPUT_DIR,
    PREDICTION_PATHS,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    SAVE_DECISION_TRACE,
)


# -------------------------------------------------------------------
# IO
# -------------------------------------------------------------------

def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, PREDICTIONS_DIR, METRICS_DIR, ANALYSIS_DIR, REPORTS_DIR]:
        os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_existing_path(model_key: str) -> Optional[str]:
    for path in PREDICTION_PATHS.get(model_key, []):
        if os.path.exists(path):
            return path
    return None


def load_prediction_file(model_key: str) -> Optional[Dict[str, dict]]:
    path = resolve_existing_path(model_key)
    if path is None:
        print(f"[WARN] Missing predictions for {model_key}")
        return None

    rows = load_json(path)
    by_id = {str(row["celebrity_id"]): row for row in rows}
    print(f"[OK] Loaded {model_key}: {path} ({len(by_id)} celebrities)")
    return by_id


# -------------------------------------------------------------------
# Probability helpers
# -------------------------------------------------------------------

def normalize_probs(probs: Dict[str, float], labels: List[str]) -> Dict[str, float]:
    values = np.array([float(probs.get(label, 0.0)) for label in labels], dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        values = np.ones(len(labels), dtype=np.float64) / len(labels)
    else:
        values = values / total
    return {label: float(value) for label, value in zip(labels, values)}


def get_prob_dict(row: dict, labels: List[str]) -> Dict[str, float]:
    if "probability_by_label" in row and isinstance(row["probability_by_label"], dict):
        return normalize_probs(row["probability_by_label"], labels)

    if "probabilities" in row:
        row_labels = row.get("labels", labels)
        raw = {label: float(prob) for label, prob in zip(row_labels, row["probabilities"])}
        return normalize_probs(raw, labels)

    pred = row.get("pred_label")
    raw = {label: 0.0 for label in labels}
    if pred in raw:
        raw[pred] = 1.0
    return raw


def argmax_label(prob_dict: Dict[str, float]) -> str:
    return max(prob_dict.items(), key=lambda x: x[1])[0]


def top2(prob_dict: Dict[str, float]) -> Tuple[str, float, str, float, float]:
    items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    top1_label, top1_prob = items[0]
    top2_label, top2_prob = items[1]
    return top1_label, float(top1_prob), top2_label, float(top2_prob), float(top1_prob - top2_prob)


def weighted_mean_probs(model_probs: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Dict[str, float]:
    active = [(key, weights[key]) for key in weights if key in model_probs]
    if not active:
        raise ValueError("No base 4-class model predictions found. Cannot run fusion.")

    total_weight = sum(w for _, w in active)
    out = {label: 0.0 for label in LABELS}
    for key, weight in active:
        normalized_weight = weight / total_weight
        for label in LABELS:
            out[label] += normalized_weight * model_probs[key].get(label, 0.0)

    return normalize_probs(out, LABELS)


def label_votes(model_probs: Dict[str, Dict[str, float]], label: str) -> int:
    return sum(1 for probs in model_probs.values() if argmax_label(probs) == label)


# -------------------------------------------------------------------
# Model-specific aux extraction
# -------------------------------------------------------------------

def get_creator_binary_prob(row: Optional[dict]) -> Optional[float]:
    if row is None:
        return None
    probs = get_prob_dict(row, ["not_creator", "creator"])
    return probs.get("creator", None)


def get_3class_pred(row: Optional[dict]) -> Optional[str]:
    if row is None:
        return None
    probs = get_prob_dict(row, ["sports", "performer", "politics"])
    return argmax_label(probs)


def get_group3_probs(row: Optional[dict]) -> Optional[Dict[str, float]]:
    if row is None:
        return None
    return get_prob_dict(row, ["sports", "entertainment", "politics"])


def get_cp_probs(row: Optional[dict]) -> Optional[Dict[str, float]]:
    if row is None:
        return None
    return get_prob_dict(row, ["creator", "performer"])


# -------------------------------------------------------------------
# Gating D
# -------------------------------------------------------------------

def creator_rescue_condition(
    base_probs: Dict[str, float],
    base_model_probs: Dict[str, Dict[str, float]],
    creator_binary_p: Optional[float],
    group3_probs: Optional[Dict[str, float]],
    cp_probs: Optional[Dict[str, float]],
) -> Tuple[bool, Dict[str, object]]:
    t = GATING_THRESHOLDS
    votes = []

    if creator_binary_p is not None and creator_binary_p >= t["creator_binary_min"]:
        votes.append("v3_creator_binary")

    if base_probs.get("creator", 0.0) >= t["base_creator_min"]:
        votes.append("base_creator_prob")

    max_4class_creator = max((p.get("creator", 0.0) for p in base_model_probs.values()), default=0.0)
    if max_4class_creator >= t["any_4class_creator_min"]:
        votes.append("any_4class_creator_prob")

    if group3_probs is not None and cp_probs is not None:
        if (
            group3_probs.get("entertainment", 0.0) >= t["v37_entertainment_min_for_creator_signal"]
            and cp_probs.get("creator", 0.0) >= t["v37_cp_creator_min_for_creator_signal"]
        ):
            votes.append("v37_entertainment_creator_signal")

    ok = len(votes) >= t["creator_rescue_votes_min"]
    return ok, {
        "votes": votes,
        "num_votes": len(votes),
        "creator_binary_p": creator_binary_p,
        "base_creator_p": base_probs.get("creator", 0.0),
        "max_4class_creator_p": max_4class_creator,
    }


def rescue_from_performer_condition(
    label: str,
    base_probs: Dict[str, float],
    base_model_probs: Dict[str, Dict[str, float]],
    group3_probs: Optional[Dict[str, float]],
    occupation_3class_pred: Optional[str],
) -> Tuple[bool, Dict[str, object]]:
    t = GATING_THRESHOLDS
    votes = []

    if label == "sports":
        group_min = t["sports_group3_min"]
        base_min = t["sports_base_min"]
    elif label == "politics":
        group_min = t["politics_group3_min"]
        base_min = t["politics_base_min"]
    else:
        raise ValueError(f"Unsupported rescue label: {label}")

    if group3_probs is not None and group3_probs.get(label, 0.0) >= group_min:
        votes.append(f"v37_group3_{label}")

    if base_probs.get(label, 0.0) >= base_min:
        votes.append(f"base_{label}_prob")

    if occupation_3class_pred == label:
        votes.append(f"v3_3class_{label}")

    n_model_votes = label_votes(base_model_probs, label)
    if n_model_votes >= 2:
        votes.append(f"two_4class_models_{label}")

    ok = len(votes) >= t["rescue_votes_min"]
    return ok, {
        "votes": votes,
        "num_votes": len(votes),
        "n_4class_model_votes": n_model_votes,
        "base_label_p": base_probs.get(label, 0.0),
        "group3_label_p": None if group3_probs is None else group3_probs.get(label, 0.0),
        "occupation_3class_pred": occupation_3class_pred,
    }


def v37_creator_specialist_condition(
    group3_probs: Optional[Dict[str, float]],
    cp_probs: Optional[Dict[str, float]],
) -> Tuple[bool, Dict[str, object]]:
    t = GATING_THRESHOLDS
    if group3_probs is None or cp_probs is None:
        return False, {"reason": "missing_v37_aux"}

    ent = group3_probs.get("entertainment", 0.0)
    ent_margin = ent - max(group3_probs.get("sports", 0.0), group3_probs.get("politics", 0.0))
    cp_creator = cp_probs.get("creator", 0.0)
    cp_performer = cp_probs.get("performer", 0.0)
    cp_margin = cp_creator - cp_performer

    ok = (
        ent >= t["entertainment_min"]
        and ent_margin >= t["entertainment_margin_min"]
        and cp_creator >= t["cp_creator_min"]
        and cp_margin >= t["cp_margin_min"]
    )

    return ok, {
        "entertainment_p": ent,
        "entertainment_margin": ent_margin,
        "cp_creator_p": cp_creator,
        "cp_performer_p": cp_performer,
        "cp_margin_creator_minus_performer": cp_margin,
    }


def decide_one(
    cid: str,
    loaded: Dict[str, Dict[str, dict]],
) -> dict:
    # Base model probabilities
    base_model_probs = {}
    true_label = None

    for model_key in BASE_MODEL_WEIGHTS:
        model_rows = loaded.get(model_key)
        if model_rows is None or cid not in model_rows:
            continue
        row = model_rows[cid]
        base_model_probs[model_key] = get_prob_dict(row, LABELS)
        if true_label is None:
            true_label = row.get("true_label")

    if true_label is None:
        # fallback from any available aux file
        for rows in loaded.values():
            if rows is not None and cid in rows:
                true_label = rows[cid].get("true_label")
                break

    base_probs = weighted_mean_probs(base_model_probs, BASE_MODEL_WEIGHTS)
    base_pred, base_top1, base_top2_label, base_top2, base_margin = top2(base_probs)

    # Auxiliary signals
    creator_binary_p = get_creator_binary_prob(
        loaded.get("v3_creator_binary", {}).get(cid) if loaded.get("v3_creator_binary") else None
    )
    occupation_3class_pred = get_3class_pred(
        loaded.get("v3_occupation_3class", {}).get(cid) if loaded.get("v3_occupation_3class") else None
    )
    group3_probs = get_group3_probs(
        loaded.get("v37_group3", {}).get(cid) if loaded.get("v37_group3") else None
    )
    cp_probs = get_cp_probs(
        loaded.get("v37_creator_performer", {}).get(cid) if loaded.get("v37_creator_performer") else None
    )

    final = base_pred
    decision = "base_keep"
    gate_info = {}

    # 1. Creator rescue
    ok_creator, info_creator = creator_rescue_condition(
        base_probs=base_probs,
        base_model_probs=base_model_probs,
        creator_binary_p=creator_binary_p,
        group3_probs=group3_probs,
        cp_probs=cp_probs,
    )
    gate_info["creator_rescue"] = info_creator

    if ok_creator and base_pred != "creator":
        final = "creator"
        decision = "creator_rescue"
    else:
        # 2. Sports/politics rescue from performer absorption
        if base_pred == GATING_THRESHOLDS["rescue_base_label"]:
            ok_sports, info_sports = rescue_from_performer_condition(
                label="sports",
                base_probs=base_probs,
                base_model_probs=base_model_probs,
                group3_probs=group3_probs,
                occupation_3class_pred=occupation_3class_pred,
            )
            ok_politics, info_politics = rescue_from_performer_condition(
                label="politics",
                base_probs=base_probs,
                base_model_probs=base_model_probs,
                group3_probs=group3_probs,
                occupation_3class_pred=occupation_3class_pred,
            )
            gate_info["sports_rescue"] = info_sports
            gate_info["politics_rescue"] = info_politics

            if ok_sports and not ok_politics:
                final = "sports"
                decision = "sports_rescue_from_performer"
            elif ok_politics and not ok_sports:
                final = "politics"
                decision = "politics_rescue_from_performer"
            elif ok_sports and ok_politics:
                # Tie-breaker by base probability
                if base_probs.get("sports", 0.0) >= base_probs.get("politics", 0.0):
                    final = "sports"
                    decision = "sports_rescue_from_performer_tiebreak"
                else:
                    final = "politics"
                    decision = "politics_rescue_from_performer_tiebreak"

        # 3. V3.7 creator-only specialist override
        if final == base_pred:
            ok_v37_creator, info_v37_creator = v37_creator_specialist_condition(group3_probs, cp_probs)
            gate_info["v37_creator_specialist"] = info_v37_creator
            if ok_v37_creator and base_pred != "creator":
                final = "creator"
                decision = "v37_creator_specialist_override"

    final_probs = dict(base_probs)
    # Store final prediction as one-hot only for compatibility with previous prediction format.
    # Base probabilities remain available separately.
    final_one_hot = {label: 1.0 if label == final else 0.0 for label in LABELS}

    return {
        "celebrity_id": cid,
        "true_label": true_label,
        "pred_label": final,
        "correct": bool(final == true_label),
        "probability_by_label": final_one_hot,
        "probabilities": [final_one_hot[label] for label in LABELS],
        "labels": LABELS,
        "base_pred_label": base_pred,
        "base_probability_by_label": final_probs,
        "base_top1_prob": base_top1,
        "base_top2_label": base_top2_label,
        "base_top2_prob": base_top2,
        "base_margin": base_margin,
        "decision": decision,
        "model_predictions": {k: argmax_label(v) for k, v in base_model_probs.items()},
        "model_probabilities": base_model_probs,
        "aux": {
            "creator_binary_p": creator_binary_p,
            "occupation_3class_pred": occupation_3class_pred,
            "v37_group3_probs": group3_probs,
            "v37_creator_performer_probs": cp_probs,
        },
        "gate_info": gate_info,
        "version": "bertweet_fusion_v5",
    }


# -------------------------------------------------------------------
# Evaluation/reporting
# -------------------------------------------------------------------

def build_metrics(predictions: List[dict]) -> dict:
    y_true = [p["true_label"] for p in predictions]
    y_pred = [p["pred_label"] for p in predictions]
    base_pred = [p["base_pred_label"] for p in predictions]

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    base_report = classification_report(
        y_true,
        base_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    base_cm = confusion_matrix(y_true, base_pred, labels=LABELS).tolist()

    decision_counts = Counter(p["decision"] for p in predictions)
    mistake_counts = Counter(
        f"{p['true_label']}->{p['pred_label']}"
        for p in predictions
        if p["true_label"] != p["pred_label"]
    )

    return {
        "version": "bertweet_fusion_v5",
        "labels": LABELS,
        "num_predictions": len(predictions),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm,
        "base_accuracy": float(accuracy_score(y_true, base_pred)),
        "base_macro_f1": float(f1_score(y_true, base_pred, labels=LABELS, average="macro", zero_division=0)),
        "base_classification_report": base_report,
        "base_confusion_matrix": base_cm,
        "decision_counts": dict(decision_counts),
        "top_mistakes": dict(mistake_counts.most_common(20)),
        "thresholds": GATING_THRESHOLDS,
        "base_model_weights": BASE_MODEL_WEIGHTS,
    }


def write_markdown_report(metrics: dict, path: str) -> None:
    labels = metrics["labels"]

    def cm_to_md(cm: List[List[int]]) -> str:
        header = "| true\\pred | " + " | ".join(labels) + " |\n"
        sep = "|---|" + "---:|" * len(labels) + "\n"
        rows = []
        for label, row in zip(labels, cm):
            rows.append("| " + label + " | " + " | ".join(str(x) for x in row) + " |")
        return header + sep + "\n".join(rows)

    lines = []
    lines.append("# BERTweetFusionV5 – Controlled Confidence Gating Report\n")
    lines.append("## Result\n")
    lines.append(f"- Accuracy: `{metrics['accuracy']:.4f}`")
    lines.append(f"- Macro-F1: `{metrics['macro_f1']:.4f}`")
    lines.append(f"- Base Accuracy before gating: `{metrics['base_accuracy']:.4f}`")
    lines.append(f"- Base Macro-F1 before gating: `{metrics['base_macro_f1']:.4f}`\n")

    lines.append("## Decision counts\n")
    lines.append("| Decision | Count |")
    lines.append("|---|---:|")
    for decision, count in metrics["decision_counts"].items():
        lines.append(f"| {decision} | {count} |")

    lines.append("\n## Confusion Matrix – Final Gated\n")
    lines.append(cm_to_md(metrics["confusion_matrix"]))

    lines.append("\n## Confusion Matrix – Base Fusion Only\n")
    lines.append(cm_to_md(metrics["base_confusion_matrix"]))

    lines.append("\n## Per-class F1 – Final\n")
    lines.append("| Label | Precision | Recall | F1 | Support |")
    lines.append("|---|---:|---:|---:|---:|")
    for label in labels:
        item = metrics["classification_report"][label]
        lines.append(
            f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1-score']:.4f} | {int(item['support'])} |"
        )

    lines.append("\n## Top mistakes\n")
    lines.append("| Mistake | Count |")
    lines.append("|---|---:|")
    for mistake, count in metrics["top_mistakes"].items():
        lines.append(f"| {mistake} | {count} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BERTweetFusionV5 controlled confidence gating")
    parser.add_argument("--split", default="test", choices=["test"], help="Currently implemented for test prediction files.")
    args = parser.parse_args()

    ensure_dirs()

    model_keys = list(PREDICTION_PATHS.keys())
    loaded = {}
    for key in model_keys:
        data = load_prediction_file(key)
        if data is not None:
            loaded[key] = data

    # Use IDs present in at least one base 4-class model. Prefer intersection of active base models.
    active_base_sets = [set(loaded[k].keys()) for k in BASE_MODEL_WEIGHTS if k in loaded]
    if not active_base_sets:
        raise RuntimeError("No active base 4-class predictions found.")

    common_ids = set.intersection(*active_base_sets)
    print(f"[INFO] Common celebrity ids across active base models: {len(common_ids)}")

    predictions = [decide_one(cid, loaded) for cid in sorted(common_ids)]

    out_pred_path = os.path.join(PREDICTIONS_DIR, "occupation_test_fusion_v5_gating_d_predictions.json")
    out_metrics_path = os.path.join(METRICS_DIR, "occupation_test_fusion_v5_gating_d_metrics.json")
    out_report_path = os.path.join(REPORTS_DIR, "occupation_test_fusion_v5_gating_d_report.md")

    metrics = build_metrics(predictions)
    save_json(predictions, out_pred_path)
    save_json(metrics, out_metrics_path)
    write_markdown_report(metrics, out_report_path)

    print(f"[OK] Saved predictions: {out_pred_path}")
    print(f"[OK] Saved metrics:     {out_metrics_path}")
    print(f"[OK] Saved report:      {out_report_path}")
    print(f"[RESULT] FusionV5 acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"[BASE]   BaseFusion acc={metrics['base_accuracy']:.4f} macro_f1={metrics['base_macro_f1']:.4f}")
    print(f"[INFO] Decision counts: {metrics['decision_counts']}")


if __name__ == "__main__":
    main()
