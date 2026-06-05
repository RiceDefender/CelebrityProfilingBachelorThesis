import argparse
import itertools
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

LABELS = ["sports", "performer", "creator", "politics"]

# Best weighted-grid result from the current FusionV5 run.
# You can override this via CLI if needed.
DEFAULT_WEIGHTS = {
    "v3": 0.30,
    "v34": 0.00,
    "v35": 0.70,
}

PREDICTION_PATHS = {
    "fusion_val": {
        "v3": os.path.join(PROJECT_ROOT, "outputs", "hybrid_v4", "bertweet_probs", "occupation_fusion_val_bertweet_v3_probs.json"),
        "v34": os.path.join(PROJECT_ROOT, "outputs", "hybrid_v4", "bertweet_probs", "occupation_fusion_val_bertweet_v34_probs.json"),
        "v35": os.path.join(PROJECT_ROOT, "outputs", "hybrid_v4", "bertweet_probs", "occupation_fusion_val_bertweet_v35_probs.json"),
    },
    "test": {
        "v3": os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3", "predictions", "occupation_test_predictions.json"),
        "v34": os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3_4_stopwords", "predictions", "occupation_test_predictions.json"),
        "v35": os.path.join(PROJECT_ROOT, "outputs", "hybrid_v4", "bertweet_probs", "occupation_test_bertweet_v35_probs.json"),
    },
}

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "bertweet_fusion_v5", "threshold_search")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
MET_DIR = os.path.join(OUTPUT_DIR, "metrics")
REP_DIR = os.path.join(OUTPUT_DIR, "reports")


def ensure_dirs():
    for p in [OUTPUT_DIR, PRED_DIR, MET_DIR, REP_DIR]:
        os.makedirs(p, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def row_id(row: dict) -> str:
    for k in ["celebrity_id", "id", "author_id"]:
        if k in row:
            return str(row[k])
    raise KeyError(f"No ID field in row keys={list(row.keys())}")


def true_label(row: dict) -> str:
    for k in ["true_label", "label", "occupation"]:
        if k in row and row[k] in LABELS:
            return row[k]
    raise KeyError(f"No true label in row for id={row_id(row)}")


def normalize_probs(raw: Dict[str, float]) -> Dict[str, float]:
    vals = np.array([float(raw.get(l, 0.0)) for l in LABELS], dtype=np.float64)
    s = vals.sum()
    if s <= 0:
        vals[:] = 1.0 / len(LABELS)
    else:
        vals /= s
    return {l: float(v) for l, v in zip(LABELS, vals)}


def get_probs(row: dict) -> Dict[str, float]:
    # Common dict field
    if isinstance(row.get("probability_by_label"), dict):
        return normalize_probs(row["probability_by_label"])

    # Common generic list field
    if isinstance(row.get("probabilities"), list):
        row_labels = row.get("labels", LABELS)
        return normalize_probs({l: p for l, p in zip(row_labels, row["probabilities"])})

    # BERTweet-specific fields:
    # bertweet_v3_probabilities / bertweet_v34_probabilities / bertweet_v35_probabilities
    for key, value in row.items():
        if key.endswith("_probabilities") and isinstance(value, list) and len(value) == len(LABELS):
            row_labels = row.get("labels", LABELS)
            return normalize_probs({l: p for l, p in zip(row_labels, value)})

    # Top-level label probs
    if any(l in row for l in LABELS):
        return normalize_probs({l: row.get(l, 0.0) for l in LABELS})

    # Fallback one-hot pred if necessary
    pred = row.get("pred_label") or row.get("prediction")
    if pred is None:
        for key, value in row.items():
            if key.endswith("_pred_label") and value in LABELS:
                pred = value
                break
    out = {l: 0.0 for l in LABELS}
    if pred in out:
        out[pred] = 1.0
    return out


def load_split(split: str, model_keys: List[str]) -> Dict[str, Dict[str, dict]]:
    loaded = {}
    for model_key in model_keys:
        path = PREDICTION_PATHS[split][model_key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {split}/{model_key}: {path}")
        rows = load_json(path)
        by_id = {row_id(r): r for r in rows}
        loaded[model_key] = by_id
        print(f"[OK] Loaded {split}/{model_key}: {path} ({len(by_id)} celebrities)")
    return loaded


def common_ids(loaded: Dict[str, Dict[str, dict]]) -> List[str]:
    sets = [set(v.keys()) for v in loaded.values()]
    return sorted(set.intersection(*sets))


def weighted_base_probs(loaded, cid: str, weights: Dict[str, float], model_keys: List[str]) -> Dict[str, float]:
    out = {l: 0.0 for l in LABELS}
    active_weight_sum = 0.0
    for m in model_keys:
        w = float(weights.get(m, 0.0))
        if w <= 0:
            continue
        probs = get_probs(loaded[m][cid])
        for l in LABELS:
            out[l] += w * probs[l]
        active_weight_sum += w

    if active_weight_sum <= 0:
        # fallback: simple mean
        for m in model_keys:
            probs = get_probs(loaded[m][cid])
            for l in LABELS:
                out[l] += probs[l] / len(model_keys)
    return normalize_probs(out)


def top_label(probs: Dict[str, float]) -> str:
    return max(probs, key=probs.get)


def second_best(probs: Dict[str, float]) -> Tuple[str, float]:
    items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return items[1]


def vote_count_for_label(loaded, cid: str, model_keys: List[str], label: str, prob_min: float) -> int:
    votes = 0
    for m in model_keys:
        p = get_probs(loaded[m][cid])[label]
        if p >= prob_min:
            votes += 1
    return votes


def max_model_prob(loaded, cid: str, model_keys: List[str], label: str) -> float:
    return max(get_probs(loaded[m][cid])[label] for m in model_keys)


def predict_with_thresholds(loaded, ids, model_keys, weights, variant: str, th: dict):
    rows = []

    for cid in ids:
        base = weighted_base_probs(loaded, cid, weights, model_keys)
        base_pred = top_label(base)
        pred = base_pred
        decision = "base_keep"

        creator_votes = vote_count_for_label(
            loaded, cid, model_keys, "creator", th["creator_vote_prob_min"]
        )
        creator_max = max_model_prob(loaded, cid, model_keys, "creator")
        creator_second_label, creator_second_prob = second_best(base)
        creator_margin_vs_base_top = base["creator"] - max(base[l] for l in LABELS if l != "creator")

        sports_votes = vote_count_for_label(
            loaded, cid, model_keys, "sports", th["rescue_vote_prob_min"]
        )
        politics_votes = vote_count_for_label(
            loaded, cid, model_keys, "politics", th["rescue_vote_prob_min"]
        )

        # Variant 2: creator rescue only
        if variant in ["v2_creator", "v3_creator_sports", "v4_creator_sports_politics"]:
            if (
                base_pred != "creator"
                and base["creator"] >= th["base_creator_min"]
                and creator_max >= th["max_creator_min"]
                and creator_votes >= th["creator_votes_min"]
                and creator_margin_vs_base_top >= th["creator_margin_min"]
            ):
                pred = "creator"
                decision = "creator_rescue"

        # Variant 3: add sports rescue from performer
        if decision == "base_keep" and variant in ["v3_creator_sports", "v4_creator_sports_politics"]:
            if (
                base_pred == "performer"
                and base["sports"] >= th["base_sports_min"]
                and sports_votes >= th["rescue_votes_min"]
            ):
                pred = "sports"
                decision = "sports_rescue_from_performer"

        # Variant 4: add politics rescue from performer
        if decision == "base_keep" and variant in ["v4_creator_sports_politics"]:
            if (
                base_pred == "performer"
                and base["politics"] >= th["base_politics_min"]
                and politics_votes >= th["rescue_votes_min"]
            ):
                pred = "politics"
                decision = "politics_rescue_from_performer"

        label = true_label(loaded[model_keys[0]][cid])
        rows.append({
            "celebrity_id": cid,
            "true_label": label,
            "pred_label": pred,
            "correct": pred == label,
            "base_pred_label": base_pred,
            "probability_by_label": base,
            "probabilities": [base[l] for l in LABELS],
            "labels": LABELS,
            "decision": decision,
            "variant": variant,
            "thresholds": th,
        })

    return rows


def evaluate_rows(rows, variant, thresholds, split, weights, model_keys):
    y_true = [r["true_label"] for r in rows]
    y_pred = [r["pred_label"] for r in rows]
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    mistakes = Counter(f"{t}->{p}" for t, p in zip(y_true, y_pred) if t != p)
    decisions = Counter(r["decision"] for r in rows)
    return {
        "version": "bertweet_fusion_v5_threshold_search",
        "variant": variant,
        "split": split,
        "labels": LABELS,
        "model_order": model_keys,
        "weights_by_model": weights,
        "thresholds": thresholds,
        "num_predictions": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm,
        "decision_counts": dict(decisions),
        "top_mistakes": dict(mistakes.most_common(20)),
    }


def write_report(metrics: dict, path: str):
    lines = [f"# FusionV5 Threshold Search – {metrics['variant']} – {metrics['split']}\n"]
    lines.append(f"- Accuracy: `{metrics['accuracy']:.4f}`")
    lines.append(f"- Macro-F1: `{metrics['macro_f1']:.4f}`")
    lines.append(f"- Models: `{', '.join(metrics['model_order'])}`")
    lines.append(f"- Weights: `{metrics['weights_by_model']}`")
    lines.append(f"- Thresholds: `{metrics['thresholds']}`\n")

    lines.append("## Decision counts\n")
    lines.append("| Decision | Count |")
    lines.append("|---|---:|")
    for k, v in metrics["decision_counts"].items():
        lines.append(f"| {k} | {v} |")

    lines.append("\n## Confusion Matrix\n")
    lines.append("| true\\pred | " + " | ".join(LABELS) + " |")
    lines.append("|---|" + "---:|" * len(LABELS))
    for label, row in zip(LABELS, metrics["confusion_matrix"]):
        lines.append("| " + label + " | " + " | ".join(str(x) for x in row) + " |")

    lines.append("\n## Per-class metrics\n")
    lines.append("| Label | Precision | Recall | F1 | Support |")
    lines.append("|---|---:|---:|---:|---:|")
    for label in LABELS:
        item = metrics["classification_report"][label]
        lines.append(
            f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1-score']:.4f} | {int(item['support'])} |"
        )

    lines.append("\n## Top mistakes\n")
    lines.append("| Mistake | Count |")
    lines.append("|---|---:|")
    for k, v in metrics["top_mistakes"].items():
        lines.append(f"| {k} | {v} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def threshold_grid():
    # Keep this intentionally small enough to run quickly.
    # It is not meant as a final exhaustive search, but as a quick ablation.
    creator_vote_prob_min_values = [0.30, 0.35, 0.40]
    base_creator_min_values = [0.18, 0.22, 0.26, 0.30]
    max_creator_min_values = [0.35, 0.40, 0.45]
    creator_votes_min_values = [1, 2]
    creator_margin_min_values = [-0.20, -0.10, 0.00]

    rescue_vote_prob_min_values = [0.35, 0.40, 0.45]
    base_sports_min_values = [0.25, 0.30, 0.35]
    base_politics_min_values = [0.25, 0.30, 0.35]
    rescue_votes_min_values = [1, 2]

    for values in itertools.product(
        creator_vote_prob_min_values,
        base_creator_min_values,
        max_creator_min_values,
        creator_votes_min_values,
        creator_margin_min_values,
        rescue_vote_prob_min_values,
        base_sports_min_values,
        base_politics_min_values,
        rescue_votes_min_values,
    ):
        (
            creator_vote_prob_min,
            base_creator_min,
            max_creator_min,
            creator_votes_min,
            creator_margin_min,
            rescue_vote_prob_min,
            base_sports_min,
            base_politics_min,
            rescue_votes_min,
        ) = values

        yield {
            "creator_vote_prob_min": creator_vote_prob_min,
            "base_creator_min": base_creator_min,
            "max_creator_min": max_creator_min,
            "creator_votes_min": creator_votes_min,
            "creator_margin_min": creator_margin_min,
            "rescue_vote_prob_min": rescue_vote_prob_min,
            "base_sports_min": base_sports_min,
            "base_politics_min": base_politics_min,
            "rescue_votes_min": rescue_votes_min,
        }


def main():
    parser = argparse.ArgumentParser(description="Search FusionV5 threshold variants on fusion_val and evaluate top-k on test.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--score", choices=["macro_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--weights", default="v3=0.30,v34=0.00,v35=0.70")
    args = parser.parse_args()

    ensure_dirs()

    weights = {}
    for part in args.weights.split(","):
        k, v = part.split("=")
        weights[k.strip()] = float(v)

    model_keys = [m for m in ["v3", "v34", "v35"] if m in weights]
    print(f"[INFO] Models: {model_keys}")
    print(f"[INFO] Weights: {weights}")

    val_loaded = load_split("fusion_val", model_keys)
    test_loaded = load_split("test", model_keys)
    val_ids = common_ids(val_loaded)
    test_ids = common_ids(test_loaded)
    print(f"[INFO] fusion_val IDs: {len(val_ids)}")
    print(f"[INFO] test IDs:       {len(test_ids)}")

    variants = [
        "v2_creator",
        "v3_creator_sports",
        "v4_creator_sports_politics",
    ]

    summary = []

    for variant in variants:
        print(f"\n========== Searching {variant} ==========")
        candidates = []
        for th in threshold_grid():
            rows_val = predict_with_thresholds(val_loaded, val_ids, model_keys, weights, variant, th)
            met_val = evaluate_rows(rows_val, variant, th, "fusion_val", weights, model_keys)
            candidates.append(met_val)

        candidates.sort(key=lambda m: (m[args.score], m["accuracy"], m["classification_report"]["creator"]["f1-score"]), reverse=True)
        top = candidates[: args.top_k]

        for rank, val_metrics in enumerate(top, start=1):
            th = val_metrics["thresholds"]
            rows_test = predict_with_thresholds(test_loaded, test_ids, model_keys, weights, variant, th)
            test_metrics = evaluate_rows(rows_test, variant, th, "test", weights, model_keys)
            test_metrics["selection"] = {
                "selected_by": f"fusion_val_{args.score}",
                "rank": rank,
                "fusion_val_accuracy": val_metrics["accuracy"],
                "fusion_val_macro_f1": val_metrics["macro_f1"],
                "fusion_val_creator_f1": val_metrics["classification_report"]["creator"]["f1-score"],
                "fusion_val_decision_counts": val_metrics["decision_counts"],
            }

            base_name = f"occupation_test_threshold_{variant}_rank{rank}"
            pred_path = os.path.join(PRED_DIR, base_name + "_predictions.json")
            met_path = os.path.join(MET_DIR, base_name + "_metrics.json")
            rep_path = os.path.join(REP_DIR, base_name + "_report.md")

            save_json(rows_test, pred_path)
            save_json(test_metrics, met_path)
            write_report(test_metrics, rep_path)

            creator_f1 = test_metrics["classification_report"]["creator"]["f1-score"]
            creator_recall = test_metrics["classification_report"]["creator"]["recall"]
            print(
                f"[RESULT] {variant} rank={rank} "
                f"val_{args.score}={val_metrics[args.score]:.4f} "
                f"test_acc={test_metrics['accuracy']:.4f} "
                f"test_macro_f1={test_metrics['macro_f1']:.4f} "
                f"test_creator_f1={creator_f1:.4f} "
                f"test_creator_recall={creator_recall:.4f} "
                f"decisions={test_metrics['decision_counts']}"
            )

            summary.append({
                "variant": variant,
                "rank": rank,
                "selected_by": args.score,
                "fusion_val_accuracy": val_metrics["accuracy"],
                "fusion_val_macro_f1": val_metrics["macro_f1"],
                "fusion_val_creator_f1": val_metrics["classification_report"]["creator"]["f1-score"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_creator_f1": creator_f1,
                "test_creator_recall": creator_recall,
                "test_decision_counts": test_metrics["decision_counts"],
                "thresholds": th,
                "metrics_path": met_path,
                "report_path": rep_path,
            })

    summary.sort(key=lambda x: (x["test_macro_f1"], x["test_accuracy"], x["test_creator_f1"]), reverse=True)
    summary_path = os.path.join(MET_DIR, "occupation_threshold_search_summary.json")
    save_json(summary, summary_path)

    md_path = os.path.join(REP_DIR, "occupation_threshold_search_summary.md")
    lines = ["# FusionV5 Threshold Search Summary\n"]
    lines.append("| Variant | Rank | Test Acc | Test Macro-F1 | Creator F1 | Creator Recall | Decisions |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for item in summary:
        lines.append(
            f"| {item['variant']} | {item['rank']} | {item['test_accuracy']:.4f} | "
            f"{item['test_macro_f1']:.4f} | {item['test_creator_f1']:.4f} | "
            f"{item['test_creator_recall']:.4f} | `{item['test_decision_counts']}` |"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[OK] Saved summary: {summary_path}")
    print(f"[OK] Saved summary report: {md_path}")
    print("\nTop 10 by test macro-F1:")
    for item in summary[:10]:
        print(
            f"- {item['variant']} rank={item['rank']}: "
            f"test_acc={item['test_accuracy']:.4f}, "
            f"test_macro_f1={item['test_macro_f1']:.4f}, "
            f"creator_f1={item['test_creator_f1']:.4f}, "
            f"decisions={item['test_decision_counts']}"
        )


if __name__ == "__main__":
    main()
