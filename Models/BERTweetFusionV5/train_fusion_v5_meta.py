import argparse
import itertools
import json
import os
import pickle
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Models.BERTweetFusionV5.config_fusion_v5_trainable import (  # noqa: E402
    EVAL_SPLIT,
    FUSION_TRAIN_SPLIT,
    GRID_STEP,
    LABELS,
    LOGREG_C_VALUES,
    LOGREG_MAX_ITER,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTION_PATHS,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
)


def ensure_dirs():
    for p in [PREDICTIONS_DIR, METRICS_DIR, REPORTS_DIR, MODELS_DIR]:
        os.makedirs(p, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_path(split: str, model_key: str) -> Optional[str]:
    for p in PREDICTION_PATHS.get(split, {}).get(model_key, []):
        if os.path.exists(p):
            return p
    return None


def normalize_probs(raw: Dict[str, float], labels: List[str]) -> Dict[str, float]:
    vals = np.array([float(raw.get(l, 0.0)) for l in labels], dtype=np.float64)
    s = vals.sum()
    if s <= 0:
        vals = np.ones(len(labels), dtype=np.float64) / len(labels)
    else:
        vals = vals / s
    return {l: float(v) for l, v in zip(labels, vals)}


def get_probs(row: dict, labels: List[str]) -> Dict[str, float]:
    """
    Robust probability parser for all current BERTweet/Fusion JSON formats.

    Supported examples:
    - probability_by_label: {"sports": ..., ...}
    - probabilities + labels
    - bertweet_v3_probabilities / bertweet_v34_probabilities / bertweet_v35_probabilities
    - any key ending with "_probabilities" containing a list of len(labels)
    - top-level sports/performer/creator/politics
    - pred_label fallback as one-hot
    """

    # 1) Dict-based probability fields
    dict_keys = [
        "probability_by_label",
        "probabilities_by_label",
        "probs_by_label",
        "class_probabilities",
        "bertweet_probs",
        "probs",
        "prob_by_label",
        "p",
    ]
    for key in dict_keys:
        if key in row and isinstance(row[key], dict):
            return normalize_probs(row[key], labels)

    # 2) Generic list field with labels
    if "probabilities" in row and isinstance(row["probabilities"], list):
        row_labels = row.get("labels", labels)
        return normalize_probs({l: p for l, p in zip(row_labels, row["probabilities"])}, labels)

    # 3) Version-specific BERTweet probability fields
    # Examples:
    # bertweet_v3_probabilities
    # bertweet_v34_probabilities
    # bertweet_v35_probabilities
    # bertweet_v37_probabilities
    for key, value in row.items():
        if (
            key.endswith("_probabilities")
            and isinstance(value, list)
            and len(value) == len(labels)
        ):
            row_labels = row.get("labels", labels)
            return normalize_probs({l: p for l, p in zip(row_labels, value)}, labels)

    # 4) Top-level label probabilities
    if any(l in row for l in labels):
        return normalize_probs({l: row.get(l, 0.0) for l in labels}, labels)

    # 5) Alternative top-level names: prob_sports, sports_prob, p_sports
    alt = {}
    for l in labels:
        for k in [f"prob_{l}", f"{l}_prob", f"p_{l}"]:
            if k in row:
                alt[l] = row[k]
                break
    if alt:
        return normalize_probs(alt, labels)

    # 6) Fallback: one-hot prediction if available
    pred = row.get("pred_label") or row.get("prediction")
    if pred is None:
        # Version-specific pred labels
        for key, value in row.items():
            if key.endswith("_pred_label") and value in labels:
                pred = value
                break

    out = {l: 0.0 for l in labels}
    if pred in out:
        out[pred] = 1.0
    return out


def row_id(row: dict) -> str:
    for key in ["celebrity_id", "id", "author_id"]:
        if key in row:
            return str(row[key])
    raise KeyError(f"Cannot find id key in row keys={list(row.keys())}")


def true_label(row: dict) -> Optional[str]:
    for key in ["true_label", "label", "occupation"]:
        if key in row and row[key] in LABELS:
            return row[key]
    return None


def load_model_predictions(split: str) -> Dict[str, Dict[str, dict]]:
    loaded = {}
    for model_key in PREDICTION_PATHS.get(split, {}):
        path = resolve_path(split, model_key)
        if path is None:
            print(f"[WARN] Missing {split} predictions for {model_key}")
            continue
        rows = load_json(path)
        by_id = {row_id(r): r for r in rows}
        loaded[model_key] = by_id
        print(f"[OK] Loaded {split}/{model_key}: {path} ({len(by_id)} celebrities)")
    return loaded


def active_common_ids(loaded: Dict[str, Dict[str, dict]]) -> List[str]:
    if not loaded:
        raise RuntimeError("No prediction files loaded.")
    sets = [set(v.keys()) for v in loaded.values()]
    common = sorted(set.intersection(*sets))
    if not common:
        raise RuntimeError("No common celebrity IDs across loaded models.")
    return common


def build_matrix(loaded: Dict[str, Dict[str, dict]], ids: List[str], model_order: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X = []
    y = []
    for cid in ids:
        feats = []
        label = None
        for model_key in model_order:
            row = loaded[model_key][cid]
            probs = get_probs(row, LABELS)
            feats.extend([probs[l] for l in LABELS])
            if label is None:
                label = true_label(row)
        if label is None:
            raise ValueError(f"No true label found for celebrity_id={cid}")
        X.append(feats)
        y.append(LABELS.index(label))
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64), ids


def weighted_probs_for_id(loaded: Dict[str, Dict[str, dict]], cid: str, model_order: List[str], weights: np.ndarray) -> Dict[str, float]:
    out = {l: 0.0 for l in LABELS}
    for w, model_key in zip(weights, model_order):
        probs = get_probs(loaded[model_key][cid], LABELS)
        for l in LABELS:
            out[l] += float(w) * probs[l]
    s = sum(out.values())
    if s <= 0:
        return {l: 1.0 / len(LABELS) for l in LABELS}
    return {l: out[l] / s for l in LABELS}


def labels_from_weighted(loaded, ids, model_order, weights):
    y_pred = []
    y_true = []
    rows = []
    for cid in ids:
        probs = weighted_probs_for_id(loaded, cid, model_order, weights)
        pred = max(probs, key=probs.get)
        label = true_label(loaded[model_order[0]][cid])
        y_true.append(label)
        y_pred.append(pred)
        rows.append({
            "celebrity_id": cid,
            "true_label": label,
            "pred_label": pred,
            "correct": pred == label,
            "probability_by_label": probs,
            "probabilities": [probs[l] for l in LABELS],
            "labels": LABELS,
            "fusion_mode": "weighted_grid",
        })
    return y_true, y_pred, rows


def generate_weight_grid(n: int, step: float):
    units = int(round(1.0 / step))
    for counts in itertools.product(range(units + 1), repeat=n):
        if sum(counts) == units:
            yield np.array(counts, dtype=np.float64) / units


def search_weighted_grid(train_loaded, train_ids, model_order, step):
    best = None
    for weights in generate_weight_grid(len(model_order), step):
        y_true, y_pred, _ = labels_from_weighted(train_loaded, train_ids, model_order, weights)
        score = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        if best is None or score > best["macro_f1"] or (score == best["macro_f1"] and acc > best["accuracy"]):
            best = {"weights": weights, "macro_f1": float(score), "accuracy": float(acc)}
    return best


def fit_logreg_grid(X, y):
    best = None
    for C in LOGREG_C_VALUES:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                max_iter=LOGREG_MAX_ITER,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        )
        clf.fit(X, y)
        pred = clf.predict(X)
        macro = f1_score(y, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)
        acc = accuracy_score(y, pred)
        if best is None or macro > best["macro_f1"] or (macro == best["macro_f1"] and acc > best["accuracy"]):
            best = {"model": clf, "C": C, "macro_f1": float(macro), "accuracy": float(acc)}
    return best


def evaluate_prediction_rows(rows: List[dict], mode: str, model_order: List[str], train_info: dict) -> dict:
    y_true = [r["true_label"] for r in rows]
    y_pred = [r["pred_label"] for r in rows]
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    mistakes = Counter(f"{t}->{p}" for t, p in zip(y_true, y_pred) if t != p)
    return {
        "version": "bertweet_fusion_v5_trainable",
        "mode": mode,
        "labels": LABELS,
        "model_order": model_order,
        "train_info": train_info,
        "num_predictions": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm,
        "top_mistakes": dict(mistakes.most_common(20)),
    }


def write_report(metrics: dict, path: str):
    labels = metrics["labels"]
    lines = [f"# FusionV5 Trainable Report – {metrics['mode']}\n"]
    lines.append(f"- Accuracy: `{metrics['accuracy']:.4f}`")
    lines.append(f"- Macro-F1: `{metrics['macro_f1']:.4f}`")
    lines.append(f"- Models: `{', '.join(metrics['model_order'])}`\n")
    lines.append("## Confusion Matrix\n")
    lines.append("| true\\pred | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---:|" * len(labels))
    for label, row in zip(labels, metrics["confusion_matrix"]):
        lines.append("| " + label + " | " + " | ".join(str(x) for x in row) + " |")
    lines.append("\n## Per-class metrics\n")
    lines.append("| Label | Precision | Recall | F1 | Support |")
    lines.append("|---|---:|---:|---:|---:|")
    for label in labels:
        item = metrics["classification_report"][label]
        lines.append(f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1-score']:.4f} | {int(item['support'])} |")
    lines.append("\n## Top mistakes\n")
    lines.append("| Mistake | Count |")
    lines.append("|---|---:|")
    for k, v in metrics["top_mistakes"].items():
        lines.append(f"| {k} | {v} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate trainable BERTweetFusionV5")
    parser.add_argument("--mode", choices=["weighted_grid", "meta_logreg", "both"], default="both")
    parser.add_argument("--train-split", default=FUSION_TRAIN_SPLIT)
    parser.add_argument("--eval-split", default=EVAL_SPLIT)
    parser.add_argument("--grid-step", type=float, default=GRID_STEP)
    args = parser.parse_args()

    ensure_dirs()
    train_loaded = load_model_predictions(args.train_split)
    eval_loaded = load_model_predictions(args.eval_split)

    model_order = sorted(set(train_loaded.keys()).intersection(eval_loaded.keys()))
    if len(model_order) < 2:
        raise RuntimeError(f"Need at least two common models across train/eval. Found: {model_order}")
    print(f"[INFO] Active models: {model_order}")

    train_ids = active_common_ids({k: train_loaded[k] for k in model_order})
    eval_ids = active_common_ids({k: eval_loaded[k] for k in model_order})
    print(f"[INFO] Train IDs common: {len(train_ids)}")
    print(f"[INFO] Eval IDs common:  {len(eval_ids)}")

    if args.mode in ["weighted_grid", "both"]:
        best = search_weighted_grid(train_loaded, train_ids, model_order, args.grid_step)
        weights = best["weights"]
        y_true, y_pred, rows = labels_from_weighted(eval_loaded, eval_ids, model_order, weights)
        for r in rows:
            r["weights_by_model"] = {m: float(w) for m, w in zip(model_order, weights)}
        train_info = {
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "grid_step": args.grid_step,
            "train_macro_f1": best["macro_f1"],
            "train_accuracy": best["accuracy"],
            "weights_by_model": {m: float(w) for m, w in zip(model_order, weights)},
        }
        metrics = evaluate_prediction_rows(rows, "weighted_grid", model_order, train_info)
        pred_path = os.path.join(PREDICTIONS_DIR, f"occupation_{args.eval_split}_fusion_v5_weighted_grid_predictions.json")
        met_path = os.path.join(METRICS_DIR, f"occupation_{args.eval_split}_fusion_v5_weighted_grid_metrics.json")
        rep_path = os.path.join(REPORTS_DIR, f"occupation_{args.eval_split}_fusion_v5_weighted_grid_report.md")
        save_json(rows, pred_path)
        save_json(metrics, met_path)
        write_report(metrics, rep_path)
        print(f"[RESULT] weighted_grid eval_acc={metrics['accuracy']:.4f} eval_macro_f1={metrics['macro_f1']:.4f}")
        print(f"[INFO] weights: {train_info['weights_by_model']}")
        print(f"[OK] Saved: {met_path}")

    if args.mode in ["meta_logreg", "both"]:
        X_train, y_train, _ = build_matrix(train_loaded, train_ids, model_order)
        X_eval, y_eval, _ = build_matrix(eval_loaded, eval_ids, model_order)
        best = fit_logreg_grid(X_train, y_train)
        clf = best["model"]
        pred_ids = clf.predict(X_eval)
        if hasattr(clf, "predict_proba"):
            prob_mat = clf.predict_proba(X_eval)
        else:
            prob_mat = np.eye(len(LABELS))[pred_ids]
        rows = []
        for cid, yi, pi, probs in zip(eval_ids, y_eval, pred_ids, prob_mat):
            prob_dict = {l: float(probs[i]) for i, l in enumerate(LABELS)}
            rows.append({
                "celebrity_id": cid,
                "true_label": LABELS[int(yi)],
                "pred_label": LABELS[int(pi)],
                "correct": bool(int(yi) == int(pi)),
                "probability_by_label": prob_dict,
                "probabilities": [prob_dict[l] for l in LABELS],
                "labels": LABELS,
                "fusion_mode": "meta_logreg",
            })
        train_info = {
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "best_C": best["C"],
            "train_macro_f1": best["macro_f1"],
            "train_accuracy": best["accuracy"],
        }
        metrics = evaluate_prediction_rows(rows, "meta_logreg", model_order, train_info)
        pred_path = os.path.join(PREDICTIONS_DIR, f"occupation_{args.eval_split}_fusion_v5_meta_logreg_predictions.json")
        met_path = os.path.join(METRICS_DIR, f"occupation_{args.eval_split}_fusion_v5_meta_logreg_metrics.json")
        rep_path = os.path.join(REPORTS_DIR, f"occupation_{args.eval_split}_fusion_v5_meta_logreg_report.md")
        mdl_path = os.path.join(MODELS_DIR, f"occupation_fusion_v5_meta_logreg.pkl")
        save_json(rows, pred_path)
        save_json(metrics, met_path)
        write_report(metrics, rep_path)
        with open(mdl_path, "wb") as f:
            pickle.dump({"model": clf, "model_order": model_order, "labels": LABELS, "train_info": train_info}, f)
        print(f"[RESULT] meta_logreg eval_acc={metrics['accuracy']:.4f} eval_macro_f1={metrics['macro_f1']:.4f}")
        print(f"[INFO] best C: {best['C']}")
        print(f"[OK] Saved: {met_path}")


if __name__ == "__main__":
    main()
