"""
Mine text-derived semantic features from PAN follower-feed texts for HybridV4 error analysis.

This script is deliberately diagnostic/post-hoc: it mines contrastive terms from the same
prediction split that it analyses. Do not report the rule simulations as final tuned test scores.
Use the results to motivate validation-based meta-features or qualitative error analysis.

Run from project root, e.g.:
    python -m Models.HybridV4.analysis.mine_text_derived_semantic_features

Outputs:
    outputs/hybrid_v4/error_analysis/text_derived_semantic_features/
        gender_text_derived_semantic_features.md/json
        occupation_text_derived_semantic_features.md/json
        text_derived_semantic_features_summary.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_VERSION = "text-derived-semantic-features-v2-ndjson-fix"

# -----------------------------------------------------------------------------
# Project imports in the same style as the existing HybridV4 analysis scripts.
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from _constants import (  # type: ignore
        test_feeds_path,
        test_label_path,
        hybrid_v4_fusion_predictions_dir,
        hybrid_v4_output_dir,
    )
except Exception as exc:  # pragma: no cover - helpful local error
    raise RuntimeError(
        "Could not import required paths from _constants.py. Run this from the project root "
        "or make sure _constants.py defines test_feeds_path, test_label_path, "
        "hybrid_v4_fusion_predictions_dir and hybrid_v4_output_dir."
    ) from exc

OUT_DIR = Path(hybrid_v4_output_dir) / "error_analysis" / "text_derived_semantic_features"

ID_KEYS = ["id", "celebrity_id", "profile_id", "author_id", "user_id"]
PRED_KEYS = [
    "fusion_pred_label",
    "pred_label",
    "prediction",
    "pred",
    "predicted",
    "predicted_label",
    "y_pred",
    "label_pred",
    "class",
    "output",
]
TRUE_KEYS = ["true_label", "gold_label", "label", "target_label", "y_true", "truth"]

TASK_LABELS = {
    "gender": ["male", "female"],
    "occupation": ["creator", "performer", "politics", "sports"],
}

# Conservative English stopword list plus Twitter/noise terms. Keep domain words such as
# vote, fans, movie, game, president, women, etc.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you",
    "your", "yours", "yourself", "yourselves", "rt", "http", "https", "amp", "com", "www",
    "tco", "via", "get", "got", "one", "two", "new", "like", "also", "would", "really",
    "much", "many", "today", "tomorrow", "yesterday", "time", "day", "year", "years",
}

TOKEN_RE = re.compile(r"[a-z][a-z']{2,}")

# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------

def read_json_or_ndjson(path: str | Path) -> Any:
    """Read either JSON or NDJSON.

    PAN label/feed files are NDJSON: each line is its own JSON object.
    A naive `json.loads(full_file)` fails on these files with
    `JSONDecodeError: Extra data`, because line 2 starts a new JSON object.
    Therefore we first try full JSON for normal .json files and fall back to
    line-by-line NDJSON whenever full parsing fails.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse {path} as JSON/NDJSON at line {lineno}: {exc}") from exc
        return rows


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts: List[str] = []
        # Prefer common text fields first, then fall back recursively.
        for key in ["text", "tweet", "content", "body", "full_text"]:
            if key in value:
                parts.append(flatten_text(value.get(key)))
        if parts:
            return " ".join(p for p in parts if p)
        return " ".join(flatten_text(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(flatten_text(v) for v in value)
    return str(value)


def first_existing(row: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def normalize_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    return str(value).strip().lower()


def load_labels(task: str) -> Dict[str, str]:
    data = read_json_or_ndjson(test_label_path)
    labels: Dict[str, str] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        cid = first_existing(row, ID_KEYS)
        if cid is None:
            continue
        # PAN labels may have columns gender/occupation or a generic target/label shape.
        val = row.get(task)
        if val is None and str(row.get("target", "")).lower() == task:
            val = first_existing(row, TRUE_KEYS)
        if val is None:
            continue
        lab = normalize_label(val)
        if lab:
            labels[str(cid)] = lab
    return labels


def load_feeds() -> Dict[str, str]:
    data = read_json_or_ndjson(test_feeds_path)
    feeds: Dict[str, str] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        cid = first_existing(row, ID_KEYS)
        if cid is None:
            continue
        # analyze_pan_contrast.py style: row["text"] contains nested follower feed text.
        if "text" in row:
            text = flatten_text(row.get("text"))
        else:
            text = flatten_text(row)
        feeds[str(cid)] = text
    return feeds


def candidate_prediction_files(task: str) -> List[Path]:
    pred_dir = Path(hybrid_v4_fusion_predictions_dir)
    patterns = [
        f"{task}_test_*fusion_predictions*.json",
        f"{task}*test*pred*.json",
        f"*{task}*test*pred*.json",
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend(Path(p) for p in glob(str(pred_dir / pat)))
    unique = sorted(set(files), key=lambda p: ("best" not in p.name.lower(), len(p.name), p.name))
    return unique


def normalize_prediction_rows(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if not isinstance(data, dict):
        return []

    for key in ["predictions", "rows", "data", "results", "items", "profiles"]:
        val = data.get(key)
        if isinstance(val, list):
            return [r for r in val if isinstance(r, dict)]
        if isinstance(val, dict):
            rows = []
            for cid, pred_val in val.items():
                if isinstance(pred_val, dict):
                    row = dict(pred_val)
                    row.setdefault("id", cid)
                    rows.append(row)
                else:
                    rows.append({"id": cid, "prediction": pred_val})
            return rows

    # Dict of id -> row/prediction.
    rows = []
    for cid, val in data.items():
        if isinstance(val, dict):
            row = dict(val)
            row.setdefault("id", cid)
            rows.append(row)
        elif isinstance(val, str):
            rows.append({"id": cid, "prediction": val})
    return rows


def load_predictions(task: str, labels: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Path]:
    files = candidate_prediction_files(task)
    if not files:
        raise FileNotFoundError(
            f"No prediction file found for task={task} in {hybrid_v4_fusion_predictions_dir}"
        )

    best_rows: List[Dict[str, Any]] = []
    best_file = files[0]
    debug: List[str] = []

    for path in files:
        data = read_json_or_ndjson(path)
        rows = normalize_prediction_rows(data)
        parsed: List[Dict[str, Any]] = []
        skipped = Counter()
        for row in rows:
            cid = first_existing(row, ID_KEYS)
            if cid is None:
                skipped["missing_id"] += 1
                continue
            cid = str(cid)
            pred = normalize_label(first_existing(row, PRED_KEYS))
            true = normalize_label(row.get(task) or first_existing(row, TRUE_KEYS) or labels.get(cid))
            if not pred:
                skipped["missing_prediction"] += 1
                continue
            if not true:
                skipped["missing_true"] += 1
                continue
            if pred not in TASK_LABELS[task] or true not in TASK_LABELS[task]:
                skipped["unknown_label"] += 1
                continue
            parsed.append({
                "id": cid,
                "true": true,
                "pred": pred,
                "raw_prediction_row": row,
            })
        debug.append(f"{path.name}: parsed={len(parsed)}, skipped={dict(skipped)}")
        if len(parsed) > len(best_rows):
            best_rows = parsed
            best_file = path

    if not best_rows:
        raise ValueError("No usable predictions found. Tried: " + " | ".join(debug))
    return best_rows, best_file

# -----------------------------------------------------------------------------
# Text mining
# -----------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    text = text.lower()
    # Remove URLs, mentions and common HTML artifacts.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    toks = TOKEN_RE.findall(text)
    return [t for t in toks if t not in STOPWORDS and len(t) >= 3]


def build_ngrams(tokens: Sequence[str], n: int) -> Iterable[str]:
    if n <= 1:
        yield from tokens
        return
    for i in range(0, max(0, len(tokens) - n + 1)):
        gram = tokens[i : i + n]
        if any(t in STOPWORDS for t in gram):
            continue
        yield " ".join(gram)


def count_terms(rows: Sequence[Dict[str, Any]], ngram_range: Tuple[int, int]) -> Tuple[Counter, int, int]:
    counts: Counter = Counter()
    token_total = 0
    doc_total = 0
    for row in rows:
        toks = row.get("tokens_list") or []
        if not toks:
            continue
        doc_total += 1
        token_total += len(toks)
        for n in range(ngram_range[0], ngram_range[1] + 1):
            counts.update(build_ngrams(toks, n))
    return counts, token_total, doc_total


def log_odds_terms(
    a_rows: Sequence[Dict[str, Any]],
    b_rows: Sequence[Dict[str, Any]],
    ngram_range: Tuple[int, int] = (1, 2),
    top_k: int = 40,
    min_count: int = 5,
    alpha: float = 0.01,
) -> List[Dict[str, Any]]:
    a_counts, a_total, _ = count_terms(a_rows, ngram_range)
    b_counts, b_total, _ = count_terms(b_rows, ngram_range)
    vocab = set(a_counts) | set(b_counts)
    v = max(len(vocab), 1)
    terms: List[Dict[str, Any]] = []
    for term in vocab:
        ca = a_counts.get(term, 0)
        cb = b_counts.get(term, 0)
        if ca + cb < min_count:
            continue
        # Smoothed log odds. Positive => more characteristic of A than B.
        pa = (ca + alpha) / (a_total + alpha * v)
        pb = (cb + alpha) / (b_total + alpha * v)
        score = math.log(pa / pb)
        # Simple z-like scale to stabilize rare terms.
        var = 1.0 / (ca + alpha) + 1.0 / (cb + alpha)
        z = score / math.sqrt(var)
        terms.append({
            "term": term,
            "score": z,
            "log_odds": score,
            "count_a": ca,
            "count_b": cb,
            "per_100k_a": 100000.0 * ca / max(a_total, 1),
            "per_100k_b": 100000.0 * cb / max(b_total, 1),
        })
    terms.sort(key=lambda x: x["score"], reverse=True)
    return terms[:top_k]


def term_score(row: Dict[str, Any], terms: Sequence[str]) -> float:
    toks = row.get("tokens_list") or []
    if not toks or not terms:
        return 0.0
    term_set = set(terms)
    counts = Counter()
    for n in (1, 2):
        counts.update(build_ngrams(toks, n))
    hits = sum(counts.get(t, 0) for t in term_set)
    return 1000.0 * hits / max(len(toks), 1)


def top_hits(row: Dict[str, Any], max_items: int = 8) -> str:
    toks = row.get("tokens_list") or []
    if not toks:
        return ""
    counts = Counter()
    for n in (1, 2):
        counts.update(build_ngrams(toks, n))
    items = [(k, v) for k, v in counts.most_common(50) if v >= 2]
    return ", ".join(f"{k}:{v}" for k, v in items[:max_items])

# -----------------------------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------------------------

def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    f1s: List[float] = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return mean(f1s)


def metrics(rows: Sequence[Dict[str, Any]], pred_key: str = "pred") -> Dict[str, float]:
    labels = sorted({r["true"] for r in rows} | {r[pred_key] for r in rows})
    y_true = [r["true"] for r in rows]
    y_pred = [r[pred_key] for r in rows]
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(rows), 1)
    return {"n": len(rows), "accuracy": acc, "macro_f1": macro_f1(y_true, y_pred, labels)}


def cohen_d(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = mean(a), mean(b)
    sa, sb = stdev(a), stdev(b)
    pooled = math.sqrt(((len(a) - 1) * sa * sa + (len(b) - 1) * sb * sb) / (len(a) + len(b) - 2))
    return (ma - mb) / pooled if pooled else 0.0


def group_name(row: Dict[str, Any]) -> str:
    return f"correct_{row['true']}" if row["true"] == row["pred"] else f"{row['true']}_to_{row['pred']}"


def build_rows(task: str) -> Tuple[List[Dict[str, Any]], Path, Dict[str, Any]]:
    labels = load_labels(task)
    feeds = load_feeds()
    preds, pred_file = load_predictions(task, labels)
    rows: List[Dict[str, Any]] = []
    feed_matched = 0
    text_nonempty = 0
    for p in preds:
        cid = p["id"]
        text = feeds.get(cid, "")
        if cid in feeds:
            feed_matched += 1
        toks = tokenize(text)
        if toks:
            text_nonempty += 1
        row = {
            "id": cid,
            "true": p["true"],
            "pred": p["pred"],
            "group": group_name(p),
            "text_len": len(text),
            "tokens": len(toks),
            "tokens_list": toks,
        }
        rows.append(row)
    meta = {
        "labels_loaded": len(labels),
        "feed_rows": len(feeds),
        "feed_matched": feed_matched,
        "text_nonempty": text_nonempty,
        "prediction_file": str(pred_file),
    }
    return rows, pred_file, meta


def contrast_specs(task: str, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = Counter(r["group"] for r in rows)
    specs: List[Dict[str, Any]] = []

    if task == "gender":
        desired = [
            ("female_to_male", "correct_male", "female", "male"),
            ("male_to_female", "correct_female", "male", "female"),
            ("correct_female", "correct_male", "female", "male"),
            ("correct_male", "correct_female", "male", "female"),
        ]
    else:
        desired = [
            ("politics_to_creator", "correct_creator", "politics", "creator"),
            ("creator_to_performer", "correct_performer", "creator", "performer"),
            ("sports_to_performer", "correct_performer", "sports", "performer"),
            ("creator_to_sports", "correct_sports", "creator", "sports"),
            ("creator_to_politics", "correct_politics", "creator", "politics"),
            ("correct_politics", "correct_creator", "politics", "creator"),
            ("correct_sports", "correct_performer", "sports", "performer"),
        ]

    for a_group, b_group, true_label, pred_label in desired:
        if groups.get(a_group, 0) >= 3 and groups.get(b_group, 0) >= 3:
            safe_name = f"auto_{a_group}_vs_{b_group}"
            specs.append({
                "name": safe_name,
                "a_group": a_group,
                "b_group": b_group,
                "true_label": true_label,
                "pred_label": pred_label,
                "feature": f"{safe_name}_score",
            })
    return specs


def mine_features(task: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)

    specs = contrast_specs(task, rows)
    for spec in specs:
        a_rows = by_group[spec["a_group"]]
        b_rows = by_group[spec["b_group"]]
        top = log_odds_terms(a_rows, b_rows, ngram_range=(1, 2), top_k=40, min_count=5)
        spec["top_terms"] = top
        spec["terms"] = [t["term"] for t in top[:25]]
        for row in rows:
            row[spec["feature"]] = term_score(row, spec["terms"])
    return specs


def group_means(rows: Sequence[Dict[str, Any]], features: Sequence[str]) -> List[Dict[str, Any]]:
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    out = []
    for group in sorted(by_group):
        rs = by_group[group]
        item = {"group": group, "n": len(rs)}
        for f in features:
            item[f] = mean([float(r.get(f, 0.0)) for r in rs])
        out.append(item)
    return out


def comparisons(rows: Sequence[Dict[str, Any]], specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    out = []
    for spec in specs:
        a = by_group.get(spec["a_group"], [])
        b = by_group.get(spec["b_group"], [])
        if not a or not b:
            continue
        f = spec["feature"]
        av = [float(r.get(f, 0.0)) for r in a]
        bv = [float(r.get(f, 0.0)) for r in b]
        out.append({
            "contrast": spec["name"],
            "feature": f,
            "a_group": spec["a_group"],
            "b_group": spec["b_group"],
            "mean_a": mean(av),
            "mean_b": mean(bv),
            "difference": mean(av) - mean(bv),
            "cohen_d": cohen_d(av, bv),
        })
    return out


def simulate_rules(task: str, rows: Sequence[Dict[str, Any]], specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base = metrics(rows)
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    results = []

    for spec in specs:
        # Only simulate directional correction for actual error contrasts A_to_B vs correct_B.
        if "_to_" not in spec["a_group"]:
            continue
        feature = spec["feature"]
        pred_label = spec["pred_label"]
        true_label = spec["true_label"]
        b_rows = by_group.get(spec["b_group"], [])
        if not b_rows:
            continue
        vals_b = [float(r.get(feature, 0.0)) for r in b_rows]
        threshold = mean(vals_b) + stdev(vals_b)

        changed = []
        new_rows = []
        for r in rows:
            nr = dict(r)
            nr["new_pred"] = r["pred"]
            if r["pred"] == pred_label and float(r.get(feature, 0.0)) >= threshold:
                nr["new_pred"] = true_label
                improved = r["pred"] != r["true"] and nr["new_pred"] == r["true"]
                harmed = r["pred"] == r["true"] and nr["new_pred"] != r["true"]
                changed.append({
                    "id": r["id"],
                    "true": r["true"],
                    "old_pred": r["pred"],
                    "new_pred": nr["new_pred"],
                    "improved": improved,
                    "harmed": harmed,
                    "tokens": r.get("tokens", 0),
                    "score": float(r.get(feature, 0.0)),
                    "top_hits": top_hits(r),
                })
            new_rows.append(nr)

        y_true = [r["true"] for r in new_rows]
        y_pred = [r["new_pred"] for r in new_rows]
        labels = TASK_LABELS[task]
        after = {
            "n": len(new_rows),
            "accuracy": sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(new_rows), 1),
            "macro_f1": macro_f1(y_true, y_pred, labels),
        }
        before_changed = [c for c in changed]
        local_before = mean([1.0 if c["old_pred"] == c["true"] else 0.0 for c in before_changed]) if before_changed else 0.0
        local_after = mean([1.0 if c["new_pred"] == c["true"] else 0.0 for c in before_changed]) if before_changed else 0.0
        results.append({
            "rule": f"{true_label}_from_{pred_label}_high_{feature}",
            "change": f"{pred_label} -> {true_label}",
            "condition": f"pred == {pred_label} AND {feature} >= {threshold:.4f}",
            "feature": feature,
            "threshold": threshold,
            "changed": len(changed),
            "improved": sum(1 for c in changed if c["improved"]),
            "harmed": sum(1 for c in changed if c["harmed"]),
            "still_wrong": sum(1 for c in changed if (not c["improved"] and not c["harmed"])),
            "global_delta_accuracy": after["accuracy"] - base["accuracy"],
            "global_delta_macro_f1": after["macro_f1"] - base["macro_f1"],
            "global_after_accuracy": after["accuracy"],
            "global_after_macro_f1": after["macro_f1"],
            "local_before_accuracy": local_before,
            "local_after_accuracy": local_after,
            "examples": changed[:30],
        })
    return results

# -----------------------------------------------------------------------------
# Markdown rendering
# -----------------------------------------------------------------------------

def fmt(x: Any, digits: int = 4) -> str:
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
    return "\n".join(lines)


def render_report(task: str, analysis: Dict[str, Any]) -> str:
    rows = analysis["rows"]
    features = [s["feature"] for s in analysis["specs"]]
    lines: List[str] = []
    lines.append(f"# Text-derived Semantic Feature Mining: {task}")
    lines.append("")
    lines.append("> Diagnostic post-hoc mining from PAN follower-feed texts and existing HybridV4 predictions. Do not report rule results as final tuned test performance.")
    lines.append("")
    lines.append(f"Script version: `{SCRIPT_VERSION}`")
    lines.append("")
    lines.append(f"Prediction file: `{analysis['meta']['prediction_file']}`")
    lines.append("")
    lines.append(f"Profiles: **{len(rows)}**")
    lines.append(f"PAN feed rows: **{analysis['meta']['feed_rows']}**")
    lines.append(f"Matched feed rows: **{analysis['meta']['feed_matched']}**")
    lines.append(f"Profiles with follower text: **{analysis['meta']['text_nonempty']}**")
    lines.append("")

    base = analysis["baseline"]
    lines.append("## Global baseline")
    lines.append("")
    lines.append(md_table(["n", "accuracy", "macro_f1"], [[base["n"], base["accuracy"], base["macro_f1"]]]))
    lines.append("")

    lines.append("## Mined contrast lexicons")
    lines.append("")
    for spec in analysis["specs"]:
        lines.append(f"### `{spec['name']}`")
        lines.append("")
        lines.append(f"Feature: `{spec['feature']}`")
        lines.append(f"Contrast: `{spec['a_group']}` vs `{spec['b_group']}`")
        top_rows = [[t["term"], t["score"], t["count_a"], t["count_b"], t["per_100k_a"], t["per_100k_b"]] for t in spec.get("top_terms", [])[:20]]
        lines.append(md_table(["term", "score", "count A", "count B", "per100k A", "per100k B"], top_rows))
        lines.append("")

    lines.append("## Group means for text-derived features")
    lines.append("")
    gm_rows = []
    for item in analysis["group_means"]:
        gm_rows.append([item["group"], item["n"]] + [item.get(f, 0.0) for f in features])
    lines.append(md_table(["group", "n"] + features, gm_rows))
    lines.append("")

    lines.append("## Contrast score comparisons")
    lines.append("")
    comp_rows = [[c["contrast"], c["feature"], c["a_group"], c["b_group"], c["mean_a"], c["mean_b"], c["difference"], c["cohen_d"]] for c in analysis["comparisons"]]
    lines.append(md_table(["contrast", "feature", "A", "B", "mean A", "mean B", "A-B", "d"], comp_rows))
    lines.append("")

    lines.append("## Diagnostic rule simulation")
    lines.append("")
    rr = analysis["rule_results"]
    rule_rows = [[r["rule"], r["change"], r["condition"], r["changed"], r["improved"], r["harmed"], r["still_wrong"], r["global_delta_accuracy"], r["global_delta_macro_f1"], r["local_before_accuracy"], r["local_after_accuracy"]] for r in rr]
    lines.append(md_table(["rule", "change", "condition", "changed", "improved", "harmed", "still wrong", "global Δacc", "global ΔF1", "local before", "local after"], rule_rows))
    lines.append("")

    lines.append("## Changed examples by rule")
    lines.append("")
    for r in rr:
        lines.append(f"### `{r['rule']}`")
        lines.append("")
        if not r["examples"]:
            lines.append("No changed profiles.")
            lines.append("")
            continue
        ex_rows = [[e["id"], e["true"], e["old_pred"], e["new_pred"], e["improved"], e["harmed"], e["tokens"], e["score"], e["top_hits"]] for e in r["examples"][:20]]
        lines.append(md_table(["id", "true", "old pred", "new pred", "improved", "harmed", "tokens", "score", "top text-derived hits"], ex_rows))
        lines.append("")

    lines.append("## Top profiles by mined feature")
    lines.append("")
    for spec in analysis["specs"]:
        f = spec["feature"]
        lines.append(f"### `{f}`")
        lines.append("")
        ranked = sorted(rows, key=lambda x: float(x.get(f, 0.0)), reverse=True)[:15]
        tr = [[r["id"], r["group"], r["true"], r["pred"], r.get("tokens", 0), r.get(f, 0.0), top_hits(r)] for r in ranked]
        lines.append(md_table(["id", "group", "true", "pred", "tokens", "score", "top hits"], tr))
        lines.append("")

    lines.append("## Interpretation note")
    lines.append("")
    lines.append("These features are generated directly from the PAN follower-feed texts by mining contrastive terms and n-grams. Because the lexicons and thresholds are mined post hoc from the analysed split, they should be used as error-analysis evidence and as motivation for validation-based meta-features, not as final test-optimized rules.")
    lines.append("")
    return "\n".join(lines)


def strip_tokens_for_json(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean = []
    for r in rows:
        item = {k: v for k, v in r.items() if k not in {"tokens_list"}}
        clean.append(item)
    return clean


def run_task(task: str) -> Dict[str, Any]:
    print(f"\n========== {task} ==========")
    rows, pred_file, meta = build_rows(task)
    print(f"[INFO] Loaded rows: {len(rows)}")
    print(f"[INFO] Feed match: {meta['feed_matched']} / {meta['feed_rows']}; text nonempty: {meta['text_nonempty']}")
    specs = mine_features(task, rows)
    features = [s["feature"] for s in specs]
    analysis = {
        "task": task,
        "script_version": SCRIPT_VERSION,
        "meta": meta,
        "baseline": metrics(rows),
        "specs": specs,
        "group_means": group_means(rows, features),
        "comparisons": comparisons(rows, specs),
        "rule_results": simulate_rules(task, rows, specs),
        "rows": rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = render_report(task, analysis)
    md_path = OUT_DIR / f"{task}_text_derived_semantic_features.md"
    json_path = OUT_DIR / f"{task}_text_derived_semantic_features.json"
    md_path.write_text(md, encoding="utf-8")

    json_analysis = dict(analysis)
    json_analysis["rows"] = strip_tokens_for_json(rows)
    json_path.write_text(json.dumps(json_analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {md_path}")
    print(f"[OK] Wrote {json_path}")
    return json_analysis


def write_summary(all_results: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Text-derived Semantic Feature Mining Summary",
        "",
        f"Script version: `{SCRIPT_VERSION}`",
        "",
        "> Diagnostic post-hoc analysis. Lexicons and thresholds are mined from the analysed prediction split; do not report rule simulations as final tuned test performance.",
        "",
    ]
    for res in all_results:
        task = res["task"]
        base = res["baseline"]
        lines.append(f"## {task}")
        lines.append("")
        lines.append(f"Rows loaded: **{base['n']}**.")
        lines.append(f"Global baseline accuracy: **{base['accuracy']:.4f}**, macro-F1: **{base['macro_f1']:.4f}**.")
        lines.append("")
        rule_rows = [[r["rule"], r["changed"], r["improved"], r["harmed"], r["global_delta_accuracy"], r["global_delta_macro_f1"], r["local_before_accuracy"], r["local_after_accuracy"]] for r in res["rule_results"]]
        if rule_rows:
            lines.append(md_table(["rule", "changed", "improved", "harmed", "global Δacc", "global ΔF1", "local before", "local after"], rule_rows))
        else:
            lines.append("No rules were simulated.")
        lines.append("")
    lines.append("## Suggested thesis wording")
    lines.append("")
    lines.append("Text-derived semantic features were mined directly from follower-feed text by contrasting error groups with correctly classified comparison groups. The resulting lexicons reveal which terms and n-grams characterize specific confusions. Since the lexicons are mined post hoc from the analysed split, they serve as qualitative error-analysis evidence and as candidates for future validation-based meta-features rather than as final optimized test rules.")
    (OUT_DIR / "text_derived_semantic_features_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["gender", "occupation", "all"], default="all")
    args = parser.parse_args()

    tasks = ["gender", "occupation"] if args.task == "all" else [args.task]
    print(f"[INFO] Running {SCRIPT_VERSION}")
    print(f"[INFO] Output dir: {OUT_DIR}")
    results = []
    for task in tasks:
        results.append(run_task(task))
    write_summary(results)
    print(f"[OK] Wrote {OUT_DIR / 'text_derived_semantic_features_summary.md'}")


if __name__ == "__main__":
    main()
