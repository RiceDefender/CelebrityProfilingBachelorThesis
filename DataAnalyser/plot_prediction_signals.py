"""
Prediction-conditioned signal analysis for PAN Celebrity Profiling / BERTweet.

This script generalizes the earlier Creator-vs-Performer n-gram analysis.
It can visualize:
  1) frequent n-grams inside prediction groups,
  2) log-odds / overrepresented n-grams against a background group,
  3) simple Twitter-style features by class/outcome.

The analysis is input-level / correlational: it does not inspect BERTweet attention
or gradients. It shows which signals occur in feeds where the model is correct,
wrong, confident, or confused.

Example calls:
  python plot_prediction_signals.py --target occupation --split test --modes frequency logodds style --ngrams 2 3
  python plot_prediction_signals.py --target occupation_3class --split test --modes frequency logodds --ngrams 2 3
  python plot_prediction_signals.py --target gender --split val --modes frequency style --ngrams 2
  python plot_prediction_signals.py --target birthyear --split val --modes frequency logodds style --ngrams 2
  python plot_prediction_signals.py --target creator_binary --split test --modes frequency logodds --ngrams 3

Useful variants:
  # Keep stopwords to reproduce the current diagnostic view:
  python plot_prediction_signals.py --target occupation --split test --ngrams 2

  # Remove stopwords to see cleaner topical signals:
  python plot_prediction_signals.py --target occupation --split test --ngrams 2 --remove-stopwords

  # Analyze all examples in each group, not only top-confident examples:
  python plot_prediction_signals.py --target occupation --split test --top-docs 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------
# Project imports / robust fallback
# ---------------------------------------------------------------------
try:
    from _constants import (  # type: ignore
        root_dir,
        train_feeds_path,
        test_feeds_path,
        plots_dir,
        bertweet_v3_predictions_dir,
    )
except Exception:
    # Allows running the script next to the uploaded files or in notebooks.
    root_dir = os.path.dirname(os.path.abspath(__file__))
    train_feeds_path = os.path.join(root_dir, "data", "train", "follower-feeds.ndjson")
    test_feeds_path = os.path.join(root_dir, "data", "test", "follower-feeds.ndjson")
    plots_dir = os.path.join(root_dir, "plots")
    bertweet_v3_predictions_dir = os.path.join(root_dir, "outputs", "bertweet_v3", "predictions")


# ---------------------------------------------------------------------
# Label order assumptions used by your prediction JSONs
# ---------------------------------------------------------------------
LABEL_ORDERS: Dict[str, List[str]] = {
    "occupation": ["sports", "performer", "creator", "politics"],
    "occupation_3class": ["sports", "performer", "politics"],
    "creator_binary": ["not_creator", "creator"],
    "gender": ["male", "female"],
    "birthyear": ["1994", "1985", "1975", "1963", "1947"],
}

DEFAULT_PRED_FILENAMES: Dict[Tuple[str, str], List[str]] = {
    ("occupation", "test"): ["occupation_test_predictions.json", "occupation_v31_test_predictions.json"],
    ("occupation", "val"): ["occupation_val_predictions.json", "occupation_v31_val_predictions.json"],
    ("occupation", "val_all"): ["occupation_val_all_predictions.json"],
    ("occupation_3class", "test"): ["occupation_3class_test_all_predictions.json"],
    ("occupation_3class", "val"): ["occupation_3class_val_predictions.json"],
    ("occupation_3class", "val_all"): ["occupation_3class_val_all_predictions.json"],
    ("creator_binary", "test"): ["creator_binary_test_predictions.json"],
    ("creator_binary", "val"): ["creator_binary_val_predictions.json"],
    ("creator_binary", "val_all"): ["creator_binary_val_all_predictions.json"],
    ("gender", "test"): ["gender_test_predictions.json"],
    ("gender", "val"): ["gender_val_predictions.json"],
    ("birthyear", "test"): ["birthyear_test_predictions.json"],
    ("birthyear", "val"): ["birthyear_val_predictions.json"],
}

# A compact English stopword list. By default we KEEP stopwords because they are
# part of the diagnostic signal. Activate with --remove-stopwords.
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "don", "dont", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "yourselves",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_']*|#[a-z0-9_]+|@user|httpurl", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u27BF"
    "]+",
    flags=re.UNICODE,
)


@dataclass
class GroupSpec:
    name: str
    predicate: Callable[[dict], bool]
    sort_key: Callable[[dict], float]
    reverse: bool = True


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    text = text.lower().replace("→", "to")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "group"


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_ndjson(path: str | Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_prediction_path(target: str, split: str, explicit_path: Optional[str] = None) -> str:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    for filename in DEFAULT_PRED_FILENAMES.get((target, split), []):
        candidates.append(Path(bertweet_v3_predictions_dir) / filename)
        candidates.append(Path(root_dir) / filename)
        candidates.append(Path.cwd() / filename)

    for p in candidates:
        if p.exists():
            return str(p)

    msg = [
        f"Could not find prediction file for target={target!r}, split={split!r}.",
        "Tried:",
        *[f"  - {p}" for p in candidates],
        "Use --pred-path to specify it manually.",
    ]
    raise FileNotFoundError("\n".join(msg))


def resolve_feed_path(split: str, explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        return explicit_path
    if split == "test":
        return test_feeds_path
    # val and val_all are validation splits from the training data.
    return train_feeds_path


def get_celebrity_id(row: dict) -> Optional[str]:
    cid = row.get("id") or row.get("celebrity_id") or row.get("author_id") or row.get("user_id")
    return None if cid is None else str(cid)


def extract_texts_recursive(obj) -> List[str]:
    texts: List[str] = []
    if obj is None:
        return texts
    if isinstance(obj, str):
        cleaned = obj.strip()
        if cleaned:
            texts.append(cleaned)
        return texts
    if isinstance(obj, list):
        for item in obj:
            texts.extend(extract_texts_recursive(item))
        return texts
    if isinstance(obj, dict):
        # Common tweet text fields
        for key in ["text", "full_text", "tweet", "content"]:
            value = obj.get(key)
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    texts.append(cleaned)
            elif isinstance(value, (list, dict)):
                texts.extend(extract_texts_recursive(value))
        # Common nested containers
        for key in ["tweets", "followers", "feed", "feeds", "items", "data"]:
            value = obj.get(key)
            if isinstance(value, (list, dict)):
                texts.extend(extract_texts_recursive(value))
        return texts
    return texts


def extract_tweets_from_feed_row(row: dict) -> List[str]:
    texts: List[str] = []
    for key in ["tweets", "followers", "text", "feed", "feeds", "items", "data"]:
        if key in row:
            texts.extend(extract_texts_recursive(row[key]))

    # Preserve order while removing duplicates.
    seen = set()
    unique: List[str] = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def load_feeds_for_ids(feed_path: str, ids: Optional[set[str]] = None) -> Dict[str, List[str]]:
    if not os.path.exists(feed_path):
        raise FileNotFoundError(f"Feed file not found: {feed_path}")

    feeds: Dict[str, List[str]] = {}
    for row in iter_ndjson(feed_path):
        cid = get_celebrity_id(row)
        if not cid:
            continue
        if ids is not None and cid not in ids:
            continue
        tweets = extract_tweets_from_feed_row(row)
        if tweets:
            feeds[cid] = tweets
    return feeds


def infer_label_order(target: str, preds: Sequence[dict]) -> List[str]:
    if target in LABEL_ORDERS:
        return LABEL_ORDERS[target]

    labels = sorted({str(p.get("true_label")) for p in preds} | {str(p.get("pred_label")) for p in preds})
    if preds and "probabilities" in preds[0] and len(labels) != len(preds[0]["probabilities"]):
        raise ValueError(
            f"Could not infer label order for {target!r}: labels={labels}, "
            f"probability length={len(preds[0]['probabilities'])}. Add it to LABEL_ORDERS."
        )
    return labels


def confidence_for_label(pred: dict, label: str, label_order: Sequence[str]) -> float:
    probs = pred.get("probabilities") or []
    if label not in label_order:
        return float("nan")
    idx = label_order.index(label)
    if idx >= len(probs):
        return float("nan")
    return float(probs[idx])


def pred_confidence(pred: dict, label_order: Sequence[str]) -> float:
    label = str(pred.get("pred_label"))
    return confidence_for_label(pred, label, label_order)


def margin_confidence(pred: dict) -> float:
    probs = sorted([float(x) for x in pred.get("probabilities", [])], reverse=True)
    if not probs:
        return 0.0
    if len(probs) == 1:
        return probs[0]
    return probs[0] - probs[1]


def normalize_text_for_ngrams(text: str, keep_hashtags: bool = True) -> str:
    text = str(text).lower()
    text = _URL_RE.sub(" HTTPURL ", text)
    text = _MENTION_RE.sub(" @USER ", text)
    if not keep_hashtags:
        text = text.replace("#", " ")
    return text


def tokenize_for_ngrams(
    text: str,
    min_token_len: int = 2,
    remove_stopwords: bool = False,
    keep_hashtags: bool = True,
) -> List[str]:
    text = normalize_text_for_ngrams(text, keep_hashtags=keep_hashtags)
    tokens = [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
    out = []
    for tok in tokens:
        plain = tok[1:] if tok.startswith("#") else tok
        if len(plain) < min_token_len and tok not in {"@user"}:
            continue
        if remove_stopwords and plain in STOP_WORDS:
            continue
        out.append(tok)
    return out


def get_ngrams_from_texts(
    texts: Sequence[str],
    n: int,
    min_token_len: int,
    remove_stopwords: bool,
    keep_hashtags: bool,
    max_tweets: Optional[int] = None,
) -> List[str]:
    phrases: List[str] = []
    iterable = texts if max_tweets is None or max_tweets <= 0 else texts[:max_tweets]
    for t in iterable:
        tokens = tokenize_for_ngrams(t, min_token_len, remove_stopwords, keep_hashtags)
        if len(tokens) >= n:
            phrases.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return phrases


def style_features(texts: Sequence[str]) -> Dict[str, float]:
    joined = "\n".join(texts)
    tokens = tokenize_for_ngrams(joined, min_token_len=1, remove_stopwords=False, keep_hashtags=True)
    num_tweets = len(texts)
    num_chars = len(joined)
    num_tokens = len(tokens)
    uppercase_letters = sum(1 for ch in joined if ch.isupper())
    letters = sum(1 for ch in joined if ch.isalpha())
    love_words = sum(1 for t in tokens if t in {"love", "loved", "lovely", "heart", "hearts"})
    fan_words = sum(1 for t in tokens if t in {"fan", "fans", "follow", "followed", "follower", "followers"})
    politics_words = sum(1 for t in tokens if t in {"vote", "voted", "election", "president", "government", "senate", "congress"})
    sports_words = sum(1 for t in tokens if t in {"game", "team", "win", "won", "season", "league", "match", "goal", "player"})
    return {
        "num_tweets": float(num_tweets),
        "num_chars": float(num_chars),
        "num_tokens": float(num_tokens),
        "avg_chars_per_tweet": num_chars / max(num_tweets, 1),
        "avg_tokens_per_tweet": num_tokens / max(num_tweets, 1),
        "url_count": float(len(_URL_RE.findall(joined))),
        "mention_count": float(len(_MENTION_RE.findall(joined))),
        "hashtag_count": float(len(_HASHTAG_RE.findall(joined))),
        "emoji_count": float(len(_EMOJI_RE.findall(joined))),
        "exclamation_count": float(joined.count("!")),
        "question_count": float(joined.count("?")),
        "uppercase_ratio": uppercase_letters / max(letters, 1),
        "love_word_count": float(love_words),
        "fan_word_count": float(fan_words),
        "politics_word_count": float(politics_words),
        "sports_word_count": float(sports_words),
    }


def select_top(preds: Sequence[dict], spec: GroupSpec, top_docs: int) -> List[dict]:
    matches = [p for p in preds if spec.predicate(p)]
    matches = sorted(matches, key=spec.sort_key, reverse=spec.reverse)
    if top_docs and top_docs > 0:
        return matches[:top_docs]
    return matches


def build_group_specs(target: str, preds: Sequence[dict], label_order: Sequence[str], analysis: str) -> List[GroupSpec]:
    labels = list(label_order)

    if target == "birthyear" and analysis == "age_direction":
        def year(x):
            try:
                return int(x)
            except Exception:
                return 0

        return [
            GroupSpec(
                "Correct youngest/high confidence",
                lambda p: p.get("true_label") == p.get("pred_label") == max(labels, key=year),
                lambda p: pred_confidence(p, label_order),
            ),
            GroupSpec(
                "Correct oldest/high confidence",
                lambda p: p.get("true_label") == p.get("pred_label") == min(labels, key=year),
                lambda p: pred_confidence(p, label_order),
            ),
            GroupSpec(
                "Predicted younger than true",
                lambda p: year(p.get("pred_label")) > year(p.get("true_label")),
                lambda p: margin_confidence(p),
            ),
            GroupSpec(
                "Predicted older than true",
                lambda p: year(p.get("pred_label")) < year(p.get("true_label")),
                lambda p: margin_confidence(p),
            ),
            GroupSpec(
                "Wrong/high confidence",
                lambda p: p.get("true_label") != p.get("pred_label"),
                lambda p: pred_confidence(p, label_order),
            ),
        ]

    if analysis == "correct_by_class":
        return [
            GroupSpec(
                f"Correct {label} high",
                lambda p, label=label: p.get("true_label") == label and p.get("pred_label") == label,
                lambda p, label=label: confidence_for_label(p, label, label_order),
            )
            for label in labels
        ]

    if analysis == "false_by_pred":
        return [
            GroupSpec(
                f"False {label} high",
                lambda p, label=label: p.get("pred_label") == label and p.get("true_label") != label,
                lambda p, label=label: confidence_for_label(p, label, label_order),
            )
            for label in labels
        ]

    if analysis == "confusion_pairs":
        pairs = sorted({(p.get("true_label"), p.get("pred_label")) for p in preds if p.get("true_label") != p.get("pred_label")})
        return [
            GroupSpec(
                f"{true_label} → {pred_label}",
                lambda p, t=true_label, pr=pred_label: p.get("true_label") == t and p.get("pred_label") == pr,
                lambda p: pred_confidence(p, label_order),
            )
            for true_label, pred_label in pairs
        ]

    raise ValueError(f"Unknown analysis: {analysis}")


def default_analyses_for_target(target: str) -> List[str]:
    if target == "birthyear":
        return ["correct_by_class", "age_direction"]
    return ["correct_by_class", "false_by_pred"]


def plot_horizontal_bars(
    rows_by_group: Dict[str, List[Tuple[str, float]]],
    title: str,
    out_png: str,
    xlabel: str,
    max_groups_per_fig: int = 8,
) -> None:
    groups = [(g, rows) for g, rows in rows_by_group.items() if rows]
    if not groups:
        print(f"[WARN] No data for plot: {title}")
        return

    for chunk_start in range(0, len(groups), max_groups_per_fig):
        chunk = groups[chunk_start : chunk_start + max_groups_per_fig]
        n_groups = len(chunk)
        ncols = 2 if n_groups > 1 else 1
        nrows = math.ceil(n_groups / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 4.8 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for ax, (group_name, rows) in zip(axes_flat, chunk):
            labels = [r[0] for r in rows][::-1]
            values = [r[1] for r in rows][::-1]
            ax.barh(labels, values)
            ax.set_title(group_name)
            ax.set_xlabel(xlabel)
            ax.tick_params(axis="y", labelsize=9)

        for ax in axes_flat[len(chunk) :]:
            ax.axis("off")

        fig.suptitle(title, fontsize=15)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        if len(groups) <= max_groups_per_fig:
            chunk_png = out_png
        else:
            stem, ext = os.path.splitext(out_png)
            chunk_png = f"{stem}_part{chunk_start // max_groups_per_fig + 1}{ext}"
        fig.savefig(chunk_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved plot: {chunk_png}")


def write_ngram_csv(path: str, group_rows: Dict[str, List[Tuple[str, float]]], value_name: str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "rank", "ngram", value_name])
        for group, rows in group_rows.items():
            for i, (ngram, value) in enumerate(rows, 1):
                writer.writerow([group, i, ngram, value])
    print(f"[OK] Saved table: {path}")


def collect_counts_for_group(
    selected_preds: Sequence[dict],
    feeds: Dict[str, List[str]],
    n: int,
    args: argparse.Namespace,
) -> Counter:
    c: Counter = Counter()
    for p in selected_preds:
        cid = str(p.get("celebrity_id"))
        texts = feeds.get(cid)
        if not texts:
            continue
        c.update(
            get_ngrams_from_texts(
                texts,
                n=n,
                min_token_len=args.min_token_len,
                remove_stopwords=args.remove_stopwords,
                keep_hashtags=not args.drop_hashtags,
                max_tweets=args.max_tweets_per_celebrity,
            )
        )
    return c


def run_frequency(
    target: str,
    split: str,
    preds: Sequence[dict],
    feeds: Dict[str, List[str]],
    label_order: Sequence[str],
    args: argparse.Namespace,
    analysis: str,
    out_dir: str,
) -> None:
    specs = build_group_specs(target, preds, label_order, analysis)
    for n in args.ngrams:
        group_rows: Dict[str, List[Tuple[str, float]]] = {}
        for spec in specs:
            selected = select_top(preds, spec, args.top_docs)
            counts = collect_counts_for_group(selected, feeds, n, args)
            group_rows[spec.name] = [(k, float(v)) for k, v in counts.most_common(args.top_k)]

        suffix = "nostop" if args.remove_stopwords else "raw"
        base = f"{target}_{split}_{analysis}_freq_ngram{n}_{suffix}"
        write_ngram_csv(os.path.join(out_dir, f"{base}.csv"), group_rows, "count")
        plot_horizontal_bars(
            group_rows,
            title=f"{target} / {split} / {analysis}: top {args.top_k} frequent {n}-grams",
            out_png=os.path.join(out_dir, f"{base}.png"),
            xlabel="Count",
        )


def log_odds_rows(group_counts: Counter, rest_counts: Counter, top_k: int, alpha: float = 0.01) -> List[Tuple[str, float]]:
    vocab = set(group_counts) | set(rest_counts)
    total_g = sum(group_counts.values())
    total_r = sum(rest_counts.values())
    v = max(len(vocab), 1)
    scores = []
    for term in vocab:
        pg = (group_counts[term] + alpha) / (total_g + alpha * v)
        pr = (rest_counts[term] + alpha) / (total_r + alpha * v)
        score = math.log(pg) - math.log(pr)
        # Filter very rare one-off artifacts a bit, but still allow rare class-specific phrases.
        if group_counts[term] >= 2:
            scores.append((term, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [(term, float(score)) for term, score in scores[:top_k]]


def run_logodds(
    target: str,
    split: str,
    preds: Sequence[dict],
    feeds: Dict[str, List[str]],
    label_order: Sequence[str],
    args: argparse.Namespace,
    analysis: str,
    out_dir: str,
) -> None:
    specs = build_group_specs(target, preds, label_order, analysis)
    all_ids = {str(p.get("celebrity_id")) for p in preds if str(p.get("celebrity_id")) in feeds}

    for n in args.ngrams:
        group_rows: Dict[str, List[Tuple[str, float]]] = {}
        for spec in specs:
            selected = select_top(preds, spec, args.top_docs)
            group_ids = {str(p.get("celebrity_id")) for p in selected if str(p.get("celebrity_id")) in feeds}
            if not group_ids:
                group_rows[spec.name] = []
                continue

            rest_ids = list(all_ids - group_ids)
            if args.max_rest_docs and args.max_rest_docs > 0:
                rest_ids = rest_ids[: args.max_rest_docs]

            group_counts = Counter()
            for cid in group_ids:
                group_counts.update(
                    get_ngrams_from_texts(
                        feeds[cid], n, args.min_token_len, args.remove_stopwords,
                        not args.drop_hashtags, args.max_tweets_per_celebrity,
                    )
                )

            rest_counts = Counter()
            for cid in rest_ids:
                rest_counts.update(
                    get_ngrams_from_texts(
                        feeds[cid], n, args.min_token_len, args.remove_stopwords,
                        not args.drop_hashtags, args.max_tweets_per_celebrity,
                    )
                )

            group_rows[spec.name] = log_odds_rows(group_counts, rest_counts, args.top_k, args.logodds_alpha)

        suffix = "nostop" if args.remove_stopwords else "raw"
        base = f"{target}_{split}_{analysis}_logodds_ngram{n}_{suffix}"
        write_ngram_csv(os.path.join(out_dir, f"{base}.csv"), group_rows, "log_odds")
        plot_horizontal_bars(
            group_rows,
            title=f"{target} / {split} / {analysis}: overrepresented {n}-grams",
            out_png=os.path.join(out_dir, f"{base}.png"),
            xlabel="Log-odds vs. background",
        )


def add_age_direction(row: dict) -> str:
    try:
        true_y = int(row["true_label"])
        pred_y = int(row["pred_label"])
    except Exception:
        return "unknown"
    if pred_y == true_y:
        return "correct"
    if pred_y > true_y:
        return "predicted_younger"
    return "predicted_older"


def run_style(
    target: str,
    split: str,
    preds: Sequence[dict],
    feeds: Dict[str, List[str]],
    label_order: Sequence[str],
    args: argparse.Namespace,
    out_dir: str,
) -> None:
    rows = []
    pred_by_id = {str(p.get("celebrity_id")): p for p in preds}
    for cid, p in pred_by_id.items():
        texts = feeds.get(cid)
        if not texts:
            continue
        feats = style_features(texts)
        row = {
            "celebrity_id": cid,
            "true_label": p.get("true_label"),
            "pred_label": p.get("pred_label"),
            "correct": p.get("true_label") == p.get("pred_label"),
            "confidence": pred_confidence(p, label_order),
            "margin": margin_confidence(p),
        }
        if target == "birthyear":
            row["age_error_direction"] = add_age_direction(p)
        row.update(feats)
        rows.append(row)

    if not rows:
        print("[WARN] No style rows; check feed path and celebrity IDs.")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"{target}_{split}_style_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved table: {csv_path}")

    feature_cols = [
        "avg_tokens_per_tweet",
        "url_count",
        "mention_count",
        "hashtag_count",
        "emoji_count",
        "exclamation_count",
        "question_count",
        "uppercase_ratio",
        "love_word_count",
        "fan_word_count",
        "politics_word_count",
        "sports_word_count",
    ]

    # Keep plots readable: one figure per feature, grouped by predicted label.
    group_col = "age_error_direction" if target == "birthyear" else "pred_label"
    for feature in feature_cols:
        if feature not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        # pandas boxplot is enough and avoids seaborn dependency.
        df.boxplot(column=feature, by=group_col, ax=ax, grid=False, rot=30)
        ax.set_title(f"{target} / {split}: {feature} by {group_col}")
        ax.set_xlabel(group_col)
        ax.set_ylabel(feature)
        fig.suptitle("")
        fig.tight_layout()
        out_png = os.path.join(out_dir, f"{target}_{split}_style_{feature}_by_{group_col}.png")
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved plot: {out_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize prediction-conditioned n-gram and style signals for BERTweet outputs."
    )
    parser.add_argument("--target", required=True, choices=sorted(LABEL_ORDERS.keys()))
    parser.add_argument("--split", default="test", choices=["test", "val", "val_all"])
    parser.add_argument("--pred-path", default=None, help="Optional explicit prediction JSON path.")
    parser.add_argument("--feed-path", default=None, help="Optional explicit follower-feeds.ndjson path.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Default: plots/prediction_signals/<target>/<split>")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["frequency", "logodds"],
        choices=["frequency", "logodds", "style"],
    )
    parser.add_argument("--ngrams", nargs="+", type=int, default=[2, 3], help="n-gram sizes to plot.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of terms per group.")
    parser.add_argument(
        "--top-docs",
        type=int,
        default=20,
        help="Top confident docs per group. Use 0 to include all docs in a group.",
    )
    parser.add_argument("--max-rest-docs", type=int, default=0, help="Optional cap for log-odds background docs; 0 = all.")
    parser.add_argument("--max-tweets-per-celebrity", type=int, default=0, help="Optional tweet cap per celebrity; 0 = all.")
    parser.add_argument("--min-token-len", type=int, default=2)
    parser.add_argument("--remove-stopwords", action="store_true", help="Remove English stopwords for cleaner topical n-grams.")
    parser.add_argument("--drop-hashtags", action="store_true", help="Remove # marker and treat hashtags as plain words.")
    parser.add_argument("--logodds-alpha", type=float, default=0.01)
    parser.add_argument(
        "--analyses",
        nargs="+",
        default=None,
        choices=["correct_by_class", "false_by_pred", "confusion_pairs", "age_direction"],
        help="Group definitions. Default: correct_by_class+false_by_pred, or age_direction for birthyear.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = find_prediction_path(args.target, args.split, args.pred_path)
    feed_path = resolve_feed_path(args.split, args.feed_path)

    preds = read_json(pred_path)
    if not isinstance(preds, list):
        raise ValueError(f"Prediction file must contain a list of prediction rows: {pred_path}")

    label_order = infer_label_order(args.target, preds)
    out_dir = args.out_dir or os.path.join(plots_dir, "prediction_signals", args.target, args.split)
    ensure_dir(out_dir)

    ids = {str(p.get("celebrity_id")) for p in preds}
    print(f"[INFO] Prediction file: {pred_path}")
    print(f"[INFO] Feed file:       {feed_path}")
    print(f"[INFO] Output dir:      {out_dir}")
    print(f"[INFO] Target/split:    {args.target}/{args.split}")
    print(f"[INFO] Label order:     {label_order}")
    print(f"[INFO] Predictions:     {len(preds)}")

    feeds = load_feeds_for_ids(feed_path, ids=ids)
    print(f"[INFO] Loaded feeds:    {len(feeds)}")
    if not feeds:
        raise RuntimeError("No feeds loaded. Check split/feed path and celebrity IDs.")

    analyses = args.analyses or default_analyses_for_target(args.target)
    if args.target != "birthyear" and "age_direction" in analyses:
        raise ValueError("age_direction is only valid for --target birthyear")

    for analysis in analyses:
        if "frequency" in args.modes:
            run_frequency(args.target, args.split, preds, feeds, label_order, args, analysis, out_dir)
        if "logodds" in args.modes:
            run_logodds(args.target, args.split, preds, feeds, label_order, args, analysis, out_dir)

    if "style" in args.modes:
        run_style(args.target, args.split, preds, feeds, label_order, args, out_dir)

    print("[DONE] Prediction signal analysis finished.")


if __name__ == "__main__":
    main()
