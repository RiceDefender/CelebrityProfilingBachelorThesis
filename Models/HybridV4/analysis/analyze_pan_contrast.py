import argparse
import json
import os
import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


TOKEN_RE = re.compile(r"[#@]?[A-Za-z0-9_]{2,}", re.UNICODE)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pan_ndjson(path: str) -> Dict[str, dict]:
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def flatten_text(text_field) -> List[str]:
    """
    PAN follower-feeds text is usually a list of lists.
    This function safely flattens it.
    """
    tweets = []

    if isinstance(text_field, str):
        return [text_field]

    if isinstance(text_field, list):
        for item in text_field:
            if isinstance(item, str):
                tweets.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, str):
                        tweets.append(sub)

    return tweets


def tokenize(tweets: List[str], keep_hashtags: bool = True, keep_mentions: bool = False) -> List[str]:
    tokens = []

    for tweet in tweets:
        for tok in TOKEN_RE.findall(tweet.lower()):
            if tok.startswith("@") and not keep_mentions:
                continue
            if tok.startswith("#") and not keep_hashtags:
                tok = tok[1:]
            tokens.append(tok)

    return tokens


def prediction_label(row: dict) -> str:
    for key in ["fusion_pred_label", "pred_label", "prediction"]:
        if key in row:
            return row[key]
    raise KeyError("No prediction label field found.")


def true_label(row: dict) -> str:
    for key in ["true_label", "gold_label", "label"]:
        if key in row:
            return row[key]
    raise KeyError("No true label field found.")


def celebrity_id(row: dict) -> str:
    for key in ["celebrity_id", "id"]:
        if key in row:
            return str(row[key])
    raise KeyError("No celebrity id field found.")


def safe_log_ratio(a: int, b: int, total_a: int, total_b: int, vocab_size: int) -> float:
    """
    Smoothed log ratio:
    log( P(token|group_a) / P(token|group_b) )
    """
    pa = (a + 1.0) / (total_a + vocab_size)
    pb = (b + 1.0) / (total_b + vocab_size)
    return math.log(pa / pb)


def build_group_counters(
    predictions: List[dict],
    pan_rows: Dict[str, dict],
    max_tweets_per_author: int = 5000,
):
    group_token_counts = defaultdict(Counter)
    group_doc_counts = defaultdict(Counter)
    group_author_counts = Counter()

    for pred in predictions:
        cid = celebrity_id(pred)
        if cid not in pan_rows:
            continue

        gold = true_label(pred)
        pred_label = prediction_label(pred)
        correct = gold == pred_label

        if correct:
            group = f"correct_{gold}"
        else:
            group = f"{gold}_to_{pred_label}"

        tweets = flatten_text(pan_rows[cid].get("text", []))

        if max_tweets_per_author and len(tweets) > max_tweets_per_author:
            tweets = tweets[:max_tweets_per_author]

        tokens = tokenize(tweets)
        token_counts = Counter(tokens)
        unique_tokens = set(token_counts.keys())

        group_token_counts[group].update(token_counts)
        group_doc_counts[group].update(unique_tokens)
        group_author_counts[group] += 1

    return group_token_counts, group_doc_counts, group_author_counts


def contrast_groups(
    group_a: str,
    group_b: str,
    group_token_counts: Dict[str, Counter],
    group_doc_counts: Dict[str, Counter],
    group_author_counts: Counter,
    min_count: int = 20,
    top_k: int = 50,
):
    counter_a = group_token_counts[group_a]
    counter_b = group_token_counts[group_b]

    total_a = sum(counter_a.values())
    total_b = sum(counter_b.values())

    vocab = set(counter_a.keys()) | set(counter_b.keys())
    vocab_size = len(vocab)

    rows = []

    for tok in vocab:
        count_a = counter_a[tok]
        count_b = counter_b[tok]

        if count_a < min_count:
            continue

        doc_a = group_doc_counts[group_a][tok]
        doc_b = group_doc_counts[group_b][tok]

        log_ratio = safe_log_ratio(count_a, count_b, total_a, total_b, vocab_size)

        rows.append({
            "token": tok,
            "count_a": count_a,
            "count_b": count_b,
            "doc_count_a": doc_a,
            "doc_count_b": doc_b,
            "group_a_authors": group_author_counts[group_a],
            "group_b_authors": group_author_counts[group_b],
            "log_ratio_a_over_b": log_ratio,
        })

    rows.sort(key=lambda x: x["log_ratio_a_over_b"], reverse=True)
    return rows[:top_k]


def write_markdown_report(
    output_path: str,
    target: str,
    group_token_counts,
    group_doc_counts,
    group_author_counts,
    comparisons: List[Tuple[str, str]],
    min_count: int,
    top_k: int,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# PAN Token Contrast Analysis: {target}\n\n")

        f.write("## Groups\n\n")
        f.write("| Group | Authors | Tokens |\n")
        f.write("|---|---:|---:|\n")
        for group, n_authors in group_author_counts.most_common():
            f.write(f"| `{group}` | {n_authors} | {sum(group_token_counts[group].values())} |\n")

        f.write("\n")

        for group_a, group_b in comparisons:
            if group_a not in group_author_counts:
                f.write(f"\n## {group_a} vs {group_b}\n\n")
                f.write(f"`{group_a}` not found.\n")
                continue

            if group_b not in group_author_counts:
                f.write(f"\n## {group_a} vs {group_b}\n\n")
                f.write(f"`{group_b}` not found.\n")
                continue

            rows = contrast_groups(
                group_a=group_a,
                group_b=group_b,
                group_token_counts=group_token_counts,
                group_doc_counts=group_doc_counts,
                group_author_counts=group_author_counts,
                min_count=min_count,
                top_k=top_k,
            )

            f.write(f"\n## Tokens overrepresented in `{group_a}` compared to `{group_b}`\n\n")
            f.write("| Token | Count A | Count B | Docs A | Docs B | Log ratio |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")

            for row in rows:
                f.write(
                    f"| `{row['token']}` "
                    f"| {row['count_a']} "
                    f"| {row['count_b']} "
                    f"| {row['doc_count_a']} "
                    f"| {row['doc_count_b']} "
                    f"| {row['log_ratio_a_over_b']:.4f} |\n"
                )


def main():
    parser = argparse.ArgumentParser(
        description="PAN token contrast analysis for HybridV4 prediction errors."
    )
    parser.add_argument("--target", required=True, choices=["occupation", "gender", "birthyear"])
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--pan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tweets-per-author", type=int, default=5000)

    args = parser.parse_args()

    predictions = load_json(args.predictions)
    pan_rows = load_pan_ndjson(args.pan)

    group_token_counts, group_doc_counts, group_author_counts = build_group_counters(
        predictions=predictions,
        pan_rows=pan_rows,
        max_tweets_per_author=args.max_tweets_per_author,
    )

    if args.target == "occupation":
        comparisons = [
            ("creator_to_performer", "correct_creator"),
            ("correct_creator", "creator_to_performer"),
            ("sports_to_performer", "correct_sports"),
            ("politics_to_performer", "correct_politics"),
            ("performer_to_creator", "correct_performer"),
        ]
    elif args.target == "gender":
        comparisons = [
            ("female_to_male", "correct_female"),
            ("male_to_female", "correct_male"),
            ("correct_female", "female_to_male"),
            ("correct_male", "male_to_female"),
        ]
    else:
        comparisons = []

    write_markdown_report(
        output_path=args.output,
        target=args.target,
        group_token_counts=group_token_counts,
        group_doc_counts=group_doc_counts,
        group_author_counts=group_author_counts,
        comparisons=comparisons,
        min_count=args.min_count,
        top_k=args.top_k,
    )

    print(f"[OK] Saved PAN contrast report: {args.output}")


if __name__ == "__main__":
    main()