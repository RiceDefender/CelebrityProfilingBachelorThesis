import argparse
import csv
import json
import os
import re
import sys
import html
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from _constants import hybrid_v4_fusion_metrics_dir


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_#@']+")


OCCUPATION_LEXICONS = {
    "sports": [
        "game", "match", "team", "season", "league", "coach", "player",
        "win", "goal", "training", "sport", "football", "basketball",
        "tennis", "race", "championship", "tournament"
    ],
    "performer": [
        "show", "stage", "music", "song", "album", "concert", "tour",
        "film", "movie", "actor", "actress", "singer", "dance",
        "performance", "theatre", "tv", "series"
    ],
    "creator": [
        "book", "writing", "writer", "author", "art", "artist", "design",
        "photo", "photography", "blog", "podcast", "youtube", "video",
        "content", "creator", "paint", "illustration"
    ],
    "politics": [
        "government", "minister", "president", "policy", "vote", "election",
        "campaign", "parliament", "senate", "law", "party", "democracy",
        "political", "public", "rights", "justice"
    ],
}


GENDER_LEXICONS = {
    "male": [
        "father", "dad", "husband", "boy", "man", "brother", "son",
        "king", "sir", "mr", "he", "his", "him"
    ],
    "female": [
        "mother", "mom", "wife", "girl", "woman", "sister", "daughter",
        "queen", "lady", "mrs", "ms", "she", "her"
    ],
}


def load_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV file: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_id(value) -> str:
    return str(value).strip()


def np_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def count_lexicon(tokens: List[str], lexicon: List[str]) -> int:
    token_counter = Counter(tokens)
    return sum(token_counter[word.lower()] for word in lexicon)


def extract_texts_from_any_json(value: Any) -> List[str]:
    """
    Robuster Fallback:
    Sucht rekursiv nach Textfeldern in beliebigen JSON-Strukturen.
    """
    texts = []

    if isinstance(value, dict):
        for key, val in value.items():
            key_lower = str(key).lower()

            if key_lower in {
                "text",
                "tweet",
                "tweet_text",
                "full_text",
                "content",
                "body",
                "document",
                "documents",
            }:
                if isinstance(val, str):
                    clean = html.unescape(val).strip()
                    if clean:
                        texts.append(clean)
                else:
                    texts.extend(extract_texts_from_any_json(val))
            else:
                texts.extend(extract_texts_from_any_json(val))

    elif isinstance(value, list):
        for item in value:
            texts.extend(extract_texts_from_any_json(item))

    elif isinstance(value, str):
        clean = html.unescape(value).strip()

        # Nicht jeden String nehmen, sonst landen IDs/Labels drin.
        # Nur strings, die tweetartig aussehen oder genug lang sind.
        if len(clean.split()) >= 4 or "@" in clean or "#" in clean or "http" in clean:
            texts.append(clean)

    return texts


def guess_id_from_json(row: dict) -> str:
    """
    Versucht verschiedene mögliche ID-Felder aus PAN20 zu erkennen.
    """
    candidate_keys = [
        "celebrity_id",
        "id",
        "user_id",
        "author_id",
        "profile_id",
        "twitter_id",
        "celebrity",
        "celeb_id",
    ]

    for key in candidate_keys:
        if key in row and row[key] not in [None, ""]:
            return normalize_id(row[key])

    # Falls die ID verschachtelt ist
    for key, value in row.items():
        if isinstance(value, dict):
            nested = guess_id_from_json(value)
            if nested:
                return nested

    return ""


def load_pan_ndjson(path: str) -> Dict[str, List[str]]:
    """
    Lädt PAN20 follower-feeds.ndjson und mapped id -> Tweets.

    Erwartetes Schema:
    {
      "id": 10186,
      "text": [
        [
          "tweet 1",
          "tweet 2",
          ...
        ]
      ]
    }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing PAN NDJSON file: {path}")

    id_to_tweets = {}
    total_lines = 0
    parsed_lines = 0
    skipped_no_id = 0
    skipped_no_text = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
                parsed_lines += 1
            except json.JSONDecodeError:
                continue

            cid = str(obj.get("id", "")).strip()
            if not cid:
                skipped_no_id += 1
                continue

            raw_text = obj.get("text", [])
            tweets = []

            # PAN20 schema: text is usually a nested list: [[tweet, tweet, ...]]
            if isinstance(raw_text, list):
                for item in raw_text:
                    if isinstance(item, list):
                        for tweet in item:
                            if isinstance(tweet, str) and tweet.strip():
                                tweets.append(html.unescape(tweet.strip()))
                    elif isinstance(item, str) and item.strip():
                        tweets.append(html.unescape(item.strip()))

            elif isinstance(raw_text, str) and raw_text.strip():
                tweets.append(html.unescape(raw_text.strip()))

            # Remove duplicate tweets, keep order
            tweets = list(dict.fromkeys(tweets))

            if not tweets:
                skipped_no_text += 1
                continue

            id_to_tweets[cid] = tweets

    print("\n========== PAN NDJSON load ==========")
    print(f"File:             {path}")
    print(f"Total lines:      {total_lines}")
    print(f"Parsed lines:     {parsed_lines}")
    print(f"IDs loaded:       {len(id_to_tweets)}")
    print(f"Skipped no ID:    {skipped_no_id}")
    print(f"Skipped no text:  {skipped_no_text}")

    return id_to_tweets


def basic_text_stats(tweets: List[str], target: str) -> dict:
    all_text = "\n".join(tweets)
    tokens = tokenize(all_text)

    num_tweets = len(tweets)
    num_chars = len(all_text)
    num_tokens = len(tokens)

    tweet_lengths = [len(t) for t in tweets]
    token_lengths = [len(tokenize(t)) for t in tweets]

    hashtags = HASHTAG_RE.findall(all_text)
    mentions = MENTION_RE.findall(all_text)
    urls = URL_RE.findall(all_text)

    stats = {
        "pan_num_tweets": num_tweets,
        "pan_num_chars": num_chars,
        "pan_num_tokens": num_tokens,
        "pan_avg_chars_per_tweet": round(sum(tweet_lengths) / max(num_tweets, 1), 2),
        "pan_avg_tokens_per_tweet": round(sum(token_lengths) / max(num_tweets, 1), 2),
        "pan_num_hashtags": len(hashtags),
        "pan_num_mentions": len(mentions),
        "pan_num_urls": len(urls),
        "pan_url_ratio": round(len(urls) / max(num_tweets, 1), 4),
        "pan_hashtag_ratio": round(len(hashtags) / max(num_tweets, 1), 4),
        "pan_mention_ratio": round(len(mentions) / max(num_tweets, 1), 4),
    }

    if target == "occupation":
        for label, lexicon in OCCUPATION_LEXICONS.items():
            stats[f"lex_{label}"] = count_lexicon(tokens, lexicon)

    if target == "gender":
        for label, lexicon in GENDER_LEXICONS.items():
            stats[f"lex_{label}"] = count_lexicon(tokens, lexicon)

    top_tokens = [
        token for token, count in Counter(tokens).most_common(50)
        if len(token) > 2 and not token.startswith("@")
    ]

    top_hashtags = [tag.lower() for tag, _ in Counter(hashtags).most_common(15)]
    top_mentions = [mention.lower() for mention, _ in Counter(mentions).most_common(15)]

    stats["pan_top_tokens"] = " ".join(top_tokens[:25])
    stats["pan_top_hashtags"] = " ".join(top_hashtags[:10])
    stats["pan_top_mentions"] = " ".join(top_mentions[:10])

    return stats


def select_representative_tweets(
    tweets: List[str],
    true_label: str,
    pred_label: str,
    target: str,
    max_examples: int = 5,
) -> List[str]:
    if not tweets:
        return []

    if target == "occupation":
        true_words = OCCUPATION_LEXICONS.get(true_label, [])
        pred_words = OCCUPATION_LEXICONS.get(pred_label, [])
    elif target == "gender":
        true_words = GENDER_LEXICONS.get(true_label, [])
        pred_words = GENDER_LEXICONS.get(pred_label, [])
    else:
        true_words = []
        pred_words = []

    true_words = set(w.lower() for w in true_words)
    pred_words = set(w.lower() for w in pred_words)

    scored = []

    for tweet in tweets:
        tokens = set(tokenize(tweet))
        true_hits = len(tokens & true_words)
        pred_hits = len(tokens & pred_words)

        score = true_hits + pred_hits
        score += min(len(HASHTAG_RE.findall(tweet)), 2) * 0.25
        score += min(len(URL_RE.findall(tweet)), 1) * 0.25
        score += min(len(MENTION_RE.findall(tweet)), 2) * 0.10

        scored.append((score, true_hits, pred_hits, len(tweet), tweet))

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)

    selected = []
    for _, _, _, _, tweet in scored:
        clean = " ".join(tweet.split())
        if clean not in selected:
            selected.append(clean)
        if len(selected) >= max_examples:
            break

    return selected


def get_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in [None, ""]:
            return default
        return float(value)
    except ValueError:
        return default


def enrich_errors_with_pan_ndjson(
    error_rows: List[dict],
    pan_ndjson: str,
    target: str,
    max_tweets_per_case: int,
) -> Tuple[List[dict], dict]:
    id_to_tweets = load_pan_ndjson(pan_ndjson)

    enriched = []
    missing_ids = []

    aggregate_by_pair = defaultdict(lambda: {
        "count": 0,
        "num_tweets": [],
        "num_tokens": [],
        "num_urls": [],
        "num_hashtags": [],
        "num_mentions": [],
        "confidences": [],
        "margins": [],
    })

    for row in error_rows:
        cid = normalize_id(row.get("celebrity_id"))
        tweets = id_to_tweets.get(cid)

        out = dict(row)
        out["pan_found"] = tweets is not None
        out["pan_source"] = pan_ndjson if tweets is not None else ""

        if not tweets:
            missing_ids.append(cid)
            enriched.append(out)
            continue

        stats = basic_text_stats(tweets, target=target)

        true_label = row.get("true_label")
        pred_label = row.get("fusion_pred_label")

        examples = select_representative_tweets(
            tweets=tweets,
            true_label=true_label,
            pred_label=pred_label,
            target=target,
            max_examples=max_tweets_per_case,
        )

        out.update(stats)

        for i in range(max_tweets_per_case):
            out[f"example_tweet_{i + 1}"] = examples[i] if i < len(examples) else ""

        enriched.append(out)

        pair_key = f"{true_label}->{pred_label}"
        agg = aggregate_by_pair[pair_key]
        agg["count"] += 1
        agg["num_tweets"].append(stats["pan_num_tweets"])
        agg["num_tokens"].append(stats["pan_num_tokens"])
        agg["num_urls"].append(stats["pan_num_urls"])
        agg["num_hashtags"].append(stats["pan_num_hashtags"])
        agg["num_mentions"].append(stats["pan_num_mentions"])
        agg["confidences"].append(get_float(row, "confidence", 0.0))
        agg["margins"].append(get_float(row, "margin", 0.0))

    summary_pairs = []

    for pair, values in aggregate_by_pair.items():
        summary_pairs.append({
            "mistake_pair": pair,
            "count": values["count"],
            "avg_num_tweets": round(np_mean(values["num_tweets"]), 2),
            "avg_num_tokens": round(np_mean(values["num_tokens"]), 2),
            "avg_num_urls": round(np_mean(values["num_urls"]), 2),
            "avg_num_hashtags": round(np_mean(values["num_hashtags"]), 2),
            "avg_num_mentions": round(np_mean(values["num_mentions"]), 2),
            "avg_confidence": round(np_mean(values["confidences"]), 4),
            "avg_margin": round(np_mean(values["margins"]), 4),
        })

    summary_pairs.sort(key=lambda x: x["count"], reverse=True)

    summary = {
        "target": target,
        "pan_ndjson": pan_ndjson,
        "num_error_rows": len(error_rows),
        "num_enriched_rows": len(enriched),
        "num_pan_ids_loaded": len(id_to_tweets),
        "num_missing_pan_ids": len(missing_ids),
        "missing_ids": missing_ids[:100],
        "mistake_pair_summary": summary_pairs,
    }

    return enriched, summary


def write_markdown_report(
    enriched_rows: List[dict],
    summary: dict,
    path: str,
    target: str,
    max_cases: int = 20,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    lines = []
    lines.append(f"# PAN Error Deep Dive: {target}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Target: `{target}`")
    lines.append(f"- PAN source: `{summary.get('pan_ndjson')}`")
    lines.append(f"- Error rows: `{summary['num_error_rows']}`")
    lines.append(f"- Enriched rows: `{summary['num_enriched_rows']}`")
    lines.append(f"- PAN IDs loaded: `{summary['num_pan_ids_loaded']}`")
    lines.append(f"- Missing PAN IDs: `{summary['num_missing_pan_ids']}`")
    lines.append("")

    lines.append("## Mistake-pair overview")
    lines.append("")
    lines.append("| Mistake pair | Count | Avg confidence | Avg margin | Avg tweets | Avg URLs | Avg hashtags | Avg mentions |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for item in summary["mistake_pair_summary"]:
        lines.append(
            f"| {item['mistake_pair']} | {item['count']} | "
            f"{item['avg_confidence']:.4f} | {item['avg_margin']:.4f} | "
            f"{item['avg_num_tweets']:.2f} | "
            f"{item['avg_num_urls']:.2f} | {item['avg_num_hashtags']:.2f} | "
            f"{item['avg_num_mentions']:.2f} |"
        )

    lines.append("")
    lines.append("## Example cases")
    lines.append("")

    def confidence_value(row):
        return get_float(row, "confidence", 0.0)

    rows_sorted = sorted(enriched_rows, key=confidence_value, reverse=True)

    for row in rows_sorted[:max_cases]:
        cid = row.get("celebrity_id")
        true_label = row.get("true_label")
        pred_label = row.get("fusion_pred_label")
        conf = row.get("confidence")
        margin = row.get("margin")

        lines.append(f"### ID {cid}: {true_label} → {pred_label}")
        lines.append("")
        lines.append(f"- Confidence: `{conf}`")
        lines.append(f"- Margin: `{margin}`")
        lines.append(f"- Agreement type: `{row.get('agreement_type')}`")
        lines.append(f"- V3: `{row.get('bertweet_v3_pred_label')}`")
        lines.append(f"- V3.4: `{row.get('bertweet_v34_pred_label')}`")
        lines.append(f"- Sparse: `{row.get('sparse_feature_pred_label')}`")
        lines.append(f"- PAN found: `{row.get('pan_found')}`")
        lines.append(f"- Tweets/text segments: `{row.get('pan_num_tweets')}`")
        lines.append(f"- Tokens: `{row.get('pan_num_tokens')}`")
        lines.append(f"- URLs: `{row.get('pan_num_urls')}`")
        lines.append(f"- Hashtags: `{row.get('pan_num_hashtags')}`")
        lines.append(f"- Mentions: `{row.get('pan_num_mentions')}`")
        lines.append(f"- Top tokens: `{row.get('pan_top_tokens')}`")
        lines.append(f"- Top hashtags: `{row.get('pan_top_hashtags')}`")
        lines.append("")

        example_cols = [
            key for key in row.keys()
            if key.startswith("example_tweet_") and row.get(key)
        ]

        if example_cols:
            lines.append("Representative tweets:")
            lines.append("")
            for key in example_cols:
                tweet = row[key]
                lines.append(f"> {tweet}")
                lines.append("")

        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Enrich HybridV4 error analysis CSV files with original PAN20 follower-feeds.ndjson data."
    )

    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear"],
        required=True,
    )
    parser.add_argument(
        "--errors-csv",
        required=True,
        help="Path to wrong_rows.csv, high_conf_wrong_top_30.csv, or low_margin_wrong_top_30.csv.",
    )
    parser.add_argument(
        "--pan-ndjson",
        required=True,
        help="Path to PAN20 follower-feeds.ndjson.",
    )
    parser.add_argument(
        "--out-name",
        default=None,
        help="Optional output name prefix. Defaults to input CSV filename.",
    )
    parser.add_argument(
        "--max-tweets-per-case",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-md-cases",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    error_rows = load_csv(args.errors_csv)

    enriched, summary = enrich_errors_with_pan_ndjson(
        error_rows=error_rows,
        pan_ndjson=args.pan_ndjson,
        target=args.target,
        max_tweets_per_case=args.max_tweets_per_case,
    )

    base_name = args.out_name
    if base_name is None:
        base_name = os.path.splitext(os.path.basename(args.errors_csv))[0]

    out_dir = os.path.join(
        hybrid_v4_fusion_metrics_dir,
        "pan_error_deep_dive",
        args.target,
    )

    enriched_csv_path = os.path.join(out_dir, f"{base_name}_pan_enriched.csv")
    summary_json_path = os.path.join(out_dir, f"{base_name}_pan_summary.json")
    report_md_path = os.path.join(out_dir, f"{base_name}_pan_report.md")

    save_csv(enriched, enriched_csv_path)
    save_json(summary, summary_json_path)
    write_markdown_report(
        enriched_rows=enriched,
        summary=summary,
        path=report_md_path,
        target=args.target,
        max_cases=args.max_md_cases,
    )

    print("\n========== PAN Error Deep Dive ==========")
    print(f"Target:             {args.target}")
    print(f"Input errors:       {args.errors_csv}")
    print(f"PAN NDJSON:         {args.pan_ndjson}")
    print(f"Error rows:         {summary['num_error_rows']}")
    print(f"PAN IDs loaded:     {summary['num_pan_ids_loaded']}")
    print(f"Missing PAN IDs:    {summary['num_missing_pan_ids']}")

    print("\n========== Top mistake pairs ==========")
    for item in summary["mistake_pair_summary"][:10]:
        print(
            f"{item['mistake_pair']:25s} "
            f"count={item['count']:3d} "
            f"avg_conf={item['avg_confidence']:.4f} "
            f"avg_margin={item['avg_margin']:.4f}"
        )

    print("\n========== Saved files ==========")
    print(f"[OK] Enriched CSV: {enriched_csv_path}")
    print(f"[OK] Summary JSON: {summary_json_path}")
    print(f"[OK] Markdown:     {report_md_path}")


if __name__ == "__main__":
    main()