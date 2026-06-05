import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


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
    train_label_path,
    test_label_path,
    supp_label_path,
    train_feeds_path,
    test_feeds_path,
    supp_feeds_path,
    hybrid_v4_output_dir,
)


OCCUPATION_ORDER = ["sports", "performer", "creator", "politics"]
GENDER_ORDER = ["male", "female"]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def iter_ndjson(path: str) -> Iterable[dict]:
    print(f"[INFO] Streaming: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid NDJSON in {path} at line {line_idx}: {e}"
                ) from e


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_text(text: str, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def get_id(row: dict) -> str:
    for key in ["celebrity_id", "id", "author_id"]:
        if key in row:
            return str(row[key])
    raise KeyError(f"No id key found in row keys={list(row.keys())}")


def load_labels(path: str) -> Dict[str, dict]:
    labels = {}
    for row in iter_ndjson(path):
        cid = get_id(row)
        labels[cid] = row
    return labels


def make_age_bin(birthyear) -> str:
    year = int(birthyear)

    if year < 1950:
        return "1940-1949"
    if year < 1960:
        return "1950-1959"
    if year < 1970:
        return "1960-1969"
    if year < 1980:
        return "1970-1979"
    if year < 1990:
        return "1980-1989"
    return "1990-1999"


def flatten_tweets(text_field: Any):
    """
    Handles both possible structures:

    1) Nested follower structure:
       text = [[tweet, tweet, ...], [tweet, tweet, ...], ...]

    2) Flat tweet structure:
       text = [tweet, tweet, tweet, ...]

    Returns:
        follower_tweet_lists: List[List[str]]
        structure_type: "nested_followers" or "flat_tweets" or "unknown"
    """
    if not isinstance(text_field, list):
        return [], "unknown"

    if len(text_field) == 0:
        return [], "empty"

    # nested: list of follower tweet lists
    if all(isinstance(x, list) for x in text_field):
        follower_lists = []

        for follower_block in text_field:
            tweets = [
                str(t)
                for t in follower_block
                if isinstance(t, str) and t.strip()
            ]
            follower_lists.append(tweets)

        return follower_lists, "nested_followers"

    # flat: list of tweets
    if all(isinstance(x, str) for x in text_field):
        tweets = [str(t) for t in text_field if str(t).strip()]
        return [tweets], "flat_tweets"

    return [], "unknown"


def text_length_stats(tweets: List[str]) -> dict:
    if not tweets:
        return {
            "tweet_char_mean": 0.0,
            "tweet_char_median": 0.0,
            "tweet_word_mean": 0.0,
            "tweet_word_median": 0.0,
            "hashtag_count": 0,
            "mention_count": 0,
            "url_count": 0,
            "retweet_count": 0,
        }

    char_lens = [len(t) for t in tweets]
    word_lens = [len(t.split()) for t in tweets]

    return {
        "tweet_char_mean": float(np.mean(char_lens)),
        "tweet_char_median": float(np.median(char_lens)),
        "tweet_word_mean": float(np.mean(word_lens)),
        "tweet_word_median": float(np.median(word_lens)),
        "hashtag_count": int(sum(t.count("#") for t in tweets)),
        "mention_count": int(sum(t.count("@") for t in tweets)),
        "url_count": int(sum(("http://" in t or "https://" in t or "t.co/" in t) for t in tweets)),
        "retweet_count": int(sum(t.strip().lower().startswith("rt ") for t in tweets)),
    }


def analyze_feeds(feeds_path: str, labels: Dict[str, dict], split_name: str) -> dict:
    per_celeb = []
    structure_counter = Counter()

    for row in iter_ndjson(feeds_path):
        cid = get_id(row)

        if cid not in labels:
            continue

        text_field = row.get("text", [])
        follower_lists, structure_type = flatten_tweets(text_field)
        structure_counter[structure_type] += 1

        all_tweets = [tweet for follower in follower_lists for tweet in follower]

        non_empty_followers = [f for f in follower_lists if len(f) > 0]

        follower_tweet_counts = [len(f) for f in follower_lists]
        non_empty_follower_tweet_counts = [len(f) for f in non_empty_followers]

        text_stats = text_length_stats(all_tweets)

        label_row = labels[cid]

        stat = {
            "celebrity_id": cid,
            "split": split_name,
            "structure_type": structure_type,
            "occupation": str(label_row["occupation"]),
            "gender": str(label_row["gender"]),
            "birthyear": int(label_row["birthyear"]),
            "age_bin": make_age_bin(label_row["birthyear"]),
            "num_followers_blocks": int(len(follower_lists)),
            "num_non_empty_followers": int(len(non_empty_followers)),
            "num_tweets_total": int(len(all_tweets)),
            "tweets_per_follower_mean": float(np.mean(follower_tweet_counts)) if follower_tweet_counts else 0.0,
            "tweets_per_follower_median": float(np.median(follower_tweet_counts)) if follower_tweet_counts else 0.0,
            "tweets_per_non_empty_follower_mean": float(np.mean(non_empty_follower_tweet_counts)) if non_empty_follower_tweet_counts else 0.0,
            "tweets_per_non_empty_follower_median": float(np.median(non_empty_follower_tweet_counts)) if non_empty_follower_tweet_counts else 0.0,
            "min_tweets_per_follower": int(min(follower_tweet_counts)) if follower_tweet_counts else 0,
            "max_tweets_per_follower": int(max(follower_tweet_counts)) if follower_tweet_counts else 0,
            **text_stats,
        }

        per_celeb.append(stat)

    return {
        "split": split_name,
        "structure_counter": dict(structure_counter),
        "per_celeb": per_celeb,
    }


def summarize_numeric(values: List[float]) -> dict:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p10": None,
            "p90": None,
        }

    arr = np.array(values, dtype=float)

    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def summarize_by_group(per_celeb: List[dict], group_key: str) -> dict:
    metrics = [
        "num_followers_blocks",
        "num_non_empty_followers",
        "num_tweets_total",
        "tweets_per_follower_mean",
        "tweets_per_non_empty_follower_mean",
        "tweet_char_mean",
        "tweet_word_mean",
        "hashtag_count",
        "mention_count",
        "url_count",
        "retweet_count",
    ]

    grouped = defaultdict(lambda: defaultdict(list))

    for row in per_celeb:
        group = str(row[group_key])

        for metric in metrics:
            grouped[group][metric].append(row[metric])

    summary = {}

    for group, metric_values in grouped.items():
        summary[group] = {
            metric: summarize_numeric(values)
            for metric, values in metric_values.items()
        }

    return summary


def plot_hist(values: List[float], title: str, out_path: str, bins: int = 40):
    if not values:
        return

    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_box_by_group(
    per_celeb: List[dict],
    group_key: str,
    metric: str,
    title: str,
    out_path: str,
    order: List[str] = None,
):
    values_by_group = defaultdict(list)

    for row in per_celeb:
        values_by_group[str(row[group_key])].append(row[metric])

    if order is None:
        order = sorted(values_by_group.keys())

    data = []
    labels = []

    for label in order:
        values = values_by_group.get(label, [])
        if values:
            labels.append(label)
            data.append(values)

    if not data:
        return

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def markdown_summary_table(title: str, summary: dict, metric: str) -> str:
    lines = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append(f"Metric: `{metric}`")
    lines.append("")
    lines.append("| group | n | mean | median | min | max | p10 | p90 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for group in sorted(summary.keys()):
        s = summary[group][metric]
        lines.append(
            f"| {group} | {s['n']} | "
            f"{s['mean']:.2f} | {s['median']:.2f} | "
            f"{s['min']:.2f} | {s['max']:.2f} | "
            f"{s['p10']:.2f} | {s['p90']:.2f} |"
        )

    lines.append("")
    return "\n".join(lines)


def build_report(results: dict, summary: dict) -> str:
    lines = []

    lines.append("# HybridV4 Tweet and Follower Statistics")
    lines.append("")

    lines.append("## Raw feed structure")
    lines.append("")
    lines.append("| split | structure_type | count |")
    lines.append("|---|---|---:|")

    for split_name, result in results.items():
        for structure_type, count in result["structure_counter"].items():
            lines.append(f"| {split_name} | {structure_type} | {count} |")

    lines.append("")

    for split_name, split_summary in summary.items():
        lines.append(f"## {split_name}")
        lines.append("")

        lines.append(markdown_summary_table(
            "Tweets per occupation",
            split_summary["by_occupation"],
            "num_tweets_total",
        ))

        lines.append(markdown_summary_table(
            "Followers per occupation",
            split_summary["by_occupation"],
            "num_non_empty_followers",
        ))

        lines.append(markdown_summary_table(
            "Tweets per gender",
            split_summary["by_gender"],
            "num_tweets_total",
        ))

        lines.append(markdown_summary_table(
            "Tweets per age bin",
            split_summary["by_age_bin"],
            "num_tweets_total",
        ))

        lines.append(markdown_summary_table(
            "Tweet word length per occupation",
            split_summary["by_occupation"],
            "tweet_word_mean",
        ))

        lines.append(markdown_summary_table(
            "Retweets per occupation",
            split_summary["by_occupation"],
            "retweet_count",
        ))

    lines.append("## Interpretation hints")
    lines.append("")
    lines.append("- If `structure_type = nested_followers`, follower dropout is feasible by dropping complete inner follower blocks.")
    lines.append("- If `structure_type = flat_tweets`, only tweet dropout is directly feasible unless follower metadata exists elsewhere.")
    lines.append("- If some classes have systematically fewer tweets or fewer non-empty follower blocks, validation and model performance may be affected by coverage rather than only language.")
    lines.append("- If occupation classes differ strongly in retweets, hashtags, mentions, or URLs, the model may learn community/activity patterns in addition to semantic content.")
    lines.append("")

    return "\n".join(lines)


def run():
    out_dir = os.path.join(hybrid_v4_output_dir, "data_audit", "tweet_follower_statistics")
    plots_dir = os.path.join(out_dir, "plots")

    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    labels_by_split = {
        "train": load_labels(train_label_path),
        "test": load_labels(test_label_path),
        "supplement": load_labels(supp_label_path),
    }

    feeds_by_split = {
        "train": train_feeds_path,
        "test": test_feeds_path,
        "supplement": supp_feeds_path,
    }

    results = {}

    for split_name in ["train", "test", "supplement"]:
        results[split_name] = analyze_feeds(
            feeds_path=feeds_by_split[split_name],
            labels=labels_by_split[split_name],
            split_name=split_name,
        )

    summary = {}

    for split_name, result in results.items():
        per_celeb = result["per_celeb"]

        summary[split_name] = {
            "by_occupation": summarize_by_group(per_celeb, "occupation"),
            "by_gender": summarize_by_group(per_celeb, "gender"),
            "by_age_bin": summarize_by_group(per_celeb, "age_bin"),
            "overall": {
                "num_tweets_total": summarize_numeric([r["num_tweets_total"] for r in per_celeb]),
                "num_non_empty_followers": summarize_numeric([r["num_non_empty_followers"] for r in per_celeb]),
                "tweet_word_mean": summarize_numeric([r["tweet_word_mean"] for r in per_celeb]),
            },
        }

        plot_hist(
            [r["num_tweets_total"] for r in per_celeb],
            f"{split_name}: tweets per celebrity",
            os.path.join(plots_dir, f"{split_name}_tweets_per_celebrity_hist.png"),
            bins=50,
        )

        plot_box_by_group(
            per_celeb,
            group_key="occupation",
            metric="num_tweets_total",
            title=f"{split_name}: tweets per celebrity by occupation",
            out_path=os.path.join(plots_dir, f"{split_name}_tweets_by_occupation.png"),
            order=OCCUPATION_ORDER,
        )

        plot_box_by_group(
            per_celeb,
            group_key="gender",
            metric="num_tweets_total",
            title=f"{split_name}: tweets per celebrity by gender",
            out_path=os.path.join(plots_dir, f"{split_name}_tweets_by_gender.png"),
            order=GENDER_ORDER,
        )

        plot_box_by_group(
            per_celeb,
            group_key="occupation",
            metric="num_non_empty_followers",
            title=f"{split_name}: non-empty followers by occupation",
            out_path=os.path.join(plots_dir, f"{split_name}_followers_by_occupation.png"),
            order=OCCUPATION_ORDER,
        )

        plot_box_by_group(
            per_celeb,
            group_key="occupation",
            metric="retweet_count",
            title=f"{split_name}: retweets by occupation",
            out_path=os.path.join(plots_dir, f"{split_name}_retweets_by_occupation.png"),
            order=OCCUPATION_ORDER,
        )

    save_json(results, os.path.join(out_dir, "tweet_follower_statistics_raw.json"))
    save_json(summary, os.path.join(out_dir, "tweet_follower_statistics_summary.json"))

    report = build_report(results, summary)
    save_text(report, os.path.join(out_dir, "tweet_follower_statistics_report.md"))

    print(f"[OK] Saved report: {os.path.join(out_dir, 'tweet_follower_statistics_report.md')}")
    print(f"[OK] Saved summary: {os.path.join(out_dir, 'tweet_follower_statistics_summary.json')}")
    print(f"[OK] Saved plots: {plots_dir}")


def main():
    run()


if __name__ == "__main__":
    main()