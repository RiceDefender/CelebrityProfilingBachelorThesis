import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

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

RANDOM_SEED = 42


SAMPLING_CONFIGS = [
    {
        "name": "raw_all",
        "mode": "raw_all",
    },
    {
        "name": "pan_style_10_followers_20_tweets",
        "mode": "fixed_followers_fixed_tweets",
        "max_followers": 10,
        "max_tweets_per_follower": 20,
    },
    {
        "name": "balanced_20_followers_50_tweets",
        "mode": "fixed_followers_fixed_tweets",
        "max_followers": 20,
        "max_tweets_per_follower": 50,
    },
    {
        "name": "balanced_30_followers_30_tweets",
        "mode": "fixed_followers_fixed_tweets",
        "max_followers": 30,
        "max_tweets_per_follower": 30,
    },
    {
        "name": "tweet_cap_5000",
        "mode": "tweet_cap",
        "max_total_tweets": 5000,
    },
    {
        "name": "tweet_cap_10000",
        "mode": "tweet_cap",
        "max_total_tweets": 10000,
    },
    {
        "name": "follower_dropout_20",
        "mode": "follower_dropout",
        "dropout_rate": 0.20,
    },
    {
        "name": "follower_dropout_40",
        "mode": "follower_dropout",
        "dropout_rate": 0.40,
    },
    {
        "name": "tweet_dropout_30",
        "mode": "tweet_dropout",
        "dropout_rate": 0.30,
    },
]


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


def normalize_followers(text_field: Any) -> Tuple[List[List[str]], str]:
    """
    Expected PAN structure:
        text = [
            [tweet, tweet, ...],
            [tweet, tweet, ...],
            ...
        ]

    Returns follower blocks even if the data is flat.
    """
    if not isinstance(text_field, list):
        return [], "unknown"

    if len(text_field) == 0:
        return [], "empty"

    if all(isinstance(x, list) for x in text_field):
        follower_blocks = []

        for block in text_field:
            tweets = [
                str(t)
                for t in block
                if isinstance(t, str) and t.strip()
            ]
            follower_blocks.append(tweets)

        return follower_blocks, "nested_followers"

    if all(isinstance(x, str) for x in text_field):
        tweets = [
            str(t)
            for t in text_field
            if isinstance(t, str) and t.strip()
        ]
        return [tweets], "flat_tweets"

    return [], "unknown"


def flatten(follower_blocks: List[List[str]]) -> List[str]:
    return [tweet for block in follower_blocks for tweet in block]


def sample_fixed_followers_fixed_tweets(
    follower_blocks: List[List[str]],
    max_followers: int,
    max_tweets_per_follower: int,
    rng: random.Random,
) -> List[List[str]]:
    non_empty = [block for block in follower_blocks if len(block) > 0]

    if len(non_empty) > max_followers:
        selected_followers = rng.sample(non_empty, max_followers)
    else:
        selected_followers = list(non_empty)

    sampled = []

    for block in selected_followers:
        if len(block) > max_tweets_per_follower:
            sampled.append(rng.sample(block, max_tweets_per_follower))
        else:
            sampled.append(list(block))

    return sampled


def sample_tweet_cap(
    follower_blocks: List[List[str]],
    max_total_tweets: int,
    rng: random.Random,
) -> List[List[str]]:
    tweets = flatten(follower_blocks)

    if len(tweets) > max_total_tweets:
        tweets = rng.sample(tweets, max_total_tweets)

    return [tweets]


def sample_follower_dropout(
    follower_blocks: List[List[str]],
    dropout_rate: float,
    rng: random.Random,
) -> List[List[str]]:
    non_empty = [block for block in follower_blocks if len(block) > 0]

    if not non_empty:
        return []

    keep_prob = 1.0 - dropout_rate

    kept = [
        block
        for block in non_empty
        if rng.random() < keep_prob
    ]

    if not kept:
        kept = [rng.choice(non_empty)]

    return [list(block) for block in kept]


def sample_tweet_dropout(
    follower_blocks: List[List[str]],
    dropout_rate: float,
    rng: random.Random,
) -> List[List[str]]:
    keep_prob = 1.0 - dropout_rate
    sampled = []

    for block in follower_blocks:
        kept = [
            tweet
            for tweet in block
            if rng.random() < keep_prob
        ]

        if kept:
            sampled.append(kept)

    if not sampled:
        all_tweets = flatten(follower_blocks)
        if all_tweets:
            sampled = [[rng.choice(all_tweets)]]

    return sampled


def apply_sampling_config(
    follower_blocks: List[List[str]],
    config: dict,
    rng: random.Random,
) -> List[List[str]]:
    mode = config["mode"]

    if mode == "raw_all":
        return [list(block) for block in follower_blocks]

    if mode == "fixed_followers_fixed_tweets":
        return sample_fixed_followers_fixed_tweets(
            follower_blocks=follower_blocks,
            max_followers=int(config["max_followers"]),
            max_tweets_per_follower=int(config["max_tweets_per_follower"]),
            rng=rng,
        )

    if mode == "tweet_cap":
        return sample_tweet_cap(
            follower_blocks=follower_blocks,
            max_total_tweets=int(config["max_total_tweets"]),
            rng=rng,
        )

    if mode == "follower_dropout":
        return sample_follower_dropout(
            follower_blocks=follower_blocks,
            dropout_rate=float(config["dropout_rate"]),
            rng=rng,
        )

    if mode == "tweet_dropout":
        return sample_tweet_dropout(
            follower_blocks=follower_blocks,
            dropout_rate=float(config["dropout_rate"]),
            rng=rng,
        )

    raise ValueError(f"Unknown sampling mode: {mode}")


def text_stats(tweets: List[str]) -> dict:
    if not tweets:
        return {
            "num_tweets_total": 0,
            "tweet_char_mean": 0.0,
            "tweet_word_mean": 0.0,
            "hashtag_count": 0,
            "mention_count": 0,
            "url_count": 0,
            "retweet_count": 0,
        }

    char_lengths = [len(t) for t in tweets]
    word_lengths = [len(t.split()) for t in tweets]

    return {
        "num_tweets_total": int(len(tweets)),
        "tweet_char_mean": float(np.mean(char_lengths)),
        "tweet_word_mean": float(np.mean(word_lengths)),
        "hashtag_count": int(sum(t.count("#") for t in tweets)),
        "mention_count": int(sum(t.count("@") for t in tweets)),
        "url_count": int(
            sum(
                ("http://" in t or "https://" in t or "t.co/" in t)
                for t in tweets
            )
        ),
        "retweet_count": int(
            sum(t.strip().lower().startswith("rt ") for t in tweets)
        ),
    }


def compute_sample_stats(
    sampled_blocks: List[List[str]],
    label_row: dict,
    cid: str,
    split_name: str,
    config_name: str,
) -> dict:
    tweets = flatten(sampled_blocks)
    non_empty_followers = [block for block in sampled_blocks if len(block) > 0]
    tweets_per_follower = [len(block) for block in non_empty_followers]

    base = {
        "celebrity_id": cid,
        "split": split_name,
        "sampling_config": config_name,
        "occupation": str(label_row["occupation"]),
        "gender": str(label_row["gender"]),
        "birthyear": int(label_row["birthyear"]),
        "age_bin": make_age_bin(label_row["birthyear"]),
        "num_non_empty_followers": int(len(non_empty_followers)),
        "tweets_per_non_empty_follower_mean": float(np.mean(tweets_per_follower))
        if tweets_per_follower
        else 0.0,
        "tweets_per_non_empty_follower_median": float(np.median(tweets_per_follower))
        if tweets_per_follower
        else 0.0,
    }

    base.update(text_stats(tweets))

    return base


def analyze_split(
    split_name: str,
    labels: Dict[str, dict],
    feeds_path: str,
    configs: List[dict],
    seed: int,
) -> List[dict]:
    all_rows = []
    structure_counter = Counter()

    for row_idx, row in enumerate(iter_ndjson(feeds_path)):
        cid = get_id(row)

        if cid not in labels:
            continue

        follower_blocks, structure_type = normalize_followers(row.get("text", []))
        structure_counter[structure_type] += 1

        for config_idx, config in enumerate(configs):
            rng = random.Random(seed + row_idx * 1009 + config_idx * 9173)

            sampled_blocks = apply_sampling_config(
                follower_blocks=follower_blocks,
                config=config,
                rng=rng,
            )

            stat = compute_sample_stats(
                sampled_blocks=sampled_blocks,
                label_row=labels[cid],
                cid=cid,
                split_name=split_name,
                config_name=config["name"],
            )

            all_rows.append(stat)

    print(f"[INFO] {split_name} structure counter: {dict(structure_counter)}")
    return all_rows


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
            "std": None,
            "cv": None,
        }

    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))

    return {
        "n": int(len(arr)),
        "mean": mean,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "std": std,
        "cv": float(std / mean) if mean != 0 else None,
    }


def summarize_by_config_and_group(
    rows: List[dict],
    group_key: str,
) -> dict:
    metrics = [
        "num_tweets_total",
        "num_non_empty_followers",
        "tweets_per_non_empty_follower_mean",
        "tweet_word_mean",
        "hashtag_count",
        "mention_count",
        "url_count",
        "retweet_count",
    ]

    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for row in rows:
        config = row["sampling_config"]
        group = str(row[group_key])

        for metric in metrics:
            grouped[config][group][metric].append(row[metric])

    summary = {}

    for config, group_dict in grouped.items():
        summary[config] = {}

        for group, metric_values in group_dict.items():
            summary[config][group] = {
                metric: summarize_numeric(values)
                for metric, values in metric_values.items()
            }

    return summary


def compute_balance_scores(summary_by_occupation: dict) -> dict:
    """
    Measures how unequal the occupation groups remain after sampling.

    Lower ratio/max-min scores are better.
    """
    metrics = [
        "num_tweets_total",
        "num_non_empty_followers",
        "retweet_count",
        "hashtag_count",
        "mention_count",
        "url_count",
        "tweet_word_mean",
    ]

    out = {}

    for config_name, occupation_summary in summary_by_occupation.items():
        out[config_name] = {}

        for metric in metrics:
            means = []

            for occ in OCCUPATION_ORDER:
                if occ in occupation_summary:
                    value = occupation_summary[occ][metric]["mean"]
                    if value is not None:
                        means.append(float(value))

            if not means:
                continue

            max_mean = max(means)
            min_mean = min(means)
            mean_of_means = float(np.mean(means))
            std_of_means = float(np.std(means))

            out[config_name][metric] = {
                "min_group_mean": float(min_mean),
                "max_group_mean": float(max_mean),
                "max_min_ratio": float(max_mean / min_mean) if min_mean > 0 else None,
                "range": float(max_mean - min_mean),
                "std_between_group_means": std_of_means,
                "cv_between_group_means": float(std_of_means / mean_of_means)
                if mean_of_means != 0
                else None,
            }

    return out


def plot_config_metric_by_occupation(
    summary_by_occupation: dict,
    metric: str,
    out_path: str,
):
    config_names = [config["name"] for config in SAMPLING_CONFIGS]
    x = np.arange(len(config_names))
    width = 0.18

    plt.figure(figsize=(14, 6))

    for idx, occ in enumerate(OCCUPATION_ORDER):
        values = []

        for config_name in config_names:
            value = None

            if (
                config_name in summary_by_occupation
                and occ in summary_by_occupation[config_name]
                and metric in summary_by_occupation[config_name][occ]
            ):
                value = summary_by_occupation[config_name][occ][metric]["mean"]

            values.append(value if value is not None else 0.0)

        plt.bar(x + (idx - 1.5) * width, values, width, label=occ)

    plt.title(f"Mean {metric} by occupation and sampling config")
    plt.xticks(x, config_names, rotation=35, ha="right")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_balance_score(
    balance_scores: dict,
    metric: str,
    score_key: str,
    out_path: str,
):
    config_names = [config["name"] for config in SAMPLING_CONFIGS]
    values = []

    for config_name in config_names:
        value = None

        if config_name in balance_scores and metric in balance_scores[config_name]:
            value = balance_scores[config_name][metric].get(score_key)

        values.append(value if value is not None else 0.0)

    plt.figure(figsize=(12, 5))
    plt.bar(config_names, values)
    plt.title(f"{score_key} for {metric} across occupation groups")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(score_key)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def md_metric_table(
    title: str,
    summary_by_occupation: dict,
    metric: str,
) -> str:
    lines = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append(f"Metric: `{metric}`")
    lines.append("")
    lines.append("| sampling_config | sports | performer | creator | politics | max/min ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for config in SAMPLING_CONFIGS:
        config_name = config["name"]
        values = []

        for occ in OCCUPATION_ORDER:
            if (
                config_name in summary_by_occupation
                and occ in summary_by_occupation[config_name]
            ):
                value = summary_by_occupation[config_name][occ][metric]["mean"]
            else:
                value = None

            values.append(value)

        numeric_values = [v for v in values if v is not None]
        ratio = max(numeric_values) / min(numeric_values) if numeric_values and min(numeric_values) > 0 else None

        def fmt(x):
            return f"{x:.2f}" if x is not None else "-"

        lines.append(
            f"| {config_name} | "
            f"{fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])} | {fmt(values[3])} | "
            f"{fmt(ratio)} |"
        )

    lines.append("")
    return "\n".join(lines)


def md_balance_score_table(
    title: str,
    balance_scores: dict,
    metric: str,
) -> str:
    lines = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append(f"Metric: `{metric}`")
    lines.append("")
    lines.append("| sampling_config | min_mean | max_mean | max/min ratio | range | cv_between_classes |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for config in SAMPLING_CONFIGS:
        config_name = config["name"]
        score = balance_scores.get(config_name, {}).get(metric, {})

        def fmt(x):
            return f"{x:.4f}" if x is not None else "-"

        lines.append(
            f"| {config_name} | "
            f"{fmt(score.get('min_group_mean'))} | "
            f"{fmt(score.get('max_group_mean'))} | "
            f"{fmt(score.get('max_min_ratio'))} | "
            f"{fmt(score.get('range'))} | "
            f"{fmt(score.get('cv_between_group_means'))} |"
        )

    lines.append("")
    return "\n".join(lines)


def build_report(
    summary: dict,
    balance_scores: dict,
) -> str:
    lines = []

    lines.append("# HybridV4 Controlled Feed Sampling Simulation")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "This simulation compares different tweet/follower sampling strategies without training a model. "
        "It checks whether controlled sampling reduces activity and coverage differences between occupation classes."
    )
    lines.append("")

    for split_name in ["train", "test", "supplement"]:
        lines.append(f"## {split_name}")
        lines.append("")

        summary_by_occupation = summary[split_name]["by_occupation"]
        split_balance = balance_scores[split_name]

        lines.append(md_metric_table(
            "Mean tweets per celebrity by occupation",
            summary_by_occupation,
            "num_tweets_total",
        ))

        lines.append(md_balance_score_table(
            "Occupation balance score for tweets per celebrity",
            split_balance,
            "num_tweets_total",
        ))

        lines.append(md_metric_table(
            "Mean non-empty followers by occupation",
            summary_by_occupation,
            "num_non_empty_followers",
        ))

        lines.append(md_balance_score_table(
            "Occupation balance score for non-empty followers",
            split_balance,
            "num_non_empty_followers",
        ))

        lines.append(md_metric_table(
            "Mean retweets by occupation",
            summary_by_occupation,
            "retweet_count",
        ))

        lines.append(md_balance_score_table(
            "Occupation balance score for retweets",
            split_balance,
            "retweet_count",
        ))

        lines.append(md_metric_table(
            "Mean tweet word length by occupation",
            summary_by_occupation,
            "tweet_word_mean",
        ))

    lines.append("## Interpretation hints")
    lines.append("")
    lines.append("- Lower `max/min ratio` means that a sampling strategy makes occupation classes more similar in coverage/activity.")
    lines.append("- If `pan_style_10_followers_20_tweets` strongly reduces tweet/follower imbalance, it is a good candidate for a BERTweet V3.5 preprocessing variant.")
    lines.append("- If follower dropout reduces imbalance but keeps enough tweets, it is a candidate for training-time augmentation.")
    lines.append("- If raw train and raw test show different activity patterns, performance differences may reflect distribution shift, not only model weakness.")
    lines.append("- Attention pooling can later build on this because sampled follower blocks can become explicit units for attention.")
    lines.append("")

    return "\n".join(lines)


def run():
    out_dir = os.path.join(
        hybrid_v4_output_dir,
        "data_audit",
        "balanced_feed_sampling_simulation",
    )
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

    all_rows_by_split = {}

    for split_name in ["train", "test", "supplement"]:
        rows = analyze_split(
            split_name=split_name,
            labels=labels_by_split[split_name],
            feeds_path=feeds_by_split[split_name],
            configs=SAMPLING_CONFIGS,
            seed=RANDOM_SEED,
        )
        all_rows_by_split[split_name] = rows

    summary = {}
    balance_scores = {}

    for split_name, rows in all_rows_by_split.items():
        summary[split_name] = {
            "by_occupation": summarize_by_config_and_group(rows, "occupation"),
            "by_gender": summarize_by_config_and_group(rows, "gender"),
            "by_age_bin": summarize_by_config_and_group(rows, "age_bin"),
        }

        balance_scores[split_name] = compute_balance_scores(
            summary[split_name]["by_occupation"]
        )

        for metric in [
            "num_tweets_total",
            "num_non_empty_followers",
            "retweet_count",
            "tweet_word_mean",
        ]:
            plot_config_metric_by_occupation(
                summary_by_occupation=summary[split_name]["by_occupation"],
                metric=metric,
                out_path=os.path.join(
                    plots_dir,
                    f"{split_name}_{metric}_by_occupation_and_sampling.png",
                ),
            )

        for metric in [
            "num_tweets_total",
            "num_non_empty_followers",
            "retweet_count",
        ]:
            plot_balance_score(
                balance_scores=balance_scores[split_name],
                metric=metric,
                score_key="max_min_ratio",
                out_path=os.path.join(
                    plots_dir,
                    f"{split_name}_{metric}_occupation_max_min_ratio.png",
                ),
            )

    save_json(
        all_rows_by_split,
        os.path.join(out_dir, "balanced_feed_sampling_raw_rows.json"),
    )
    save_json(
        summary,
        os.path.join(out_dir, "balanced_feed_sampling_summary.json"),
    )
    save_json(
        balance_scores,
        os.path.join(out_dir, "balanced_feed_sampling_balance_scores.json"),
    )

    report = build_report(
        summary=summary,
        balance_scores=balance_scores,
    )

    save_text(
        report,
        os.path.join(out_dir, "balanced_feed_sampling_simulation_report.md"),
    )

    print(f"[OK] Saved report: {os.path.join(out_dir, 'balanced_feed_sampling_simulation_report.md')}")
    print(f"[OK] Saved summary: {os.path.join(out_dir, 'balanced_feed_sampling_summary.json')}")
    print(f"[OK] Saved balance scores: {os.path.join(out_dir, 'balanced_feed_sampling_balance_scores.json')}")
    print(f"[OK] Saved plots: {plots_dir}")


def main():
    run()


if __name__ == "__main__":
    main()