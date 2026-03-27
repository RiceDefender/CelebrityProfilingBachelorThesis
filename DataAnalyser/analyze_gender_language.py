import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

from _constants import (
    train_label_path, train_feeds_path,
    supp_label_path, supp_feeds_path,
    test_label_path, test_feeds_path,
    plots_dir
)

from _analyze_helper import merge_stats

PERSONAL_PRONOUNS = {
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "mine", "yours"
}

DETERMINERS = {
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their"
}

QUANTIFIERS = {
    "all", "some", "many", "few", "several",
    "much", "more", "most", "less", "least",
    "two", "three", "four", "five"
}


def load_labels(path):
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            labels[obj["id"]] = obj["gender"]
    return labels


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def init_stats():
    return defaultdict(lambda: {
        "celeb_count": 0,          # Number of celebrities by gender
        "tweet_count": 0,          # Total number of tweets
        "pronoun_count": 0,        # Total number of pronouns
        "det_quant_count": 0       # Total number of determiners and quantifiers
    })


def stream_analyze_dataset(label_path, feeds_path):
    labels = load_labels(label_path)   # klein -> ok im RAM
    stats = init_stats()

    with open(feeds_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            celeb_id = obj["id"]

            if celeb_id not in labels:
                continue

            gender = labels[celeb_id]
            follower_lists = obj["text"]

            celeb_tweet_count = 0
            celeb_pronouns = 0
            celeb_det_quant = 0

            for follower_tweets in follower_lists:
                for tweet in follower_tweets:
                    celeb_tweet_count += 1
                    tokens = tokenize(tweet)

                    celeb_pronouns += sum(1 for t in tokens if t in PERSONAL_PRONOUNS)
                    celeb_det_quant += sum(
                        1 for t in tokens if t in DETERMINERS or t in QUANTIFIERS
                    )

            stats[gender]["celeb_count"] += 1
            stats[gender]["tweet_count"] += celeb_tweet_count
            stats[gender]["pronoun_count"] += celeb_pronouns
            stats[gender]["det_quant_count"] += celeb_det_quant

            if i % 1000 == 0:
                print(f"Processed {i} lines from {os.path.basename(feeds_path)}")

    return stats



def plot_avg_tweets_per_celebrity(stats):
    genders = []
    values = []

    for gender, s in stats.items():
        if s["celeb_count"] == 0:
            continue
        genders.append(gender)
        values.append(s["tweet_count"] / s["celeb_count"])

    plt.figure(figsize=(6, 4))
    plt.bar(genders, values)
    plt.title("Average Tweets per Celebrity by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Average number of tweets")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "avg_tweets_per_celebrity_by_gender.png"))
    plt.close()


def plot_avg_pronouns_per_tweet(stats):
    genders = []
    values = []

    for gender, s in stats.items():
        if s["tweet_count"] == 0:
            continue
        genders.append(gender)
        values.append(s["pronoun_count"] / s["tweet_count"])

    plt.figure(figsize=(6, 4))
    plt.bar(genders, values)
    plt.title("Average Personal Pronouns per Tweet by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Average pronouns per tweet")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "avg_pronouns_per_tweet_by_gender.png"))
    plt.close()


def plot_avg_det_quant_per_tweet(stats):
    genders = []
    values = []

    for gender, s in stats.items():
        if s["tweet_count"] == 0:
            continue
        genders.append(gender)
        values.append(s["det_quant_count"] / s["tweet_count"])

    plt.figure(figsize=(6, 4))
    plt.bar(genders, values)
    plt.title("Average Determiners + Quantifiers per Tweet by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Average count per tweet")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "avg_det_quant_per_tweet_by_gender.png"))
    plt.close()


if __name__ == "__main__":
    os.makedirs(plots_dir, exist_ok=True)

    train_stats = stream_analyze_dataset(train_label_path, train_feeds_path)
    # supp_stats = stream_analyze_dataset(supp_label_path, supp_feeds_path)
    test_stats = stream_analyze_dataset(test_label_path, test_feeds_path)

    all_stats = merge_stats(train_stats, test_stats, init_stats=init_stats)

    print("\nFinal stats:")
    for gender, s in all_stats.items():
        print(gender, s)

    plot_avg_tweets_per_celebrity(all_stats)
    plot_avg_pronouns_per_tweet(all_stats)
    plot_avg_det_quant_per_tweet(all_stats)

    print(f"\nPlots saved to: {plots_dir}")