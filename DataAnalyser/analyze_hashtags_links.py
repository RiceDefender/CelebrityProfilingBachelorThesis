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

REFERENCE_YEAR = 2020

# ---------------------------
# HELPERS
# ---------------------------
def load_labels(path):
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            labels[obj["id"]] = {
                "gender": obj["gender"],
                "occupation": obj["occupation"],
                "birthyear": int(obj["birthyear"])
            }
    return labels


def age_group(birthyear):
    age = REFERENCE_YEAR - birthyear

    if age < 20:
        return "<20"
    elif age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    else:
        return "50+"


def init_stats():
    return defaultdict(lambda: {
        "celeb_count": 0,
        "tweet_count": 0,
        "hashtag_count": 0,
        "link_count": 0,
        "tweets_with_hashtag": 0,
        "tweets_with_link": 0
    })


def extract_features(tweet):
    # Hashtags (robuster als count("#"))
    hashtags = re.findall(r"#\w+", tweet)

    # Links (Twitter typisch)
    links = re.findall(r"http\S+", tweet)

    return len(hashtags), len(links)


# ---------------------------
# CORE
# ---------------------------
def analyze_dataset(label_path, feeds_path):
    labels = load_labels(label_path)

    gender_stats = init_stats()
    occupation_stats = init_stats()
    age_stats = init_stats()

    with open(feeds_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            obj = json.loads(line)
            cid = obj["id"]

            if cid not in labels:
                continue

            meta = labels[cid]
            gender = meta["gender"]
            occupation = meta["occupation"]
            age = age_group(meta["birthyear"])

            tweets = [t for follower in obj["text"] for t in follower]

            celeb_tweets = 0
            celeb_hashtags = 0
            celeb_links = 0
            celeb_tweets_hashtag = 0
            celeb_tweets_link = 0

            for tweet in tweets:
                celeb_tweets += 1

                h_count, l_count = extract_features(tweet)

                celeb_hashtags += h_count
                celeb_links += l_count

                if h_count > 0:
                    celeb_tweets_hashtag += 1
                if l_count > 0:
                    celeb_tweets_link += 1

            # --- Update stats ---
            for stats, key in [
                (gender_stats, gender),
                (occupation_stats, occupation),
                (age_stats, age)
            ]:
                stats[key]["celeb_count"] += 1
                stats[key]["tweet_count"] += celeb_tweets
                stats[key]["hashtag_count"] += celeb_hashtags
                stats[key]["link_count"] += celeb_links
                stats[key]["tweets_with_hashtag"] += celeb_tweets_hashtag
                stats[key]["tweets_with_link"] += celeb_tweets_link

            if i % 500 == 0:
                print(f"Processed {i} celebrities")

    return gender_stats, occupation_stats, age_stats


# ---------------------------
# PLOTTING
# ---------------------------
def plot_metric(stats, value_fn, title, ylabel, filename):
    groups = []
    values = []

    for g, s in stats.items():
        if s["tweet_count"] == 0:
            continue

        groups.append(g)
        values.append(value_fn(s))

    plt.figure(figsize=(8, 5))
    plt.bar(groups, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

def merge_stats(*stats_list):
    merged = init_stats()
    for stats in stats_list:
        for group_name, values in stats.items():
            for key, value in values.items():
                merged[group_name][key] += value
    return merged


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    os.makedirs(plots_dir, exist_ok=True)

    # Train
    train_gender, train_occupation, train_age = analyze_dataset(
        train_label_path, train_feeds_path
    )

    # Test
    test_gender, test_occupation, test_age = analyze_dataset(
        test_label_path, test_feeds_path
    )

    gender = merge_stats(train_gender, test_gender)
    occupation = merge_stats(train_occupation, test_occupation)
    age = merge_stats(train_age, test_age)

    # ---------- HASHTAGS ----------
    plot_metric(
        gender,
        lambda s: s["hashtag_count"] / s["tweet_count"],
        "Avg Hashtags per Tweet (Gender)",
        "hashtags per tweet",
        "hashtags_gender.png"
    )

    plot_metric(
        gender,
        lambda s: s["tweets_with_hashtag"] / s["tweet_count"],
        "Share Tweets with Hashtags (Gender)",
        "share",
        "hashtags_share_gender.png"
    )

    # ---------- LINKS ----------
    plot_metric(
        gender,
        lambda s: s["link_count"] / s["tweet_count"],
        "Avg Links per Tweet (Gender)",
        "links per tweet",
        "links_gender.png"
    )

    plot_metric(
        gender,
        lambda s: s["tweets_with_link"] / s["tweet_count"],
        "Share Tweets with Links (Gender)",
        "share",
        "links_share_gender.png"
    )

    # ---------- OCCUPATION ----------
    plot_metric(
        occupation,
        lambda s: s["hashtag_count"] / s["tweet_count"],
        "Avg Hashtags per Tweet (Occupation)",
        "hashtags per tweet",
        "hashtags_occupation.png"
    )

    plot_metric(
        occupation,
        lambda s: s["tweets_with_hashtag"] / s["tweet_count"],
        "Share Tweets with Hashtags (Occupation)",
        "share",
        "hashtags_share_occupation.png"
    )

    plot_metric(
        occupation,
        lambda s: s["link_count"] / s["tweet_count"],
        "Avg Links per Tweet (Occupation)",
        "links per tweet",
        "links_occupation.png"
    )

    plot_metric(
        occupation,
        lambda s: s["tweets_with_link"] / s["tweet_count"],
        "Share Tweets with Links (Occupation)",
        "share",
        "links_share_occupation.png"
    )

    # ---------- AGE ----------
    plot_metric(
        age,
        lambda s: s["hashtag_count"] / s["tweet_count"],
        "Avg Hashtags per Tweet (Age)",
        "hashtags per tweet",
        "hashtags_age.png"
    )

    plot_metric(
        age,
        lambda s: s["tweets_with_hashtag"] / s["tweet_count"],
        "Share Tweets with Hashtags (Age)",
        "share",
        "hashtags_share_age.png"
    )

    plot_metric(
        age,
        lambda s: s["link_count"] / s["tweet_count"],
        "Avg Links per Tweet (Age)",
        "links per tweet",
        "links_age.png"
    )

    plot_metric(
        age,
        lambda s: s["tweets_with_link"] / s["tweet_count"],
        "Share Tweets with Links (Age)",
        "share",
        "links_share_age.png"
    )

    print("\nDone. Plots saved.")