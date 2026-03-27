import json
import os
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import emoji

from _constants import (
    train_label_path, train_feeds_path,
    supp_label_path, supp_feeds_path,
    test_label_path, test_feeds_path,
    plots_dir
)


# ---------------------------
# CONFIG
# ---------------------------
REFERENCE_YEAR = 2020  # PAN 2020 Dataset


# ---------------------------
# HELPERS
# ---------------------------
def load_labels(path):
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            labels[obj["id"]] = {
                "gender": obj["gender"],
                "occupation": obj["occupation"],
                "birthyear": int(obj["birthyear"])
            }
    return labels


def age_group_from_birthyear(birthyear, reference_year=REFERENCE_YEAR):
    age = reference_year - birthyear

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


def init_group_stats():
    return defaultdict(lambda: {
        "celeb_count": 0,
        "tweet_count": 0,
        "emoji_count": 0,
        "tweets_with_emoji": 0,
        "emoji_counter": Counter()
    })


def merge_group_stats(*stats_list):
    merged = init_group_stats()

    for stats in stats_list:
        for group_name, values in stats.items():
            merged[group_name]["celeb_count"] += values["celeb_count"]
            merged[group_name]["tweet_count"] += values["tweet_count"]
            merged[group_name]["emoji_count"] += values["emoji_count"]
            merged[group_name]["tweets_with_emoji"] += values["tweets_with_emoji"]
            merged[group_name]["emoji_counter"].update(values["emoji_counter"])

    return merged


def extract_emojis(tweet):
    """
    Uses the emoji library to ensure that composite emojis
    (e.g. with skin tones, ZWJ sequences, etc.) are recognised correctly.

    IMPORTANT:
    If emojis are stored in the JSON as \\ud83e\\udd1d,
    json.loads(...) will normally convert this correctly to Unicode.
    """
    found = emoji.emoji_list(tweet)
    return [entry["emoji"] for entry in found]


# ---------------------------
# CORE ANALYSIS
# ---------------------------
def analyze_dataset_emoji(label_path, feeds_path):
    labels = load_labels(label_path)

    gender_stats = init_group_stats()
    occupation_stats = init_group_stats()
    age_stats = init_group_stats()

    with open(feeds_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            celeb_id = obj["id"]

            if celeb_id not in labels:
                continue

            meta = labels[celeb_id]
            gender = meta["gender"]
            occupation = meta["occupation"]
            age_group = age_group_from_birthyear(meta["birthyear"])

            tweets = [tweet for follower in obj["text"] for tweet in follower]

            # Pro Celebrity aggregieren
            celeb_tweet_count = 0
            celeb_emoji_count = 0
            celeb_tweets_with_emoji = 0
            celeb_emoji_counter = Counter()

            for tweet in tweets:
                celeb_tweet_count += 1

                emojis_in_tweet = extract_emojis(tweet)
                emoji_count = len(emojis_in_tweet)

                celeb_emoji_count += emoji_count

                if emoji_count > 0:
                    celeb_tweets_with_emoji += 1
                    celeb_emoji_counter.update(emojis_in_tweet)

            # Gender
            gender_stats[gender]["celeb_count"] += 1
            gender_stats[gender]["tweet_count"] += celeb_tweet_count
            gender_stats[gender]["emoji_count"] += celeb_emoji_count
            gender_stats[gender]["tweets_with_emoji"] += celeb_tweets_with_emoji
            gender_stats[gender]["emoji_counter"].update(celeb_emoji_counter)

            # Occupation
            occupation_stats[occupation]["celeb_count"] += 1
            occupation_stats[occupation]["tweet_count"] += celeb_tweet_count
            occupation_stats[occupation]["emoji_count"] += celeb_emoji_count
            occupation_stats[occupation]["tweets_with_emoji"] += celeb_tweets_with_emoji
            occupation_stats[occupation]["emoji_counter"].update(celeb_emoji_counter)

            # Age Group
            age_stats[age_group]["celeb_count"] += 1
            age_stats[age_group]["tweet_count"] += celeb_tweet_count
            age_stats[age_group]["emoji_count"] += celeb_emoji_count
            age_stats[age_group]["tweets_with_emoji"] += celeb_tweets_with_emoji
            age_stats[age_group]["emoji_counter"].update(celeb_emoji_counter)

            if i % 500 == 0:
                print(f"Processed {i} celebrities from {os.path.basename(feeds_path)}")

    return gender_stats, occupation_stats, age_stats


# ---------------------------
# PLOTTING
# ---------------------------
def plot_avg_emojis_per_tweet(stats, title, filename):
    groups = []
    values = []

    for group_name, s in stats.items():
        if s["tweet_count"] == 0:
            continue
        groups.append(group_name)
        values.append(s["emoji_count"] / s["tweet_count"])

    plt.figure(figsize=(8, 5))
    plt.bar(groups, values)
    plt.title(title)
    plt.xlabel("Group")
    plt.ylabel("Average emojis per tweet")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()


def plot_share_tweets_with_emoji(stats, title, filename):
    groups = []
    values = []

    for group_name, s in stats.items():
        if s["tweet_count"] == 0:
            continue
        groups.append(group_name)
        values.append(s["tweets_with_emoji"] / s["tweet_count"])

    plt.figure(figsize=(8, 5))
    plt.bar(groups, values)
    plt.title(title)
    plt.xlabel("Group")
    plt.ylabel("Share of tweets containing emojis")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()


def save_top_emojis(stats, filename, top_n=10):
    out_path = os.path.join(plots_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        for group_name, s in stats.items():
            f.write(f"[{group_name}]\n")
            top = s["emoji_counter"].most_common(top_n)
            if not top:
                f.write("No emojis found.\n\n")
                continue

            for emj, count in top:
                f.write(f"{emj}\t{count}\n")
            f.write("\n")


def print_summary(stats, group_label):
    print(f"\n=== {group_label} ===")
    for group_name, s in stats.items():
        tweet_count = s["tweet_count"]
        emoji_count = s["emoji_count"]
        tweets_with_emoji = s["tweets_with_emoji"]

        avg_emojis = (emoji_count / tweet_count) if tweet_count else 0.0
        share_emoji_tweets = (tweets_with_emoji / tweet_count) if tweet_count else 0.0

        print(
            f"{group_name:>12} | "
            f"celebs={s['celeb_count']:5d} | "
            f"tweets={tweet_count:8d} | "
            f"avg_emojis/tweet={avg_emojis:.4f} | "
            f"share_tweets_with_emoji={share_emoji_tweets:.4f}"
        )


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    os.makedirs(plots_dir, exist_ok=True)

    # Train
    train_gender, train_occ, train_age = analyze_dataset_emoji(train_label_path, train_feeds_path)

    # Supplement
    # supp_gender, supp_occ, supp_age = analyze_dataset_emoji(supp_label_path, supp_feeds_path)

    # Test
    test_gender, test_occ, test_age = analyze_dataset_emoji(test_label_path, test_feeds_path)

    # Mergen
    gender_stats = merge_group_stats(train_gender, test_gender)
    occupation_stats = merge_group_stats(train_occ, test_occ)
    age_stats = merge_group_stats(train_age, test_age)

    # Summary
    print_summary(gender_stats, "Gender")
    print_summary(occupation_stats, "Occupation")
    print_summary(age_stats, "Age Group")

    # Plots
    plot_avg_emojis_per_tweet(
        gender_stats,
        "Average Emojis per Tweet by Gender",
        "emoji_avg_per_tweet_gender.png"
    )
    plot_share_tweets_with_emoji(
        gender_stats,
        "Share of Tweets with Emojis by Gender",
        "emoji_share_tweets_gender.png"
    )

    plot_avg_emojis_per_tweet(
        occupation_stats,
        "Average Emojis per Tweet by Occupation",
        "emoji_avg_per_tweet_occupation.png"
    )
    plot_share_tweets_with_emoji(
        occupation_stats,
        "Share of Tweets with Emojis by Occupation",
        "emoji_share_tweets_occupation.png"
    )

    plot_avg_emojis_per_tweet(
        age_stats,
        "Average Emojis per Tweet by Age Group",
        "emoji_avg_per_tweet_age.png"
    )
    plot_share_tweets_with_emoji(
        age_stats,
        "Share of Tweets with Emojis by Age Group",
        "emoji_share_tweets_age.png"
    )

    # Top Emojis
    save_top_emojis(gender_stats, "top_emojis_gender.txt", top_n=10)
    save_top_emojis(occupation_stats, "top_emojis_occupation.txt", top_n=10)
    save_top_emojis(age_stats, "top_emojis_age.txt", top_n=10)

    print(f"\nDone. Results saved in: {plots_dir}")