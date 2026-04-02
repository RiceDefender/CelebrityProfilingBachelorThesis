import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from _constants import (
    train_label_path, train_feeds_path,
    test_label_path, test_feeds_path,
    plots_dir
)

from analyze_emojis import extract_emojis
from analyze_hashtags_links import extract_features

from _analyze_helper import load_labels, age_group


def collect_per_celeb_values(label_path, feeds_path):
    labels = load_labels(label_path)

    gender_data = defaultdict(list)
    occupation_data = defaultdict(list)
    age_data = defaultdict(list)

    with open(feeds_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            cid = obj["id"]

            if cid not in labels:
                continue

            meta = labels[cid]
            gender = meta["gender"]
            occupation = meta["occupation"]
            age = age_group(meta["birthyear"])

            tweets = [t for follower in obj["text"] for t in follower]
            if not tweets:
                continue

            total_emojis = 0
            total_hashtags = 0
            total_links = 0

            for tweet in tweets:
                total_emojis += len(extract_emojis(tweet))

                h_count, l_count = extract_features(tweet)
                total_hashtags += h_count
                total_links += l_count

            emoji_avg = total_emojis / len(tweets)
            hashtag_avg = total_hashtags / len(tweets)
            link_avg = total_links / len(tweets)

            gender_data[gender].append((emoji_avg, hashtag_avg, link_avg))
            occupation_data[occupation].append((emoji_avg, hashtag_avg, link_avg))
            age_data[age].append((emoji_avg, hashtag_avg, link_avg))

            if i % 500 == 0:
                print(f"Processed {i} celebrities from {os.path.basename(feeds_path)}")

    return gender_data, occupation_data, age_data


def merge_group_lists(*group_dicts):
    merged = defaultdict(list)

    for group_dict in group_dicts:
        for key, values in group_dict.items():
            merged[key].extend(values)

    return merged


def sort_age_groups(keys):
    preferred = ["<20", "20-29", "30-39", "40-49", "50+", "unknown"]
    ordered = [k for k in preferred if k in keys]
    remaining = [k for k in keys if k not in preferred]
    return ordered + sorted(remaining)


def plot_boxplot(data, index, title, ylabel, filename, ordered_keys=None):
    if ordered_keys is None:
        ordered_keys = list(data.keys())

    groups = []
    values = []

    for group in ordered_keys:
        vals = [x[index] for x in data[group]]
        if not vals:
            continue
        groups.append(group)
        values.append(vals)

    plt.figure(figsize=(9, 5))
    plt.boxplot(values, tick_labels=groups)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()


if __name__ == "__main__":
    os.makedirs(plots_dir, exist_ok=True)

    # train
    train_gender, train_occ, train_age = collect_per_celeb_values(
        train_label_path, train_feeds_path
    )

    # test
    test_gender, test_occ, test_age = collect_per_celeb_values(
        test_label_path, test_feeds_path
    )

    # merge
    gender_data = merge_group_lists(train_gender, test_gender)
    occupation_data = merge_group_lists(train_occ, test_occ)
    age_data = merge_group_lists(train_age, test_age)

    age_keys = sort_age_groups(list(age_data.keys()))

    # ---------------------------
    # EMOJIS
    # index 0
    # ---------------------------
    plot_boxplot(
        gender_data, 0,
        "Emoji Distribution per Celebrity (Gender)",
        "avg emojis per tweet",
        "box_emoji_gender.png"
    )

    plot_boxplot(
        occupation_data, 0,
        "Emoji Distribution per Celebrity (Occupation)",
        "avg emojis per tweet",
        "box_emoji_occupation.png"
    )

    plot_boxplot(
        age_data, 0,
        "Emoji Distribution per Celebrity (Age)",
        "avg emojis per tweet",
        "box_emoji_age.png",
        ordered_keys=age_keys
    )

    # ---------------------------
    # HASHTAGS
    # index 1
    # ---------------------------
    plot_boxplot(
        gender_data, 1,
        "Hashtag Distribution per Celebrity (Gender)",
        "avg hashtags per tweet",
        "box_hashtag_gender.png"
    )

    plot_boxplot(
        occupation_data, 1,
        "Hashtag Distribution per Celebrity (Occupation)",
        "avg hashtags per tweet",
        "box_hashtag_occupation.png"
    )

    plot_boxplot(
        age_data, 1,
        "Hashtag Distribution per Celebrity (Age)",
        "avg hashtags per tweet",
        "box_hashtag_age.png",
        ordered_keys=age_keys
    )

    # ---------------------------
    # LINKS
    # index 2
    # ---------------------------
    plot_boxplot(
        gender_data, 2,
        "Link Distribution per Celebrity (Gender)",
        "avg links per tweet",
        "box_links_gender.png"
    )

    plot_boxplot(
        occupation_data, 2,
        "Link Distribution per Celebrity (Occupation)",
        "avg links per tweet",
        "box_links_occupation.png"
    )

    plot_boxplot(
        age_data, 2,
        "Link Distribution per Celebrity (Age)",
        "avg links per tweet",
        "box_links_age.png",
        ordered_keys=age_keys
    )

    print("\nDone. Boxplots saved.")
