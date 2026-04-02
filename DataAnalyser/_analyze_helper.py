import os
from collections import defaultdict
import json
import matplotlib.pyplot as plt

from _constants import plots_dir

REFERENCE_YEAR = 2020 # PAN 2020 Dataset

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

def merge_stats(*stats_list, init_stats=lambda: defaultdict(lambda: defaultdict(int))):
    merged = init_stats()
    for stats in stats_list:
        for group_name, values in stats.items():
            for key, value in values.items():
                merged[group_name][key] += value
    return merged

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