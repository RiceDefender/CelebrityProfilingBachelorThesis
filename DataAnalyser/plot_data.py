import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from _constants import train_label_path, supp_label_path, test_label_path, plots_dir


def load_labels(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({
                "gender": obj.get("gender"),
                "age": obj.get("birthyear"),
                "occupation": obj.get("occupation")
            })
    return pd.DataFrame(rows)


def compare_barplots(datasets, column, title, sort_index=False):
    fig, axes = plt.subplots(1, len(datasets), figsize=(16, 5))

    if len(datasets) == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, datasets.items()):
        counts = df[column].value_counts(dropna=False)
        if sort_index:
            counts = counts.sort_index()

        counts.plot(kind="bar", ax=ax)
        ax.set_title(f"{name}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{column}_distribution.png"))


def main():
    print("Train path:", train_label_path)
    print("Supplement path:", supp_label_path)
    print("Test path:", test_label_path)

    datasets = {
        "Train": load_labels(train_label_path),
        "Supplement": load_labels(supp_label_path),
        "Test": load_labels(test_label_path),
    }

    compare_barplots(datasets, "gender", "Gender Distribution")
    compare_barplots(datasets, "age", "Age Distribution", sort_index=True)
    compare_barplots(datasets, "occupation", "Occupation Distribution")


if __name__ == "__main__":
    main()