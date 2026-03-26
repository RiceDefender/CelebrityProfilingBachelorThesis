import os
import json
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")

train_path = os.path.join(
    data_dir,
    "pan20-celebrity-profiling-training-dataset-2020-02-28",
    "pan20-celebrity-profiling-training-dataset-2020-02-28",
    "labels.ndjson"
)

supp_path = os.path.join(
    data_dir,
    "pan20-celebrity-profiling-supplement-dataset-2020-02-28",
    "pan20-celebrity-profiling-supplement-dataset-2020-02-28",
    "labels.ndjson"
)

test_path = os.path.join(
    data_dir,
    "pan20-celebrity-profiling-test-dataset-2020-02-28",
    "pan20-celebrity-profiling-test-dataset-2020-02-28",
    "labels.ndjson"
)


def load_labels(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({
                "gender": obj.get("gender"),
                "age": obj.get("age"),
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
    plt.show()


def main():
    print("Train path:", train_path)
    print("Supplement path:", supp_path)
    print("Test path:", test_path)

    datasets = {
        "Train": load_labels(train_path),
        "Supplement": load_labels(supp_path),
        "Test": load_labels(test_path),
    }

    compare_barplots(datasets, "gender", "Gender Distribution")
    compare_barplots(datasets, "age", "Age Distribution", sort_index=True)
    compare_barplots(datasets, "occupation", "Occupation Distribution")


if __name__ == "__main__":
    main()