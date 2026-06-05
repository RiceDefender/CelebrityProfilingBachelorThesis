import json
import os
import sys
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from _constants import (
    bertweet_v3_metrics_dir,
    bertweet_v3_output_dir,
)


TARGETS = ["occupation", "gender", "birthyear"]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    rows: List[Dict] = []

    for target in TARGETS:
        path = os.path.join(bertweet_v3_metrics_dir, f"{target}_metrics.json")

        if not os.path.exists(path):
            print(f"[WARN] Missing: {path}")
            continue

        metrics = load_json(path)

        rows.append({
            "model": "bertweet_v3",
            "target": target,
            "val_accuracy": metrics.get("val_accuracy"),
            "val_macro_f1": metrics.get("val_macro_f1"),
            "num_val_celebrities": metrics.get("num_val_celebrities"),
            "num_val_chunks": metrics.get("num_val_chunks"),
            "max_train_chunks_per_celebrity": metrics.get("max_train_chunks_per_celebrity"),
            "max_val_chunks_per_celebrity": metrics.get("max_val_chunks_per_celebrity"),
            "model_name": metrics.get("model_name"),
            "voting_strategy": metrics.get("voting_strategy"),
        })

    df = pd.DataFrame(rows)

    output_dir = os.path.join(bertweet_v3_output_dir, "comparison")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "bertweet_v3_val_summary.csv")
    json_path = os.path.join(output_dir, "bertweet_v3_val_summary.json")
    md_path = os.path.join(output_dir, "bertweet_v3_val_summary.md")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BERTweet V3 Validation Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print(df)
    print(f"[OK] Saved CSV:  {csv_path}")
    print(f"[OK] Saved JSON: {json_path}")
    print(f"[OK] Saved MD:   {md_path}")


if __name__ == "__main__":
    main()