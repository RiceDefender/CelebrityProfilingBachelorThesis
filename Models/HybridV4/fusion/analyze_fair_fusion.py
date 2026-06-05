import argparse
import json
import os
from collections import Counter
from contextlib import redirect_stdout

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_analysis(rows):
    labels = rows[0]["labels"]
    y_true = [r["true_label"] for r in rows]
    y_pred = [r["fusion_pred_label"] for r in rows]

    print("\n========== Classification report ==========")
    print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))

    print("\n========== Confusion matrix ==========")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("labels:", labels)
    print(cm)

    print("\n========== Mistakes by true -> pred ==========")
    mistakes = Counter()

    for r in rows:
        if r["true_label"] != r["fusion_pred_label"]:
            mistakes[(r["true_label"], r["fusion_pred_label"])] += 1

    for (true_label, pred_label), count in mistakes.most_common():
        print(f"{true_label:10s} -> {pred_label:10s}: {count}")

    print("\n========== Model agreement ==========")
    agreement_counter = Counter()

    for r in rows:
        base_preds = [
            r["bertweet_v3_pred_label"],
            r["bertweet_v34_pred_label"],
            r["sparse_feature_pred_label"],
        ]

        if len(set(base_preds)) == 1:
            key = f"all_base_agree_on_{base_preds[0]}"
        elif r["bertweet_v34_pred_label"] == r["fusion_pred_label"]:
            key = "fusion_follows_v34"
        elif r["sparse_feature_pred_label"] == r["fusion_pred_label"]:
            key = "fusion_follows_sparse"
        elif r["bertweet_v3_pred_label"] == r["fusion_pred_label"]:
            key = "fusion_follows_v3"
        else:
            key = "fusion_other"

        agreement_counter[key] += 1

    for key, count in agreement_counter.most_common():
        print(f"{key:30s}: {count}")

    print("\n========== Creator boost diagnostics ==========")

    creator_rows = [r for r in rows if r["fusion_pred_label"] == "creator"]
    true_creator_rows = [r for r in rows if r["true_label"] == "creator"]
    false_creator_rows = [
        r for r in rows
        if r["fusion_pred_label"] == "creator" and r["true_label"] != "creator"
    ]

    print(f"Predicted creator:       {len(creator_rows)}")
    print(f"True creator examples:   {len(true_creator_rows)}")
    print(f"False creator positives: {len(false_creator_rows)}")

    for name, group in [
        ("all predicted creator", creator_rows),
        ("true creators", true_creator_rows),
        ("false creator positives", false_creator_rows),
    ]:
        signals = [
            r["creator_signal"]
            for r in group
            if r.get("creator_signal") is not None
        ]

        if signals:
            print(
                f"{name:24s}: "
                f"mean={np.mean(signals):.4f} "
                f"median={np.median(signals):.4f} "
                f"min={np.min(signals):.4f} "
                f"max={np.max(signals):.4f}"
            )

    print("\n========== High-confidence wrong predictions ==========")

    wrong_rows = [r for r in rows if r["true_label"] != r["fusion_pred_label"]]

    def confidence(row):
        return max(row["fusion_probabilities"])

    wrong_rows = sorted(wrong_rows, key=confidence, reverse=True)

    for r in wrong_rows[:30]:
        probs = dict(zip(r["labels"], r["fusion_probabilities"]))
        print(
            f"id={r['celebrity_id']} "
            f"true={r['true_label']} "
            f"pred={r['fusion_pred_label']} "
            f"conf={confidence(r):.4f} "
            f"probs={{{', '.join(f'{k}:{v:.3f}' for k, v in probs.items())}}} "
            f"v3={r['bertweet_v3_pred_label']} "
            f"v34={r['bertweet_v34_pred_label']} "
            f"sparse={r['sparse_feature_pred_label']} "
            f"creator_signal={r.get('creator_signal')}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = load_json(args.predictions)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                print_analysis(rows)

        print(f"[OK] Saved analysis report: {args.output}")

    print_analysis(rows)


if __name__ == "__main__":
    main()