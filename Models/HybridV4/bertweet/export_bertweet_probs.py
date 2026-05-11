import argparse
import json
import os
import sys
from typing import Dict, List, Optional


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
    bertweet_v3_predictions_dir,
    bertweet_v34_predictions_dir,
    hybrid_v4_bertweet_probs_dir,
)

from Models.BERTweet.config_bertweet_model import LABEL_ORDERS as V3_LABEL_ORDERS
from Models.BERTweetV34.config_bertweet_v34_model import LABEL_ORDERS as V34_LABEL_ORDERS


TARGETS = ["occupation", "gender", "birthyear"]


def ensure_dirs():
    os.makedirs(hybrid_v4_bertweet_probs_dir, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def prediction_path(predictions_dir: str, target: str, split: str) -> str:
    return os.path.join(
        predictions_dir,
        f"{target}_{split}_predictions.json",
    )


def load_predictions(
    predictions_dir: str,
    target: str,
    split: str,
    source_name: str,
) -> Dict[str, dict]:
    path = prediction_path(predictions_dir, target, split)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {source_name} predictions for target={target}, split={split}: {path}"
        )

    rows = load_json(path)

    result = {}
    for row in rows:
        cid = str(row["celebrity_id"])
        result[cid] = row

    return result


def validate_label_order(target: str):
    v3_labels = V3_LABEL_ORDERS[target]
    v34_labels = V34_LABEL_ORDERS[target]

    if v3_labels != v34_labels:
        raise ValueError(
            f"Label order mismatch for target={target}: "
            f"V3={v3_labels}, V3.4={v34_labels}"
        )

    return v3_labels


def merge_target_split(target: str, split: str, require_both: bool = True) -> List[dict]:
    labels = validate_label_order(target)

    v3_rows: Optional[Dict[str, dict]] = None
    v34_rows: Optional[Dict[str, dict]] = None

    try:
        v3_rows = load_predictions(
            predictions_dir=bertweet_v3_predictions_dir,
            target=target,
            split=split,
            source_name="BERTweet V3",
        )
    except FileNotFoundError:
        if require_both:
            raise

    try:
        v34_rows = load_predictions(
            predictions_dir=bertweet_v34_predictions_dir,
            target=target,
            split=split,
            source_name="BERTweet V3.4",
        )
    except FileNotFoundError:
        if require_both:
            raise

    ids = set()
    if v3_rows is not None:
        ids.update(v3_rows.keys())
    if v34_rows is not None:
        ids.update(v34_rows.keys())

    merged = []

    for cid in sorted(ids):
        v3 = v3_rows.get(cid) if v3_rows is not None else None
        v34 = v34_rows.get(cid) if v34_rows is not None else None

        if require_both and (v3 is None or v34 is None):
            raise ValueError(
                f"Missing paired prediction for celebrity_id={cid}, "
                f"target={target}, split={split}: "
                f"has_v3={v3 is not None}, has_v34={v34 is not None}"
            )

        true_label = None
        if v3 is not None:
            true_label = v3["true_label"]
        if v34 is not None:
            if true_label is not None and true_label != v34["true_label"]:
                raise ValueError(
                    f"True-label mismatch for celebrity_id={cid}, target={target}: "
                    f"V3={true_label}, V3.4={v34['true_label']}"
                )
            true_label = v34["true_label"]

        merged.append({
            "celebrity_id": cid,
            "target": target,
            "split": split,
            "true_label": true_label,
            "labels": labels,
            "bertweet_v3_probabilities": v3["probabilities"] if v3 is not None else None,
            "bertweet_v3_pred_label": v3["pred_label"] if v3 is not None else None,
            "bertweet_v34_probabilities": v34["probabilities"] if v34 is not None else None,
            "bertweet_v34_pred_label": v34["pred_label"] if v34 is not None else None,
        })

    return merged


def export_one(target: str, split: str, require_both: bool):
    print(f"\n========== HybridV4 BERTweet export: target={target}, split={split} ==========")

    rows = merge_target_split(
        target=target,
        split=split,
        require_both=require_both,
    )

    output_path = os.path.join(
        hybrid_v4_bertweet_probs_dir,
        f"{target}_{split}_bertweet_probs.json",
    )

    save_json(rows, output_path)

    print(f"[OK] Saved: {output_path}")
    print(f"[INFO] Rows: {len(rows)}")


def resolve_targets(target: str) -> List[str]:
    if target == "all":
        return TARGETS
    return [target]


def resolve_splits(split: str) -> List[str]:
    if split == "all":
        return ["val", "test"]
    return [split]


def main():
    parser = argparse.ArgumentParser(
        description="Export BERTweet V3/V3.4 probabilities into HybridV4 format."
    )
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear", "all"],
        default="all",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "all"],
        default="test",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow V3 or V3.4 predictions to be missing. Default requires both.",
    )

    args = parser.parse_args()

    ensure_dirs()

    for target in resolve_targets(args.target):
        for split in resolve_splits(args.split):
            export_one(
                target=target,
                split=split,
                require_both=not args.allow_missing,
            )


if __name__ == "__main__":
    main()