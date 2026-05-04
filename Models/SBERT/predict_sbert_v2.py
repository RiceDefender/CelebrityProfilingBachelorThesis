import argparse
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np

from _constants import (
    sbert_test_vectors_path,
    sbert_v2_checkpoints_dir,
    sbert_v2_predictions_dir,
)

from Models.SBERT.config_sbert_model import TARGET_LABEL


BIRTHYEAR_BUCKETS = ["1994", "1985", "1975", "1963", "1947"]


def resolve_targets(target: str) -> List[str]:
    if target == "all":
        return ["occupation", "gender", "birthyear"]
    return [target]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_birthyear_to_bucket(year) -> str:
    year = int(year)
    bucket_years = [int(y) for y in BIRTHYEAR_BUCKETS]
    nearest = min(bucket_years, key=lambda b: abs(year - b))
    return str(nearest)


def load_checkpoint(target_label: str):
    path = os.path.join(
        sbert_v2_checkpoints_dir,
        f"{target_label}_logreg_voting.pkl",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def predict_one_target(target_label: str):
    print(f"\n========== SBERT V2 PREDICT TARGET: {target_label} ==========")

    checkpoint = load_checkpoint(target_label)

    clf = checkpoint["classifier"]
    scaler = checkpoint["scaler"]
    id_to_label: Dict[int, str] = checkpoint["id_to_label"]
    voting_strategy = checkpoint.get("voting_strategy", "soft")

    rows = load_json(sbert_test_vectors_path)

    grouped_probs = defaultdict(list)
    grouped_preds = defaultdict(list)
    grouped_true = {}

    for row in rows:
        cid = str(row["celebrity_id"])
        emb = np.asarray(row["embedding"], dtype=np.float32).reshape(1, -1)

        if scaler is not None:
            emb = scaler.transform(emb)

        probs = clf.predict_proba(emb)[0]
        pred_id = int(np.argmax(probs))

        grouped_probs[cid].append(probs)
        grouped_preds[cid].append(pred_id)

        if target_label in row:
            true_label = row[target_label]
            if target_label == "birthyear":
                true_label = map_birthyear_to_bucket(true_label)
            grouped_true[cid] = str(true_label)

    predictions = []

    for cid in sorted(grouped_probs.keys()):
        probs_stack = np.stack(grouped_probs[cid], axis=0)

        if voting_strategy == "soft":
            final_probs = probs_stack.mean(axis=0)
            final_pred_id = int(np.argmax(final_probs))
        elif voting_strategy == "hard":
            counts = np.bincount(
                grouped_preds[cid],
                minlength=len(id_to_label),
            )
            final_probs = counts / counts.sum()
            final_pred_id = int(np.argmax(counts))
        else:
            raise ValueError(f"Unsupported voting strategy: {voting_strategy}")

        item = {
            "celebrity_id": cid,
            "pred_label": id_to_label[final_pred_id],
            "probabilities": final_probs.tolist(),
            "num_chunks": len(grouped_probs[cid]),
            "version": "sbert_v2",
            "classifier_type": "logistic_regression",
            "voting_strategy": voting_strategy,
        }

        if cid in grouped_true:
            item["true_label"] = grouped_true[cid]

        predictions.append(item)

    os.makedirs(sbert_v2_predictions_dir, exist_ok=True)

    out_path = os.path.join(
        sbert_v2_predictions_dir,
        f"{target_label}_test_predictions.json",
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved predictions: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict SBERT V2 Logistic Regression with voting")
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear", "all"],
        default=TARGET_LABEL,
    )

    args = parser.parse_args()

    for target in resolve_targets(args.target):
        predict_one_target(target)


if __name__ == "__main__":
    main()