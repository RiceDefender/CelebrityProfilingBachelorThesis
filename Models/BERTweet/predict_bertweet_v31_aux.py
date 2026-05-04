import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from _constants import (
    bertweet_train_tokenized_path,
    bertweet_test_tokenized_path,
    bertweet_v3_checkpoints_dir,
    bertweet_v3_predictions_dir,
)

from Models.BERTweet.config_bertweet_model import (
    LABEL_ORDERS,
    RANDOM_SEED,
    VAL_RATIO,
)


MAX_PREDICT_CHUNKS_PER_CELEB = 128
MAX_VAL_CHUNKS_PER_CELEB = 64
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


TARGETS = ["occupation","creator_binary", "occupation_3class"]


def ensure_dirs():
    os.makedirs(bertweet_v3_predictions_dir, exist_ok=True)


def load_json_or_ndjson(path: str):
    print(f"[INFO] Loading: {path}")

    if path.endswith(".ndjson") or path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_label(row: dict, target: str) -> str:
    if target == "occupation":
        return str(row["occupation"])

    if target == "creator_binary":
        return "creator" if row["occupation"] == "creator" else "not_creator"

    if target == "occupation_3class":
        # Important:
        # The 3-class model can predict only sports/performer/politics,
        # but true_label remains the original 4-class occupation for gating evaluation.
        return str(row["occupation"])

    raise ValueError(f"Unsupported target: {target}")


def build_label_maps(target: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = LABEL_ORDERS[target]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def group_rows_by_celebrity(rows: List[dict]):
    grouped = defaultdict(list)

    for row in rows:
        cid = str(row["celebrity_id"])
        grouped[cid].append(row)

    return grouped


def celebrity_label_for_stratify(rows: List[dict]):
    grouped = group_rows_by_celebrity(rows)
    celebrity_to_label = {}

    for cid, celeb_rows in grouped.items():
        celebrity_to_label[cid] = celeb_rows[0]["occupation"]

    return grouped, celebrity_to_label


def get_val_celebrity_ids(rows: List[dict]) -> List[str]:
    """
    Recreates the exact celebrity-level train/val split used by train_bertweet.py
    for the 4-class occupation target.
    This is important so V3.1 threshold tuning uses the same Val split.
    """
    grouped, celebrity_to_label = celebrity_label_for_stratify(rows)

    celebrity_ids = sorted(grouped.keys())
    labels = [celebrity_to_label[cid] for cid in celebrity_ids]

    _, val_ids = train_test_split(
        celebrity_ids,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    return list(val_ids)


def select_evenly_spaced(rows: List[dict], max_chunks: int) -> List[dict]:
    if max_chunks is None or len(rows) <= max_chunks:
        return rows

    indices = np.linspace(0, len(rows) - 1, num=max_chunks, dtype=int)
    return [rows[int(i)] for i in indices]


def prepare_rows_for_split(
    rows: List[dict],
    split_name: str,
    max_chunks_per_celeb: int,
) -> List[dict]:
    grouped = group_rows_by_celebrity(rows)

    selected = []

    for cid in sorted(grouped.keys()):
        celeb_rows = sorted(grouped[cid], key=lambda r: int(r.get("chunk_id", 0)))

        # Validation originally used random sampling in training.
        # For V3.1 thresholding, we use deterministic evenly spaced chunks.
        # This is fine as long as all compared aux models use the same rows.
        selected.extend(select_evenly_spaced(celeb_rows, max_chunks_per_celeb))

    print(f"[INFO] Prepared {split_name} rows: {len(selected)}")
    print(f"[INFO] Prepared {split_name} celebrities: {len(set(str(r['celebrity_id']) for r in selected))}")

    return selected


class BERTweetPredictDataset(Dataset):
    def __init__(self, rows: List[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
        }


def collate_batch(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
    }


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def predict_logits(model, dataset: Dataset, batch_size: int) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    all_logits = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting chunks"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            all_logits.append(logits)

    return np.concatenate(all_logits, axis=0)


def aggregate_predictions(
    rows: List[dict],
    logits: np.ndarray,
    target: str,
    id_to_label: Dict[int, str],
    split_name: str,
):
    probs = softmax_np(logits)

    grouped_probs = defaultdict(list)
    grouped_true = {}

    for row, prob in zip(rows, probs):
        cid = str(row["celebrity_id"])
        grouped_probs[cid].append(prob)
        grouped_true[cid] = get_label(row, target)

    predictions = []

    for cid in sorted(grouped_probs.keys()):
        prob_stack = np.stack(grouped_probs[cid], axis=0)
        final_probs = prob_stack.mean(axis=0)

        pred_id = int(np.argmax(final_probs))
        pred_label = id_to_label[pred_id]
        true_label = grouped_true[cid]

        predictions.append({
            "celebrity_id": cid,
            "true_label": true_label,
            "pred_label": pred_label,
            "probabilities": final_probs.tolist(),
            "num_chunks": int(len(grouped_probs[cid])),
            "version": "bertweet_v3_1",
            "target": target,
            "split": split_name,
            "model_name": "vinai/bertweet-base",
            "checkpoint": "final_model",
            "voting_strategy": "soft",
            "chunk_selection": "evenly_spaced",
            "max_chunks_per_celebrity": int(len(grouped_probs[cid])),
        })

    return predictions


def predict_target_split(target: str, rows: List[dict], split_name: str, output_suffix: str):
    print(f"\n========== V3.1 AUX PREDICT target={target} split={split_name} ==========")

    _, id_to_label = build_label_maps(target)

    model_dir = os.path.join(
        bertweet_v3_checkpoints_dir,
        target,
        "final_model",
    )

    print(f"[INFO] Loading model: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=False,
        normalization=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

    dataset = BERTweetPredictDataset(rows)
    logits = predict_logits(model, dataset, BATCH_SIZE)

    predictions = aggregate_predictions(
        rows=rows,
        logits=logits,
        target=target,
        id_to_label=id_to_label,
        split_name=split_name,
    )

    output_path = os.path.join(
        bertweet_v3_predictions_dir,
        f"{target}_{output_suffix}_predictions.json",
    )

    save_json(predictions, output_path)

    print(f"[OK] Saved predictions: {output_path}")
    print(f"[INFO] Num celebrity predictions: {len(predictions)}")


def main():
    parser = argparse.ArgumentParser(description="Predict V3.1 auxiliary BERTweet models")
    parser.add_argument(
        "--split",
        choices=["val", "test", "all"],
        default="all",
    )
    parser.add_argument(
        "--target",
        choices=["occupation", "creator_binary", "occupation_3class", "all"],
        default="all",
    )

    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    ensure_dirs()

    targets = TARGETS if args.target == "all" else [args.target]

    if args.split in ["val", "all"]:
        train_rows_all = load_json_or_ndjson(bertweet_train_tokenized_path)
        val_ids = set(get_val_celebrity_ids(train_rows_all))

        val_rows_all = [
            row for row in train_rows_all
            if str(row["celebrity_id"]) in val_ids
        ]

        val_rows = prepare_rows_for_split(
            rows=val_rows_all,
            split_name="val_all",
            max_chunks_per_celeb=MAX_VAL_CHUNKS_PER_CELEB,
        )

        for target in targets:
            predict_target_split(
                target=target,
                rows=val_rows,
                split_name="val_all",
                output_suffix="val_all",
            )

    if args.split in ["test", "all"]:
        test_rows_all = load_json_or_ndjson(bertweet_test_tokenized_path)

        test_rows = prepare_rows_for_split(
            rows=test_rows_all,
            split_name="test",
            max_chunks_per_celeb=MAX_PREDICT_CHUNKS_PER_CELEB,
        )

        for target in targets:
            if target == "occupation_3class":
                suffix = "test_all"
            elif target == "occupation":
                suffix = "test_all"
            else:
                suffix = "test"
            predict_target_split(
                target=target,
                rows=test_rows,
                split_name="test",
                output_suffix=suffix,
            )


if __name__ == "__main__":
    main()