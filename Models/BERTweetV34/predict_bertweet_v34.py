import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from _constants import (
    bertweet_v34_test_tokenized_path,
    bertweet_v34_checkpoints_dir,
    bertweet_v34_predictions_dir,
    bertweet_v34_age_bins_path,
)
from Models.BERTweetV34.config_bertweet_v34_model import (
    MODEL_NAME,
    VERSION,
    LABEL_ORDERS,
    RANDOM_SEED,
    MAX_PREDICT_CHUNKS_PER_CELEB,
    PREDICT_BATCH_SIZE,
)
from Models.BERTweetV34.age_bins_v34 import load_age_bins, map_year_to_age_bin

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dirs():
    os.makedirs(bertweet_v34_predictions_dir, exist_ok=True)


def load_json_or_ndjson(path: str):
    print(f"[INFO] Loading: {path}")
    if path.endswith((".ndjson", ".jsonl")):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def map_birthyear_to_bucket(year) -> str:
    year = int(year)
    buckets = [int(x) for x in LABEL_ORDERS["birthyear"]]
    nearest = min(buckets, key=lambda b: abs(year - b))
    return str(nearest)


def maybe_load_age_bins(target: str):
    if target == "birthyear_8range":
        bins = load_age_bins(bertweet_v34_age_bins_path)
        print(f"[INFO] Loaded age bins: {bertweet_v34_age_bins_path}")
        return bins
    return None


def get_label(row: dict, target: str, age_bins=None) -> str:
    if target == "creator_binary":
        return "creator" if row["occupation"] == "creator" else "not_creator"
    if target == "occupation_3class":
        return str(row["occupation"])
    if target == "birthyear":
        return map_birthyear_to_bucket(row["birthyear"])
    if target == "birthyear_8range":
        return map_year_to_age_bin(row["birthyear"], age_bins)
    return str(row[target])


def build_label_maps(target: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = LABEL_ORDERS[target]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def group_rows_by_celebrity(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["celebrity_id"])].append(row)
    return grouped


def select_evenly_spaced(rows, max_chunks):
    if max_chunks is None or len(rows) <= max_chunks:
        return rows
    indices = np.linspace(0, len(rows) - 1, num=max_chunks, dtype=int)
    return [rows[int(i)] for i in indices]


def prepare_prediction_rows(rows, max_chunks_per_celeb):
    grouped = group_rows_by_celebrity(rows)
    selected = []
    for cid in sorted(grouped.keys()):
        celeb_rows = sorted(grouped[cid], key=lambda r: int(r.get("chunk_id", 0)))
        selected.extend(select_evenly_spaced(celeb_rows, max_chunks_per_celeb))
    return selected


class BERTweetPredictDataset(Dataset):
    def __init__(self, rows):
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


def softmax_np(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def predict_logits(model, dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    all_logits = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting chunks"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits.detach().cpu().numpy()
            all_logits.append(logits)
    return np.concatenate(all_logits, axis=0)


def aggregate_predictions(rows, logits, target, id_to_label, age_bins=None):
    probs = softmax_np(logits)
    grouped_probs = defaultdict(list)
    grouped_true = {}
    for row, prob in zip(rows, probs):
        cid = str(row["celebrity_id"])
        grouped_probs[cid].append(prob)
        grouped_true[cid] = get_label(row, target, age_bins=age_bins)
    predictions = []
    for cid in sorted(grouped_probs.keys()):
        prob_stack = np.stack(grouped_probs[cid], axis=0)
        final_probs = prob_stack.mean(axis=0)
        pred_id = int(np.argmax(final_probs))
        predictions.append({
            "celebrity_id": cid,
            "true_label": grouped_true[cid],
            "pred_label": id_to_label[pred_id],
            "probabilities": final_probs.tolist(),
            "num_chunks": int(len(grouped_probs[cid])),
            "version": VERSION,
            "target": target,
            "model_name": MODEL_NAME,
            "checkpoint": "final_model",
            "voting_strategy": "soft",
            "chunk_selection": "evenly_spaced",
            "max_predict_chunks_per_celebrity": MAX_PREDICT_CHUNKS_PER_CELEB,
        })
    return predictions


def predict_one_target(target: str):
    print(f"\n========== BERTweet V3.4 TEST PREDICT TARGET: {target} ==========")
    _, id_to_label = build_label_maps(target)
    age_bins = maybe_load_age_bins(target)
    test_rows_all = load_json_or_ndjson(bertweet_v34_test_tokenized_path)
    if target == "occupation_3class":
        test_rows_all = [r for r in test_rows_all if r["occupation"] != "creator"]
    test_rows = prepare_prediction_rows(test_rows_all, MAX_PREDICT_CHUNKS_PER_CELEB)
    print(f"[INFO] Test rows total:    {len(test_rows_all)}")
    print(f"[INFO] Test rows selected: {len(test_rows)}")
    print(f"[INFO] Test celebrities:   {len(set(str(r['celebrity_id']) for r in test_rows))}")
    model_dir = os.path.join(bertweet_v34_checkpoints_dir, target, "final_model")
    print(f"[INFO] Loading model: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, normalization=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)
    logits = predict_logits(model, BERTweetPredictDataset(test_rows), PREDICT_BATCH_SIZE)
    predictions = aggregate_predictions(test_rows, logits, target, id_to_label, age_bins=age_bins)
    output_path = os.path.join(bertweet_v34_predictions_dir, f"{target}_test_predictions.json")
    save_json(predictions, output_path)
    print(f"[OK] Saved predictions: {output_path}")


def resolve_targets(target):
    if target == "all":
        return ["occupation", "gender", "birthyear"]
    if target == "all_with_age8":
        return ["occupation", "gender", "birthyear", "birthyear_8range"]
    return [target]


def main():
    parser = argparse.ArgumentParser(description="Predict BERTweet V3.4 test labels")
    parser.add_argument("--target", choices=["occupation", "gender", "birthyear", "birthyear_8range", "creator_binary", "occupation_3class", "all", "all_with_age8"], default="all")
    args = parser.parse_args()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    ensure_dirs()
    for target in resolve_targets(args.target):
        predict_one_target(target)


if __name__ == "__main__":
    main()
