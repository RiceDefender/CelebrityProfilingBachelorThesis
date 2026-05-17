import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from _constants import bertweet_test_tokenized_path

from Models.BERTweetV37.config_bertweet_v37 import (
    MODEL_NAME,
    LABEL_ORDERS,
    RANDOM_SEED,
    V37_TARGETS,
)


# -------------------------------------------------------------------
# V3.7 output dirs
# -------------------------------------------------------------------
BERTWEET_V37_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "bertweet_v37")
BERTWEET_V37_CHECKPOINTS_DIR = os.path.join(BERTWEET_V37_OUTPUT_DIR, "checkpoints")
BERTWEET_V37_PREDICTIONS_DIR = os.path.join(BERTWEET_V37_OUTPUT_DIR, "predictions")


MAX_PREDICT_CHUNKS_PER_CELEB = 128
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_dirs():
    os.makedirs(BERTWEET_V37_PREDICTIONS_DIR, exist_ok=True)


def load_json_or_ndjson(path: str):
    print(f"[INFO] Loading: {path}")

    if path.endswith(".ndjson") or path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid NDJSON in {path} at line {line_idx}: {e}"
                    ) from e
        return rows

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_group3_label(occupation: str) -> str:
    if occupation in {"creator", "performer"}:
        return "entertainment"
    if occupation in {"sports", "politics"}:
        return occupation
    raise ValueError(f"Unsupported occupation label: {occupation}")


def get_true_label_for_target(row: dict, target: str) -> str:
    occupation = str(row["occupation"])

    if target == "occupation_group3":
        return get_group3_label(occupation)

    if target == "creator_performer":
        # During inference the specialist must be available for every celebrity,
        # because the stage-1 model decides whether it is used. For non-entertainment
        # examples this value is only metadata, not used as a target metric.
        if occupation in {"creator", "performer"}:
            return occupation
        return "not_creator_performer"

    if target == "occupation":
        return occupation

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


def select_evenly_spaced(rows: List[dict], max_chunks: int) -> List[dict]:
    if max_chunks is None or len(rows) <= max_chunks:
        return rows

    indices = np.linspace(0, len(rows) - 1, num=max_chunks, dtype=int)
    return [rows[int(i)] for i in indices]


def prepare_prediction_rows(rows: List[dict], max_chunks_per_celeb: int) -> List[dict]:
    grouped = group_rows_by_celebrity(rows)

    selected = []
    for cid in sorted(grouped.keys()):
        celeb_rows = sorted(grouped[cid], key=lambda r: int(r.get("chunk_id", 0)))
        selected.extend(select_evenly_spaced(celeb_rows, max_chunks_per_celeb))

    return selected


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
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
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def predict_logits(model, dataset: Dataset, batch_size: int) -> np.ndarray:
    loader = torch.utils.data.DataLoader(
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
):
    probs = softmax_np(logits)

    grouped_probs = defaultdict(list)
    grouped_true_label = {}
    grouped_true_occupation = {}

    for row, prob in zip(rows, probs):
        cid = str(row["celebrity_id"])
        grouped_probs[cid].append(prob)
        grouped_true_label[cid] = get_true_label_for_target(row, target)
        grouped_true_occupation[cid] = str(row["occupation"])

    predictions = []
    for cid in sorted(grouped_probs.keys()):
        prob_stack = np.stack(grouped_probs[cid], axis=0)
        final_probs = prob_stack.mean(axis=0)

        pred_id = int(np.argmax(final_probs))
        pred_label = id_to_label[pred_id]

        prob_dict = {
            id_to_label[i]: float(final_probs[i])
            for i in range(len(id_to_label))
        }

        predictions.append({
            "celebrity_id": cid,
            "true_occupation": grouped_true_occupation[cid],
            "true_label": grouped_true_label[cid],
            "pred_label": pred_label,
            "probabilities": final_probs.tolist(),
            "probability_by_label": prob_dict,
            "labels": [id_to_label[i] for i in range(len(id_to_label))],
            "num_chunks": int(len(grouped_probs[cid])),
            "version": "bertweet_v37",
            "model_name": MODEL_NAME,
            "target": target,
            "checkpoint": "final_model",
            "voting_strategy": "soft",
            "chunk_selection": "evenly_spaced",
            "max_predict_chunks_per_celebrity": MAX_PREDICT_CHUNKS_PER_CELEB,
        })

    return predictions


def predict_one_target(target: str):
    print(f"\n========== BERTweet V3.7 TEST PREDICT TARGET: {target} ==========")

    label_to_id, id_to_label = build_label_maps(target)

    test_rows_all = load_json_or_ndjson(bertweet_test_tokenized_path)
    test_rows = prepare_prediction_rows(
        test_rows_all,
        max_chunks_per_celeb=MAX_PREDICT_CHUNKS_PER_CELEB,
    )

    print(f"[INFO] Test rows total:    {len(test_rows_all)}")
    print(f"[INFO] Test rows selected: {len(test_rows)}")
    print(f"[INFO] Test celebrities:   {len(set(str(r['celebrity_id']) for r in test_rows))}")
    print(f"[INFO] Labels:             {label_to_id}")

    model_dir = os.path.join(
        BERTWEET_V37_CHECKPOINTS_DIR,
        target,
        "final_model",
    )

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Run training first for target={target}."
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

    dataset = BERTweetPredictDataset(test_rows)
    logits = predict_logits(model, dataset, BATCH_SIZE)

    predictions = aggregate_predictions(
        rows=test_rows,
        logits=logits,
        target=target,
        id_to_label=id_to_label,
    )

    output_path = os.path.join(
        BERTWEET_V37_PREDICTIONS_DIR,
        f"{target}_test_predictions.json",
    )

    save_json(predictions, output_path)

    print(f"[OK] Saved predictions: {output_path}")
    print(f"[INFO] Num celebrity predictions: {len(predictions)}")


def resolve_targets(target: str):
    if target == "v37_occupation":
        return V37_TARGETS
    return [target]


def main():
    parser = argparse.ArgumentParser(description="Predict BERTweet V3.7 hierarchical occupation models")
    parser.add_argument(
        "--target",
        choices=["occupation", "occupation_group3", "creator_performer", "v37_occupation"],
        default="v37_occupation",
    )

    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ensure_dirs()

    for target in resolve_targets(args.target):
        predict_one_target(target)


if __name__ == "__main__":
    main()
