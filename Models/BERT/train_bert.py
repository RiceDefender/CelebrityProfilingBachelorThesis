import os
import sys
import json
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoTokenizer,
)

from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------
# Project root / imports
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from _constants import (
    bert_train_tokenized_path,
    bert_test_tokenized_path,
    bert_output_dir,
)
from Preprocessing.tokenizers.bert.config_bert import MODEL_NAME
from Models.BERT.config_bert_model import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    RANDOM_SEED,
    CHUNK_AGGREGATION_METHOD,
)

from Preprocessing.normalize import TWEET_SEP, FOLLOWER_SEP, URL_TOKEN, MENTION_TOKEN


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# JSON loader
# ---------------------------------------------------------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# Target configs
# ---------------------------------------------------------
SUPPORTED_TARGETS = ["occupation", "gender", "birthyear"]

BIRTHYEAR_BUCKETS = ["1994", "1985", "1975", "1963", "1947"]

FIXED_LABELS = {
    "occupation": ["sports", "performer", "creator", "politics"],
    "gender": ["male", "female"],
    "birthyear": BIRTHYEAR_BUCKETS,
}

def map_birthyear_to_bucket(year):
    """
    Maps exact birthyear to the 5 PAN-style representative birthyear buckets.
    Buckets are chosen by nearest representative year.
    """
    year = int(year)
    bucket_years = [int(y) for y in BIRTHYEAR_BUCKETS]
    nearest = min(bucket_years, key=lambda b: abs(year - b))
    return str(nearest)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class TokenizedChunkDataset(Dataset):
    def __init__(self, rows, target_label, label2id):
        self.rows = rows
        self.target_label = target_label
        self.label2id = label2id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        label_value = row[self.target_label]
        if self.target_label == "birthyear":
            label_value = map_birthyear_to_bucket(label_value)

        item = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.label2id[label_value], dtype=torch.long),
        }
        return item


# ---------------------------------------------------------
# Split
# ---------------------------------------------------------
def split_train_val(rows, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    val_size = max(1, int(len(rows) * val_ratio))
    val_indices = set(indices[:val_size])

    train_rows = [rows[i] for i in range(len(rows)) if i not in val_indices]
    val_rows = [rows[i] for i in range(len(rows)) if i in val_indices]

    return train_rows, val_rows


# ---------------------------------------------------------
# Label mapping
# ---------------------------------------------------------
def build_label_mapping(rows, target_label):
    if target_label in FIXED_LABELS:
        labels = FIXED_LABELS[target_label]
    else:
        labels = sorted({str(r[target_label]) for r in rows})

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


# ---------------------------------------------------------
# Metrics for Trainer (chunk-level)
# ---------------------------------------------------------
def compute_chunk_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


# ---------------------------------------------------------
# Celebrity-level aggregation
# ---------------------------------------------------------
def aggregate_by_celebrity(rows, logits, label2id, id2label, target_label, method="mean_logits"):
    grouped_logits = defaultdict(list)
    grouped_true = {}

    for row, logit in zip(rows, logits):
        cid = row["celebrity_id"]
        grouped_logits[cid].append(logit)

        true_value = row[target_label]
        if target_label == "birthyear":
            true_value = map_birthyear_to_bucket(true_value)
        grouped_true[cid] = label2id[true_value]

    y_true = []
    y_pred = []
    detailed_predictions = []

    for cid, logit_list in grouped_logits.items():
        stacked = np.stack(logit_list, axis=0)

        if method == "mean_logits":
            agg_logits = stacked.mean(axis=0)
            pred_id = int(np.argmax(agg_logits))
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

        true_id = grouped_true[cid]

        y_true.append(true_id)
        y_pred.append(pred_id)

        detailed_predictions.append({
            "celebrity_id": cid,
            "true_label": id2label[true_id],
            "pred_label": id2label[pred_id],
            "num_chunks": len(logit_list),
        })

    metrics = {
        "celebrity_accuracy": accuracy_score(y_true, y_pred),
        "celebrity_macro_f1": f1_score(y_true, y_pred, average="macro"),
        "num_celebrities": len(y_true),
    }

    return metrics, detailed_predictions


# ---------------------------------------------------------
# Save helpers
# ---------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------
# Train one target
# ---------------------------------------------------------
def train_one_target(target_label, train_rows, test_rows):
    print(f"\n========== TARGET: {target_label} ==========")

    # build label mapping only from training data
    label2id, id2label = build_label_mapping(train_rows, target_label)
    num_labels = len(label2id)

    print(f"[{target_label}] Num labels: {num_labels}")
    print(f"[{target_label}] Labels: {label2id}")

    train_split, val_split = split_train_val(train_rows, val_ratio=0.1, seed=RANDOM_SEED)

    train_dataset = TokenizedChunkDataset(train_split, target_label, label2id)
    val_dataset = TokenizedChunkDataset(val_split, target_label, label2id)
    test_dataset = TokenizedChunkDataset(test_rows, target_label, label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                URL_TOKEN,
                MENTION_TOKEN,
                TWEET_SEP,
                FOLLOWER_SEP,
            ]
        }
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    model.resize_token_embeddings(len(tokenizer))

    target_output_dir = os.path.join(bert_output_dir, target_label)
    ckpt_dir = os.path.join(target_output_dir, "checkpoints_v1")
    pred_dir = os.path.join(target_output_dir, "predictions_v1")
    metrics_dir = os.path.join(target_output_dir, "metrics_v1")

    ensure_dir(ckpt_dir)
    ensure_dir(pred_dir)
    ensure_dir(metrics_dir)

    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        save_total_limit=2,
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_chunk_metrics,
    )

    trainer.train()

    # ---- validation chunk-level
    val_output = trainer.predict(val_dataset)
    val_chunk_metrics = compute_chunk_metrics((val_output.predictions, val_output.label_ids))

    # ---- validation celebrity-level
    val_celeb_metrics, val_celeb_preds = aggregate_by_celebrity(
        val_split,
        val_output.predictions,
        label2id,
        id2label,
        target_label,
        method=CHUNK_AGGREGATION_METHOD,
    )

    # ---- test chunk-level
    test_output = trainer.predict(test_dataset)
    test_chunk_metrics = compute_chunk_metrics((test_output.predictions, test_output.label_ids))

    # ---- test celebrity-level
    test_celeb_metrics, test_celeb_preds = aggregate_by_celebrity(
        test_rows,
        test_output.predictions,
        label2id,
        id2label,
        target_label,
        method=CHUNK_AGGREGATION_METHOD,
    )

    all_metrics = {
        "target": target_label,
        "label2id": label2id,
        "id2label": id2label,
        "chunk_level": {
            "validation": val_chunk_metrics,
            "test": test_chunk_metrics,
        },
        "celebrity_level": {
            "validation": val_celeb_metrics,
            "test": test_celeb_metrics,
        },
        "aggregation_method": CHUNK_AGGREGATION_METHOD,
        "train_examples": len(train_split),
        "val_examples": len(val_split),
        "test_examples": len(test_rows),
    }

    save_json(all_metrics, os.path.join(metrics_dir, f"{target_label}_metrics.json"))
    save_json(val_celeb_preds, os.path.join(pred_dir, f"{target_label}_val_celeb_predictions.json"))
    save_json(test_celeb_preds, os.path.join(pred_dir, f"{target_label}_test_celeb_predictions.json"))

    print(f"[{target_label}] Validation chunk metrics_v1: {val_chunk_metrics}")
    print(f"[{target_label}] Validation celebrity metrics_v1: {val_celeb_metrics}")
    print(f"[{target_label}] Test chunk metrics_v1: {test_chunk_metrics}")
    print(f"[{target_label}] Test celebrity metrics_v1: {test_celeb_metrics}")

    return all_metrics


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train BERT MVP for celebrity profiling")
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear", "all"],
        default="occupation",
        help="Which target to train"
    )
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    train_rows = load_json(bert_train_tokenized_path)
    test_rows = load_json(bert_test_tokenized_path)

    print(f"Loaded train rows: {len(train_rows)}")
    print(f"Loaded test rows: {len(test_rows)}")

    if args.target == "all":
        results = {}
        for target_label in SUPPORTED_TARGETS:
            results[target_label] = train_one_target(target_label, train_rows, test_rows)

        save_json(results, os.path.join(bert_output_dir, "all_targets_summary.json"))
    else:
        train_one_target(args.target, train_rows, test_rows)


if __name__ == "__main__":
    main()
