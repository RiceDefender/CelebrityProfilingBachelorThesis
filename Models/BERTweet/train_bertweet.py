import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

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

from _constants import (
    bertweet_train_tokenized_path,
    bertweet_v3_output_dir,
    bertweet_v3_checkpoints_dir,
    bertweet_v3_logs_dir,
    bertweet_v3_predictions_dir,
    bertweet_v3_metrics_dir,
)

from Models.BERTweet.config_bertweet_model import (
    MODEL_NAME,
    TARGET_LABEL,
    RANDOM_SEED,
    VAL_RATIO,
    NUM_EPOCHS,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    USE_FP16,
    MAX_TRAIN_CHUNKS_PER_CELEB,
    MAX_VAL_CHUNKS_PER_CELEB,
    VOTING_STRATEGY,
    LABEL_ORDERS,
    CLASS_WEIGHT_BY_TARGET,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_dirs():
    os.makedirs(bertweet_v3_output_dir, exist_ok=True)
    os.makedirs(bertweet_v3_checkpoints_dir, exist_ok=True)
    os.makedirs(bertweet_v3_logs_dir, exist_ok=True)
    os.makedirs(bertweet_v3_predictions_dir, exist_ok=True)
    os.makedirs(bertweet_v3_metrics_dir, exist_ok=True)


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


def map_birthyear_to_bucket(year) -> str:
    year = int(year)
    buckets = [int(x) for x in LABEL_ORDERS["birthyear"]]
    nearest = min(buckets, key=lambda b: abs(year - b))
    return str(nearest)


def get_label(row: dict, target: str) -> str:
    label = row[target]
    if target == "birthyear":
        label = map_birthyear_to_bucket(label)
    return str(label)


def build_label_maps(target: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = LABEL_ORDERS[target]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def group_rows_by_celebrity(rows: List[dict], target: str):
    grouped = defaultdict(list)
    celebrity_to_label = {}

    for row in rows:
        cid = str(row["celebrity_id"])
        label = get_label(row, target)

        grouped[cid].append(row)
        celebrity_to_label[cid] = label

    return grouped, celebrity_to_label


def sample_rows_per_celebrity(
    grouped: Dict[str, List[dict]],
    celebrity_ids: List[str],
    max_chunks_per_celebrity: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    sampled_rows = []

    for cid in celebrity_ids:
        rows = grouped[cid]

        if max_chunks_per_celebrity is None or len(rows) <= max_chunks_per_celebrity:
            sampled_rows.extend(rows)
        else:
            sampled_rows.extend(rng.sample(rows, max_chunks_per_celebrity))

    return sampled_rows


def split_by_celebrity(
    grouped: Dict[str, List[dict]],
    celebrity_to_label: Dict[str, str],
    label_to_id: Dict[str, int],
):
    celebrity_ids = sorted(grouped.keys())
    y = [label_to_id[celebrity_to_label[cid]] for cid in celebrity_ids]

    train_ids, val_ids = train_test_split(
        celebrity_ids,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    return train_ids, val_ids


def build_class_weight_tensor(
    target: str,
    train_rows: List[dict],
    label_to_id: Dict[str, int],
):
    config = CLASS_WEIGHT_BY_TARGET.get(target, None)

    if config is None:
        return None

    num_labels = len(label_to_id)

    if config == "balanced":
        labels = [label_to_id[get_label(row, target)] for row in train_rows]
        counts = np.bincount(labels, minlength=num_labels).astype(np.float32)
        total = counts.sum()

        weights = total / (num_labels * np.maximum(counts, 1.0))
        return torch.tensor(weights, dtype=torch.float32)

    weights = np.ones(num_labels, dtype=np.float32)

    for label, weight in config.items():
        if label in label_to_id:
            weights[label_to_id[label]] = float(weight)

    return torch.tensor(weights, dtype=torch.float32)

def validate_tokenized_rows(rows: List[dict], tokenizer, model, max_print: int = 5):
    vocab_size = model.config.vocab_size
    tokenizer_size = len(tokenizer)
    max_position_embeddings = getattr(model.config, "max_position_embeddings", None)

    bad_rows = []
    max_seen_id = -1
    min_seen_id = 10**18
    max_seq_len = -1

    # RoBERTa-style models reserve position ids internally.
    # Effective safe length is usually max_position_embeddings - 2.
    max_safe_seq_len = None
    if max_position_embeddings is not None:
        max_safe_seq_len = max_position_embeddings - 2

    for idx, row in enumerate(rows):
        input_ids = row["input_ids"]

        row_max = max(input_ids)
        row_min = min(input_ids)
        seq_len = len(input_ids)

        max_seen_id = max(max_seen_id, row_max)
        min_seen_id = min(min_seen_id, row_min)
        max_seq_len = max(max_seq_len, seq_len)

        bad_vocab = row_min < 0 or row_max >= vocab_size
        bad_position = (
            max_safe_seq_len is not None
            and seq_len > max_safe_seq_len
        )

        if bad_vocab or bad_position:
            bad_rows.append({
                "row_index": idx,
                "celebrity_id": row.get("celebrity_id"),
                "chunk_id": row.get("chunk_id"),
                "min_id": row_min,
                "max_id": row_max,
                "seq_len": seq_len,
                "vocab_size": vocab_size,
                "tokenizer_size": tokenizer_size,
                "max_position_embeddings": max_position_embeddings,
                "max_safe_seq_len": max_safe_seq_len,
                "bad_vocab": bad_vocab,
                "bad_position": bad_position,
            })

            if len(bad_rows) >= max_print:
                break

    print(f"[INFO] Min input_id seen:       {min_seen_id}")
    print(f"[INFO] Max input_id seen:       {max_seen_id}")
    print(f"[INFO] Max sequence length:     {max_seq_len}")
    print(f"[INFO] Model vocab size:        {vocab_size}")
    print(f"[INFO] Tokenizer size:          {tokenizer_size}")
    print(f"[INFO] Max position embeddings: {max_position_embeddings}")
    print(f"[INFO] Max safe seq len:        {max_safe_seq_len}")

    if bad_rows:
        print("[ERROR] Found invalid tokenized rows:")
        for item in bad_rows:
            print(item)

        raise ValueError(
            "Invalid tokenized rows. Most likely sequence length exceeds "
            "BERTweet max_position_embeddings. Retokenize with a smaller MAX_LENGTH."
        )


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
class BERTweetChunkDataset(Dataset):
    def __init__(self, rows: List[dict], target: str, label_to_id: Dict[str, int]):
        self.rows = rows
        self.target = target
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        label = get_label(row, self.target)

        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.label_to_id[label], dtype=torch.long),
        }


# -------------------------------------------------------------------
# Weighted Trainer
# -------------------------------------------------------------------
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
def make_compute_metrics(id_to_label: Dict[int, str]):
    labels = [id_to_label[i] for i in range(len(id_to_label))]

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=1)

        y_true_labels = [id_to_label[int(i)] for i in y_true]
        y_pred_labels = [id_to_label[int(i)] for i in y_pred]

        return {
            "chunk_accuracy": accuracy_score(y_true_labels, y_pred_labels),
            "chunk_macro_f1": f1_score(
                y_true_labels,
                y_pred_labels,
                labels=labels,
                average="macro",
                zero_division=0,
            ),
        }

    return compute_metrics


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def aggregate_celebrity_predictions(
    rows: List[dict],
    logits: np.ndarray,
    target: str,
    id_to_label: Dict[int, str],
):
    probs = softmax_np(logits)

    grouped_probs = defaultdict(list)
    grouped_true = {}

    for row, prob in zip(rows, probs):
        cid = str(row["celebrity_id"])
        grouped_probs[cid].append(prob)
        grouped_true[cid] = get_label(row, target)

    predictions = []
    y_true = []
    y_pred = []

    for cid in sorted(grouped_probs.keys()):
        prob_stack = np.stack(grouped_probs[cid], axis=0)

        if VOTING_STRATEGY == "soft":
            final_probs = prob_stack.mean(axis=0)
        else:
            raise ValueError(f"Unsupported VOTING_STRATEGY: {VOTING_STRATEGY}")

        pred_id = int(np.argmax(final_probs))
        pred_label = id_to_label[pred_id]
        true_label = grouped_true[cid]

        predictions.append({
            "celebrity_id": cid,
            "true_label": true_label,
            "pred_label": pred_label,
            "probabilities": final_probs.tolist(),
            "num_chunks": int(len(grouped_probs[cid])),
            "version": "bertweet_v3",
            "model_name": MODEL_NAME,
            "voting_strategy": VOTING_STRATEGY,
        })

        y_true.append(true_label)
        y_pred.append(pred_label)

    return predictions, y_true, y_pred


def evaluate_celebrity_level(
    target: str,
    val_rows: List[dict],
    trainer: Trainer,
    val_dataset: Dataset,
    id_to_label: Dict[int, str],
):
    labels = LABEL_ORDERS[target]

    pred_output = trainer.predict(val_dataset)
    logits = pred_output.predictions

    predictions, y_true, y_pred = aggregate_celebrity_predictions(
        rows=val_rows,
        logits=logits,
        target=target,
        id_to_label=id_to_label,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "target_label": target,
        "version": "bertweet_v3",
        "model_name": MODEL_NAME,
        "voting_strategy": VOTING_STRATEGY,
        "num_val_celebrities": int(len(set(row["celebrity_id"] for row in val_rows))),
        "num_val_chunks": int(len(val_rows)),
        "val_accuracy": float(accuracy_score(y_true, y_pred)),
        "val_macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "classification_report": report,
    }

    return predictions, metrics


def build_training_args(target: str):
    output_dir = os.path.join(bertweet_v3_checkpoints_dir, target)
    logging_dir = os.path.join(bertweet_v3_logs_dir, target)

    common_kwargs = dict(
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="chunk_macro_f1",
        greater_is_better=True,
        fp16=USE_FP16 and torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        seed=RANDOM_SEED,
    )

    # transformers versions differ: older uses evaluation_strategy,
    # newer versions may use eval_strategy.
    try:
        return TrainingArguments(
            evaluation_strategy="steps",
            save_strategy="steps",
            **common_kwargs,
        )
    except TypeError:
        return TrainingArguments(
            eval_strategy="steps",
            save_strategy="steps",
            **common_kwargs,
        )


# -------------------------------------------------------------------
# Train
# -------------------------------------------------------------------
def train_one_target(target: str):
    print(f"\n========== BERTweet V3 TARGET: {target} ==========")

    set_seed(RANDOM_SEED)

    rows = load_json_or_ndjson(bertweet_train_tokenized_path)

    label_to_id, id_to_label = build_label_maps(target)

    grouped, celebrity_to_label = group_rows_by_celebrity(rows, target)

    train_ids, val_ids = split_by_celebrity(
        grouped=grouped,
        celebrity_to_label=celebrity_to_label,
        label_to_id=label_to_id,
    )

    train_rows = sample_rows_per_celebrity(
        grouped=grouped,
        celebrity_ids=train_ids,
        max_chunks_per_celebrity=MAX_TRAIN_CHUNKS_PER_CELEB,
        seed=RANDOM_SEED,
    )

    val_rows = sample_rows_per_celebrity(
        grouped=grouped,
        celebrity_ids=val_ids,
        max_chunks_per_celebrity=MAX_VAL_CHUNKS_PER_CELEB,
        seed=RANDOM_SEED + 1,
    )

    print(f"[INFO] Train celebrities: {len(train_ids)}")
    print(f"[INFO] Val celebrities:   {len(val_ids)}")
    print(f"[INFO] Train chunks:      {len(train_rows)}")
    print(f"[INFO] Val chunks:        {len(val_rows)}")
    print(f"[INFO] Labels:            {label_to_id}")

    train_dataset = BERTweetChunkDataset(train_rows, target, label_to_id)
    val_dataset = BERTweetChunkDataset(val_rows, target, label_to_id)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        normalization=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_to_id),
        id2label={int(k): v for k, v in id_to_label.items()},
        label2id=label_to_id,
    )

    model.resize_token_embeddings(len(tokenizer))

    print(f"[INFO] Tokenizer vocab size: {len(tokenizer)}")
    print(f"[INFO] Model vocab size:     {model.config.vocab_size}")
    print(f"[INFO] Max position embeddings: {model.config.max_position_embeddings}")
    print(f"[INFO] Example sequence length: {len(train_rows[0]['input_ids'])}")

    validate_tokenized_rows(train_rows, tokenizer, model)
    validate_tokenized_rows(val_rows, tokenizer, model)

    class_weights = build_class_weight_tensor(
        target=target,
        train_rows=train_rows,
        label_to_id=label_to_id,
    )

    if class_weights is not None:
        print(f"[INFO] Class weights: {class_weights.tolist()}")
    else:
        print("[INFO] Class weights: None")

    training_args = build_training_args(target)

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(id_to_label),
        class_weights=class_weights,
    )

    trainer.train()

    final_model_dir = os.path.join(
        bertweet_v3_checkpoints_dir,
        target,
        "final_model",
    )

    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    predictions, metrics = evaluate_celebrity_level(
        target=target,
        val_rows=val_rows,
        trainer=trainer,
        val_dataset=val_dataset,
        id_to_label=id_to_label,
    )

    metrics["label_to_id"] = label_to_id
    metrics["id_to_label"] = id_to_label
    metrics["num_train_celebrities"] = int(len(train_ids))
    metrics["num_train_chunks"] = int(len(train_rows))
    metrics["max_train_chunks_per_celebrity"] = MAX_TRAIN_CHUNKS_PER_CELEB
    metrics["max_val_chunks_per_celebrity"] = MAX_VAL_CHUNKS_PER_CELEB

    metrics_path = os.path.join(
        bertweet_v3_metrics_dir,
        f"{target}_metrics.json",
    )

    predictions_path = os.path.join(
        bertweet_v3_predictions_dir,
        f"{target}_val_predictions.json",
    )

    save_json(metrics, metrics_path)
    save_json(predictions, predictions_path)

    print(f"[OK] Saved model:       {final_model_dir}")
    print(f"[OK] Saved metrics:     {metrics_path}")
    print(f"[OK] Saved predictions: {predictions_path}")
    print(
        f"[RESULT] {target} "
        f"val_acc={metrics['val_accuracy']:.4f} "
        f"val_macro_f1={metrics['val_macro_f1']:.4f}"
    )


def resolve_targets(target: str):
    if target == "all":
        return ["occupation", "gender", "birthyear"]
    return [target]


def main():
    parser = argparse.ArgumentParser(description="Train BERTweet V3")
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear", "all"],
        default=TARGET_LABEL,
    )

    args = parser.parse_args()

    ensure_dirs()

    for target in resolve_targets(args.target):
        train_one_target(target)


if __name__ == "__main__":
    main()