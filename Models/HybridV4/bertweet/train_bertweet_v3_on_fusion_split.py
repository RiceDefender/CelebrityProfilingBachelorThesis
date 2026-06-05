import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


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
    bertweet_train_tokenized_path,
    bertweet_test_tokenized_path,
    bertweet_v3_checkpoints_dir,
    bertweet_v3_logs_dir,
    hybrid_v4_splits_dir,
    hybrid_v4_bertweet_probs_dir,
)

# later define config in hybrid_v4_bertweet_config.py and import here
from Models.BERTweet.config_bertweet_model import (
    MODEL_NAME,
    RANDOM_SEED,
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dirs():
    os.makedirs(hybrid_v4_bertweet_probs_dir, exist_ok=True)
    os.makedirs(bertweet_v3_checkpoints_dir, exist_ok=True)
    os.makedirs(bertweet_v3_logs_dir, exist_ok=True)


def iter_ndjson(path: str) -> Iterable[dict]:
    print(f"[INFO] Streaming: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid NDJSON in {path} at line {line_idx}: {e}"
                ) from e


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_fusion_split_ids(target: str):
    path = os.path.join(
        hybrid_v4_splits_dir,
        f"{target}_fusion_split.ndjson",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing fusion split file: {path}")

    train_ids = set()
    val_ids = set()

    for row in iter_ndjson(path):
        if row["target"] != target:
            continue

        cid = str(row["celebrity_id"])

        if row["split"] == "fusion_train":
            train_ids.add(cid)
        elif row["split"] == "fusion_val":
            val_ids.add(cid)

    print(f"[INFO] Loaded fusion split for {target}:")
    print(f"[INFO]   fusion_train IDs: {len(train_ids)}")
    print(f"[INFO]   fusion_val IDs:   {len(val_ids)}")

    if not train_ids or not val_ids:
        raise ValueError(f"Invalid fusion split for target={target}")

    return train_ids, val_ids


def map_birthyear_to_bucket(year, target: str = "birthyear") -> str:
    year = int(year)
    buckets = [int(x) for x in LABEL_ORDERS[target]]
    nearest = min(buckets, key=lambda b: abs(year - b))
    return str(nearest)


def get_label(row: dict, target: str) -> str:
    if target == "creator_binary":
        return "creator" if row["occupation"] == "creator" else "not_creator"

    if target == "occupation_3class":
        occupation = row["occupation"]
        if occupation == "creator":
            raise ValueError("occupation_3class received creator row.")
        return str(occupation)

    if target == "birthyear":
        return map_birthyear_to_bucket(row["birthyear"], target="birthyear")

    return str(row[target])


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

        if target == "occupation_3class" and row["occupation"] == "creator":
            continue

        grouped[cid].append(row)
        celebrity_to_label[cid] = get_label(row, target)

    return grouped, celebrity_to_label


def load_train_rows_for_split(target: str, train_ids: set, val_ids: set) -> List[dict]:
    wanted_ids = train_ids | val_ids
    rows = []

    for row in iter_ndjson(bertweet_train_tokenized_path):
        cid = str(row["celebrity_id"])

        if cid not in wanted_ids:
            continue

        if target == "occupation_3class" and row["occupation"] == "creator":
            continue

        rows.append(row)

    print(f"[INFO] Loaded train-tokenized rows for fusion split: {len(rows)}")
    print(f"[INFO] Celebrities in loaded rows: {len(set(str(r['celebrity_id']) for r in rows))}")

    return rows


def load_test_rows(target: str) -> List[dict]:
    rows = []

    for row in iter_ndjson(bertweet_test_tokenized_path):
        if target == "occupation_3class" and row["occupation"] == "creator":
            continue
        rows.append(row)

    print(f"[INFO] Loaded official test rows: {len(rows)}")
    print(f"[INFO] Test celebrities: {len(set(str(r['celebrity_id']) for r in rows))}")

    return rows


def sample_rows_per_celebrity(
    grouped,
    celebrity_ids: List[str],
    max_chunks_per_celebrity,
    seed: int,
):
    rng = random.Random(seed)
    sampled_rows = []

    for cid in celebrity_ids:
        rows = grouped[cid]

        if max_chunks_per_celebrity is None or len(rows) <= max_chunks_per_celebrity:
            sampled_rows.extend(rows)
        else:
            sampled_rows.extend(rng.sample(rows, max_chunks_per_celebrity))

    return sampled_rows


def select_evenly_spaced(rows: List[dict], max_chunks):
    if max_chunks is None or len(rows) <= max_chunks:
        return rows

    indices = np.linspace(0, len(rows) - 1, num=max_chunks, dtype=int)
    return [rows[int(i)] for i in indices]


def prepare_prediction_rows(grouped, celebrity_ids: List[str], max_chunks_per_celebrity):
    selected = []

    for cid in celebrity_ids:
        celeb_rows = sorted(grouped[cid], key=lambda r: int(r.get("chunk_id", 0)))
        selected.extend(select_evenly_spaced(celeb_rows, max_chunks_per_celebrity))

    return selected


def build_class_weight_tensor(target: str, train_rows: List[dict], label_to_id: Dict[str, int]):
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


def make_compute_metrics(id_to_label):
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
    prob_key: str,
    pred_key: str,
    split: str,
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

    for cid in sorted(grouped_probs.keys(), key=lambda x: int(x) if x.isdigit() else x):
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
            "target": target,
            "split": split,
            "true_label": true_label,
            "labels": LABEL_ORDERS[target],
            prob_key: final_probs.tolist(),
            pred_key: pred_label,
            "num_chunks": int(len(grouped_probs[cid])),
            "version": "bertweet_v3",
            "model_name": MODEL_NAME,
            "voting_strategy": VOTING_STRATEGY,
        })

        y_true.append(true_label)
        y_pred.append(pred_label)

    metrics = {
        "target": target,
        "split": split,
        "version": "bertweet_v3",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABEL_ORDERS[target],
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABEL_ORDERS[target],
            output_dict=True,
            zero_division=0,
        ),
        "num_celebrities": len(predictions),
    }

    return predictions, metrics


def build_training_args(target: str):
    output_dir = os.path.join(
        bertweet_v3_checkpoints_dir,
        f"hybrid_v4_fusion_split_{target}",
    )
    logging_dir = os.path.join(
        bertweet_v3_logs_dir,
        f"hybrid_v4_fusion_split_{target}",
    )

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


def predict_logits(model, rows: List[dict], batch_size: int = 32):
    dataset = BERTweetPredictDataset(rows)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    all_logits = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting chunks"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits.detach().cpu().numpy()
            all_logits.append(logits)

    return np.concatenate(all_logits, axis=0)


def train_one_target(target: str):
    print(f"\n========== HybridV4 BERTweet V3 on fusion split: target={target} ==========")

    set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    train_ids, val_ids = load_fusion_split_ids(target)
    label_to_id, id_to_label = build_label_maps(target)

    train_tokenized_rows = load_train_rows_for_split(target, train_ids, val_ids)
    grouped, celebrity_to_label = group_rows_by_celebrity(train_tokenized_rows, target)

    train_ids_sorted = sorted([cid for cid in train_ids if cid in grouped], key=lambda x: int(x) if x.isdigit() else x)
    val_ids_sorted = sorted([cid for cid in val_ids if cid in grouped], key=lambda x: int(x) if x.isdigit() else x)

    train_rows = sample_rows_per_celebrity(
        grouped=grouped,
        celebrity_ids=train_ids_sorted,
        max_chunks_per_celebrity=MAX_TRAIN_CHUNKS_PER_CELEB,
        seed=RANDOM_SEED,
    )

    val_rows_train_eval = sample_rows_per_celebrity(
        grouped=grouped,
        celebrity_ids=val_ids_sorted,
        max_chunks_per_celebrity=MAX_VAL_CHUNKS_PER_CELEB,
        seed=RANDOM_SEED + 1,
    )

    val_rows_predict = prepare_prediction_rows(
        grouped=grouped,
        celebrity_ids=val_ids_sorted,
        max_chunks_per_celebrity=128,
    )

    print(f"[INFO] Fusion train celebrities: {len(train_ids_sorted)}")
    print(f"[INFO] Fusion val celebrities:   {len(val_ids_sorted)}")
    print(f"[INFO] Train chunks:             {len(train_rows)}")
    print(f"[INFO] Val chunks train-eval:    {len(val_rows_train_eval)}")
    print(f"[INFO] Val chunks prediction:    {len(val_rows_predict)}")
    print(f"[INFO] Labels:                   {label_to_id}")

    train_dataset = BERTweetChunkDataset(train_rows, target, label_to_id)
    val_dataset = BERTweetChunkDataset(val_rows_train_eval, target, label_to_id)

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

    class_weights = build_class_weight_tensor(target, train_rows, label_to_id)

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Class weights: {class_weights.tolist() if class_weights is not None else None}")

    trainer = WeightedLossTrainer(
        model=model,
        args=build_training_args(target),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(id_to_label),
        class_weights=class_weights,
    )

    trainer.train()

    final_model_dir = os.path.join(
        bertweet_v3_checkpoints_dir,
        f"hybrid_v4_fusion_split_{target}",
        "final_model",
    )

    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    model.to(DEVICE)

    val_logits = predict_logits(
        model=model,
        rows=val_rows_predict,
        batch_size=32,
    )

    val_predictions, val_metrics = aggregate_celebrity_predictions(
        rows=val_rows_predict,
        logits=val_logits,
        target=target,
        id_to_label=id_to_label,
        prob_key="bertweet_v3_probabilities",
        pred_key="bertweet_v3_pred_label",
        split="fusion_val",
    )

    test_rows_all = load_test_rows(target)
    test_grouped, _ = group_rows_by_celebrity(test_rows_all, target)
    test_ids = sorted(test_grouped.keys(), key=lambda x: int(x) if x.isdigit() else x)

    test_rows_predict = prepare_prediction_rows(
        grouped=test_grouped,
        celebrity_ids=test_ids,
        max_chunks_per_celebrity=128,
    )

    test_logits = predict_logits(
        model=model,
        rows=test_rows_predict,
        batch_size=32,
    )

    test_predictions, test_metrics = aggregate_celebrity_predictions(
        rows=test_rows_predict,
        logits=test_logits,
        target=target,
        id_to_label=id_to_label,
        prob_key="bertweet_v3_probabilities",
        pred_key="bertweet_v3_pred_label",
        split="test",
    )

    val_pred_path = os.path.join(
        hybrid_v4_bertweet_probs_dir,
        f"{target}_fusion_val_bertweet_v3_probs.json",
    )
    test_pred_path = os.path.join(
        hybrid_v4_bertweet_probs_dir,
        f"{target}_test_bertweet_v3_probs.json",
    )
    val_metrics_path = os.path.join(
        hybrid_v4_bertweet_probs_dir,
        f"{target}_fusion_val_bertweet_v3_metrics.json",
    )
    test_metrics_path = os.path.join(
        hybrid_v4_bertweet_probs_dir,
        f"{target}_test_bertweet_v3_metrics.json",
    )

    save_json(val_predictions, val_pred_path)
    save_json(test_predictions, test_pred_path)
    save_json(val_metrics, val_metrics_path)
    save_json(test_metrics, test_metrics_path)

    print(f"[OK] Saved model:            {final_model_dir}")
    print(f"[OK] Saved fusion val preds: {val_pred_path}")
    print(f"[OK] Saved test preds:       {test_pred_path}")
    print(
        f"[RESULT] {target} BERTweet V3 "
        f"fusion_val_acc={val_metrics['accuracy']:.4f} "
        f"fusion_val_macro_f1={val_metrics['macro_f1']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train BERTweet V3 on shared HybridV4 fusion split."
    )
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear"],
        default="occupation",
    )

    args = parser.parse_args()

    ensure_dirs()
    train_one_target(args.target)


if __name__ == "__main__":
    main()