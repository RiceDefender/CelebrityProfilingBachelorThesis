import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, set_seed


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from _constants import (
    hybrid_v4_splits_dir,
    hybrid_v4_bertweet_probs_dir,
    outputs_dir,
)

from Preprocessing.tokenizers.bertweet.config_bertweet_v35_controlled_sampling import (
    bertweet_v35_train_tokenized_path,
    bertweet_v35_test_tokenized_path,
)

from Models.BERTweetV36Attention.config_bertweet_v36_attention import (
    MODEL_NAME,
    VERSION,
    RANDOM_SEED,
    NUM_EPOCHS,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    MAX_GRAD_NORM,
    USE_FP16,
    MAX_TRAIN_CHUNKS_PER_CELEB,
    MAX_VAL_CHUNKS_PER_CELEB,
    MAX_PREDICT_CHUNKS_PER_CELEB,
    ATTENTION_DROPOUT,
    CLASSIFIER_DROPOUT,
    CHUNK_DROPOUT_MODE,
    RANDOM_CHUNK_DROPOUT_RATE,
    CLASS_SPECIFIC_CHUNK_DROPOUT,
    MIN_CHUNKS_AFTER_DROPOUT,
    LABEL_ORDERS,
    CLASS_WEIGHT_BY_TARGET,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bertweet_v36_output_dir = os.path.join(outputs_dir, "bertweet_v3_6_attention_pooling")
bertweet_v36_checkpoints_dir = os.path.join(bertweet_v36_output_dir, "checkpoints")
bertweet_v36_logs_dir = os.path.join(bertweet_v36_output_dir, "logs")
bertweet_v36_metrics_dir = os.path.join(bertweet_v36_output_dir, "metrics")


def ensure_dirs():
    os.makedirs(hybrid_v4_bertweet_probs_dir, exist_ok=True)
    os.makedirs(bertweet_v36_output_dir, exist_ok=True)
    os.makedirs(bertweet_v36_checkpoints_dir, exist_ok=True)
    os.makedirs(bertweet_v36_logs_dir, exist_ok=True)
    os.makedirs(bertweet_v36_metrics_dir, exist_ok=True)


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
        if str(row["target"]) != target:
            continue

        cid = str(row["celebrity_id"])

        if row["split"] == "fusion_train":
            train_ids.add(cid)
        elif row["split"] == "fusion_val":
            val_ids.add(cid)

    print(f"[INFO] Loaded fusion split for {target}")
    print(f"[INFO] fusion_train IDs: {len(train_ids)}")
    print(f"[INFO] fusion_val IDs:   {len(val_ids)}")

    if not train_ids or not val_ids:
        raise ValueError(f"Invalid fusion split for target={target}")

    return train_ids, val_ids


def map_birthyear_to_bucket(year) -> str:
    year = int(year)
    buckets = [int(x) for x in LABEL_ORDERS["birthyear"]]
    nearest = min(buckets, key=lambda b: abs(year - b))
    return str(nearest)


def get_label(row: dict, target: str) -> str:
    if target == "birthyear":
        return map_birthyear_to_bucket(row["birthyear"])
    return str(row[target])


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

    for cid in grouped:
        grouped[cid] = sorted(grouped[cid], key=lambda r: int(r.get("chunk_id", 0)))

    return grouped


def load_rows_for_ids(path: str, wanted_ids: Optional[set] = None) -> List[dict]:
    rows = []

    for row in iter_ndjson(path):
        cid = str(row["celebrity_id"])

        if wanted_ids is not None and cid not in wanted_ids:
            continue

        rows.append(row)

    print(f"[INFO] Loaded rows: {len(rows)}")
    print(f"[INFO] Loaded celebrities: {len(set(str(r['celebrity_id']) for r in rows))}")
    return rows


def select_evenly_spaced(rows: List[dict], max_chunks: Optional[int]):
    if max_chunks is None or len(rows) <= max_chunks:
        return list(rows)

    indices = np.linspace(0, len(rows) - 1, num=max_chunks, dtype=int)
    return [rows[int(i)] for i in indices]


def select_random_chunks(rows: List[dict], max_chunks: Optional[int], rng: random.Random):
    if max_chunks is None or len(rows) <= max_chunks:
        return list(rows)
    return rng.sample(rows, max_chunks)


def apply_chunk_dropout(
    rows: List[dict],
    label: str,
    mode: str,
    rng: random.Random,
) -> List[dict]:
    if mode == "none":
        return rows

    if len(rows) <= MIN_CHUNKS_AFTER_DROPOUT:
        return rows

    if mode == "random":
        dropout_rate = RANDOM_CHUNK_DROPOUT_RATE
    elif mode == "class_specific":
        dropout_rate = CLASS_SPECIFIC_CHUNK_DROPOUT.get(label, RANDOM_CHUNK_DROPOUT_RATE)
    else:
        raise ValueError(f"Unknown CHUNK_DROPOUT_MODE: {mode}")

    keep_prob = 1.0 - dropout_rate

    kept = [row for row in rows if rng.random() < keep_prob]

    if len(kept) < MIN_CHUNKS_AFTER_DROPOUT:
        kept = rng.sample(rows, min(MIN_CHUNKS_AFTER_DROPOUT, len(rows)))

    return kept


class CelebrityChunkDataset(Dataset):
    def __init__(
        self,
        grouped_rows,
        celebrity_ids: List[str],
        target: str,
        label_to_id: Dict[str, int],
        max_chunks_per_celeb: Optional[int],
        chunk_selection: str,
        train_mode: bool,
        chunk_dropout_mode: str,
        seed: int,
    ):
        self.grouped_rows = grouped_rows
        self.celebrity_ids = list(celebrity_ids)
        self.target = target
        self.label_to_id = label_to_id
        self.max_chunks_per_celeb = max_chunks_per_celeb
        self.chunk_selection = chunk_selection
        self.train_mode = train_mode
        self.chunk_dropout_mode = chunk_dropout_mode
        self.seed = seed

    def __len__(self):
        return len(self.celebrity_ids)

    def __getitem__(self, idx):
        cid = self.celebrity_ids[idx]
        all_rows = self.grouped_rows[cid]
        label = get_label(all_rows[0], self.target)

        rng = random.Random(self.seed + idx * 7919)

        if self.chunk_selection == "random":
            rows = select_random_chunks(all_rows, self.max_chunks_per_celeb, rng)
        elif self.chunk_selection == "evenly_spaced":
            rows = select_evenly_spaced(all_rows, self.max_chunks_per_celeb)
        else:
            raise ValueError(f"Unknown chunk_selection: {self.chunk_selection}")

        if self.train_mode:
            rows = apply_chunk_dropout(
                rows=rows,
                label=label,
                mode=self.chunk_dropout_mode,
                rng=rng,
            )

        input_ids = torch.tensor([r["input_ids"] for r in rows], dtype=torch.long)
        attention_mask = torch.tensor([r["attention_mask"] for r in rows], dtype=torch.long)

        return {
            "celebrity_id": cid,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(self.label_to_id[label], dtype=torch.long),
            "true_label": label,
            "num_chunks": len(rows),
        }


def collate_celebrities(batch):
    batch_size = len(batch)
    max_chunks = max(item["input_ids"].shape[0] for item in batch)
    seq_len = batch[0]["input_ids"].shape[1]

    input_ids = torch.zeros((batch_size, max_chunks, seq_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_chunks, seq_len), dtype=torch.long)
    chunk_mask = torch.zeros((batch_size, max_chunks), dtype=torch.bool)

    labels = torch.stack([item["label"] for item in batch])

    celebrity_ids = []
    true_labels = []
    num_chunks = []

    for i, item in enumerate(batch):
        n = item["input_ids"].shape[0]
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]
        chunk_mask[i, :n] = True
        celebrity_ids.append(item["celebrity_id"])
        true_labels.append(item["true_label"])
        num_chunks.append(item["num_chunks"])

    return {
        "celebrity_ids": celebrity_ids,
        "true_labels": true_labels,
        "num_chunks": num_chunks,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "chunk_mask": chunk_mask,
        "labels": labels,
    }


class BERTweetAttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        attention_dropout: float,
        classifier_dropout: float,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(attention_dropout),
            nn.Linear(hidden_size, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask, chunk_mask):
        batch_size, num_chunks, seq_len = input_ids.shape

        flat_input_ids = input_ids.view(batch_size * num_chunks, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_chunks, seq_len)

        outputs = self.encoder(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )

        # BERTweet/RoBERTa-style CLS token is first token.
        cls = outputs.last_hidden_state[:, 0, :]
        cls = cls.view(batch_size, num_chunks, -1)

        attention_logits = self.attention(cls).squeeze(-1)

        # Mask padded chunks.
        mask_value = torch.finfo(attention_logits.dtype).min
        attention_logits = attention_logits.masked_fill(~chunk_mask, mask_value)

        attention_weights = torch.softmax(attention_logits, dim=1)

        pooled = torch.sum(cls * attention_weights.unsqueeze(-1), dim=1)

        logits = self.classifier(pooled)

        return logits, attention_weights


def build_class_weight_tensor(target: str, grouped_rows, train_ids, label_to_id):
    config = CLASS_WEIGHT_BY_TARGET.get(target, None)

    if config is None:
        return None

    num_labels = len(label_to_id)

    labels = []
    for cid in train_ids:
        row = grouped_rows[cid][0]
        labels.append(label_to_id[get_label(row, target)])

    if config == "balanced":
        counts = np.bincount(labels, minlength=num_labels).astype(np.float32)
        total = counts.sum()
        weights = total / (num_labels * np.maximum(counts, 1.0))
        return torch.tensor(weights, dtype=torch.float32)

    weights = np.ones(num_labels, dtype=np.float32)
    for label, weight in config.items():
        if label in label_to_id:
            weights[label_to_id[label]] = float(weight)

    return torch.tensor(weights, dtype=torch.float32)


def evaluate_model(
    model,
    dataloader,
    target: str,
    id_to_label: Dict[int, str],
    split_name: str,
    chunk_dropout_mode: str,
    return_attention: bool = False,
):
    model.eval()

    labels_order = LABEL_ORDERS[target]
    all_predictions = []
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            chunk_mask = batch["chunk_mask"].to(DEVICE)

            logits, attention_weights = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                chunk_mask=chunk_mask,
            )

            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            pred_ids = np.argmax(probs, axis=1)

            attention_weights_np = attention_weights.detach().cpu().numpy()

            for i in range(len(batch["celebrity_ids"])):
                cid = batch["celebrity_ids"][i]
                true_label = batch["true_labels"][i]
                pred_label = id_to_label[int(pred_ids[i])]

                row = {
                    "celebrity_id": cid,
                    "target": target,
                    "split": split_name,
                    "true_label": true_label,
                    "labels": labels_order,
                    "bertweet_v36_attention_probabilities": probs[i].tolist(),
                    "bertweet_v36_attention_pred_label": pred_label,
                    "num_chunks": int(batch["num_chunks"][i]),
                    "version": VERSION,
                    "model_name": MODEL_NAME,
                    "voting_strategy": "attention_pooling",
                    "max_predict_chunks_per_celebrity": MAX_PREDICT_CHUNKS_PER_CELEB,
                    "chunk_dropout_mode": chunk_dropout_mode,
                }

                if return_attention:
                    n = int(batch["num_chunks"][i])
                    row["attention_weights"] = attention_weights_np[i, :n].tolist()

                all_predictions.append(row)
                all_true.append(true_label)
                all_pred.append(pred_label)

    metrics = {
        "target": target,
        "split": split_name,
        "version": VERSION,
        "chunk_dropout_mode": CHUNK_DROPOUT_MODE,
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "macro_f1": float(
            f1_score(
                all_true,
                all_pred,
                labels=labels_order,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            all_true,
            all_pred,
            labels=labels_order,
            output_dict=True,
            zero_division=0,
        ),
        "num_celebrities": len(all_predictions),
    }

    return all_predictions, metrics


def train_one_target(target: str, chunk_dropout_mode_override: Optional[str] = None):
    print(f"\n========== BERTweet V3.6 Attention Pooling: target={target} ==========")

    set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    chunk_dropout_mode = chunk_dropout_mode_override or CHUNK_DROPOUT_MODE

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Chunk dropout mode: {chunk_dropout_mode}")

    train_ids, val_ids = load_fusion_split_ids(target)
    label_to_id, id_to_label = build_label_maps(target)

    wanted_ids = train_ids | val_ids
    train_tokenized_rows = load_rows_for_ids(
        bertweet_v35_train_tokenized_path,
        wanted_ids=wanted_ids,
    )

    grouped = group_rows_by_celebrity(train_tokenized_rows)

    train_ids_sorted = sorted(
        [cid for cid in train_ids if cid in grouped],
        key=lambda x: int(x) if x.isdigit() else x,
    )
    val_ids_sorted = sorted(
        [cid for cid in val_ids if cid in grouped],
        key=lambda x: int(x) if x.isdigit() else x,
    )

    print(f"[INFO] Fusion train celebrities: {len(train_ids_sorted)}")
    print(f"[INFO] Fusion val celebrities:   {len(val_ids_sorted)}")
    print(f"[INFO] Labels: {label_to_id}")

    train_dataset = CelebrityChunkDataset(
        grouped_rows=grouped,
        celebrity_ids=train_ids_sorted,
        target=target,
        label_to_id=label_to_id,
        max_chunks_per_celeb=MAX_TRAIN_CHUNKS_PER_CELEB,
        chunk_selection="random",
        train_mode=True,
        chunk_dropout_mode=chunk_dropout_mode,
        seed=RANDOM_SEED,
    )

    val_train_eval_dataset = CelebrityChunkDataset(
        grouped_rows=grouped,
        celebrity_ids=val_ids_sorted,
        target=target,
        label_to_id=label_to_id,
        max_chunks_per_celeb=MAX_VAL_CHUNKS_PER_CELEB,
        chunk_selection="evenly_spaced",
        train_mode=False,
        chunk_dropout_mode="none",
        seed=RANDOM_SEED + 1,
    )

    val_predict_dataset = CelebrityChunkDataset(
        grouped_rows=grouped,
        celebrity_ids=val_ids_sorted,
        target=target,
        label_to_id=label_to_id,
        max_chunks_per_celeb=MAX_PREDICT_CHUNKS_PER_CELEB,
        chunk_selection="evenly_spaced",
        train_mode=False,
        chunk_dropout_mode="none",
        seed=RANDOM_SEED + 2,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_celebrities,
    )

    val_loader = DataLoader(
        val_train_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_celebrities,
    )

    val_predict_loader = DataLoader(
        val_predict_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_celebrities,
    )

    model = BERTweetAttentionPoolingClassifier(
        model_name=MODEL_NAME,
        num_labels=len(label_to_id),
        attention_dropout=ATTENTION_DROPOUT,
        classifier_dropout=CLASSIFIER_DROPOUT,
    )

    model.to(DEVICE)

    class_weights = build_class_weight_tensor(
        target=target,
        grouped_rows=grouped,
        train_ids=train_ids_sorted,
        label_to_id=label_to_id,
    )

    if class_weights is not None:
        class_weights = class_weights.to(DEVICE)

    print(f"[INFO] Class weights: {class_weights.detach().cpu().tolist() if class_weights is not None else None}")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_update_steps = math.ceil(
        (len(train_loader) * NUM_EPOCHS) / GRADIENT_ACCUMULATION_STEPS
    )
    warmup_steps = int(total_update_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_FP16 and torch.cuda.is_available()
    )

    best_val_macro_f1 = -1.0
    best_model_path = os.path.join(
        bertweet_v36_checkpoints_dir,
        f"hybrid_v4_fusion_split_{target}_{chunk_dropout_mode}",
        "best_model.pt",
    )
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    global_step = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")

        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Training epoch {epoch}")

        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            chunk_mask = batch["chunk_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.cuda.amp.autocast(enabled=USE_FP16 and torch.cuda.is_available()):
                logits, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    chunk_mask=chunk_mask,
                )
                loss = loss_fn(logits, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            running_loss += float(loss.detach().cpu()) * GRADIENT_ACCUMULATION_STEPS

            if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            progress.set_postfix({
                "loss": f"{running_loss / max(step, 1):.4f}",
                "updates": global_step,
            })

        val_predictions, val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            target=target,
            id_to_label=id_to_label,
            split_name="fusion_val_train_eval",
            chunk_dropout_mode=chunk_dropout_mode,
            return_attention=False,
        )

        print(
            f"[VAL] epoch={epoch} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "target": target,
                    "version": VERSION,
                    "chunk_dropout_mode": chunk_dropout_mode,
                    "best_val_macro_f1": best_val_macro_f1,
                    "epoch": epoch,
                },
                best_model_path,
            )
            print(f"[OK] Saved best model: {best_model_path}")

    print(f"\n[INFO] Loading best model: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    val_predictions, val_metrics = evaluate_model(
        model=model,
        dataloader=val_predict_loader,
        target=target,
        id_to_label=id_to_label,
        split_name="fusion_val",
        chunk_dropout_mode=chunk_dropout_mode,
        return_attention=True,
    )

    test_rows = load_rows_for_ids(
        bertweet_v35_test_tokenized_path,
        wanted_ids=None,
    )
    test_grouped = group_rows_by_celebrity(test_rows)
    test_ids = sorted(test_grouped.keys(), key=lambda x: int(x) if x.isdigit() else x)

    test_dataset = CelebrityChunkDataset(
        grouped_rows=test_grouped,
        celebrity_ids=test_ids,
        target=target,
        label_to_id=label_to_id,
        max_chunks_per_celeb=MAX_PREDICT_CHUNKS_PER_CELEB,
        chunk_selection="evenly_spaced",
        train_mode=False,
        chunk_dropout_mode="none",
        seed=RANDOM_SEED + 3,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_celebrities,
    )

    test_predictions, test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        target=target,
        id_to_label=id_to_label,
        split_name="test",
        chunk_dropout_mode=chunk_dropout_mode,
        return_attention=True,
    )

    suffix = f"bertweet_v36_attention_{chunk_dropout_mode}"

    val_pred_path = os.path.join(
        hybrid_v4_bertweet_probs_dir,
        f"{target}_fusion_val_{suffix}_probs.json",
    )
    test_pred_path = os.path.join(
        hybrid_v4_bertweet_probs_dir,
        f"{target}_test_{suffix}_probs.json",
    )
    val_metrics_path = os.path.join(
        hybrid_v4_bertweet_probs_dir,
        f"{target}_fusion_val_{suffix}_metrics.json",
    )
    test_metrics_path = os.path.join(
        hybrid_v4_bertweet_probs_dir,
        f"{target}_test_{suffix}_metrics.json",
    )

    save_json(val_predictions, val_pred_path)
    save_json(test_predictions, test_pred_path)
    save_json(val_metrics, val_metrics_path)
    save_json(test_metrics, test_metrics_path)

    report_path = os.path.join(
        bertweet_v36_metrics_dir,
        f"{target}_fusion_split_{chunk_dropout_mode}_report.json",
    )

    save_json(
        {
            "target": target,
            "version": VERSION,
            "chunk_dropout_mode": chunk_dropout_mode,
            "best_model_path": best_model_path,
            "fusion_val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "max_train_chunks_per_celebrity": MAX_TRAIN_CHUNKS_PER_CELEB,
            "max_val_chunks_per_celebrity": MAX_VAL_CHUNKS_PER_CELEB,
            "max_predict_chunks_per_celebrity": MAX_PREDICT_CHUNKS_PER_CELEB,
            "attention_dropout": ATTENTION_DROPOUT,
            "classifier_dropout": CLASSIFIER_DROPOUT,
            "class_specific_chunk_dropout": CLASS_SPECIFIC_CHUNK_DROPOUT,
        },
        report_path,
    )

    print(f"[OK] Saved fusion val preds: {val_pred_path}")
    print(f"[OK] Saved test preds:       {test_pred_path}")
    print(f"[OK] Saved report:           {report_path}")

    print(
        f"[RESULT] {target} V3.6 Attention "
        f"dropout={chunk_dropout_mode} "
        f"fusion_val_acc={val_metrics['accuracy']:.4f} "
        f"fusion_val_macro_f1={val_metrics['macro_f1']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train BERTweet V3.6 Attention Pooling on HybridV4 fusion split"
    )

    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear"],
        default="occupation",
    )

    parser.add_argument(
        "--chunk-dropout-mode",
        choices=["none", "random", "class_specific"],
        default=None,
        help="Override CHUNK_DROPOUT_MODE from config.",
    )

    args = parser.parse_args()

    ensure_dirs()
    train_one_target(args.target, chunk_dropout_mode_override=args.chunk_dropout_mode)


if __name__ == "__main__":
    main()