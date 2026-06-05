import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
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
    outputs_dir,
    hybrid_v4_bertweet_probs_dir,
)

from Preprocessing.tokenizers.bertweet.config_bertweet_v35_controlled_sampling import (
    bertweet_v35_supp_tokenized_path,
)

from Models.BERTweetV35.config_bertweet_v35_model import (
    MODEL_NAME,
    VERSION,
    MAX_PREDICT_CHUNKS_PER_CELEB,
    PREDICT_BATCH_SIZE,
    LABEL_ORDERS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_MODEL_DIR = os.path.join(
    outputs_dir,
    "bertweet_v3_5_controlled_sampling",
    "checkpoints",
    "hybrid_v4_fusion_split_occupation",
    "final_model",
)

OUT_DIR = os.path.join(outputs_dir, "hybrid_v4", "supplement_audit")

CONF_THRESHOLDS = {
    "sports": 0.80,
    "politics": 0.80,
    "performer": 0.90,
    "creator": 0.90,
}

MARGIN_THRESHOLDS = {
    "sports": 0.20,
    "politics": 0.20,
    "performer": 0.25,
    "creator": 0.25,
}


def make_thresholds(conf: float, margin: float, creator_conf: float = None, performer_conf: float = None):
    thresholds = {
        "sports": conf,
        "politics": conf,
        "performer": conf,
        "creator": conf,
    }

    if creator_conf is not None:
        thresholds["creator"] = creator_conf

    if performer_conf is not None:
        thresholds["performer"] = performer_conf

    margins = {
        "sports": margin,
        "politics": margin,
        "performer": margin,
        "creator": margin,
    }

    return thresholds, margins


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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
                raise ValueError(f"Invalid NDJSON at line {line_idx}: {e}") from e


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_text(text: str, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def entropy(probs: List[float]) -> float:
    arr = np.array(probs, dtype=float)
    arr = np.clip(arr, 1e-12, 1.0)
    return float(-np.sum(arr * np.log(arr)))


def top_margin(probs: List[float]) -> float:
    arr = np.array(probs, dtype=float)
    sorted_probs = np.sort(arr)[::-1]
    if len(sorted_probs) < 2:
        return 0.0
    return float(sorted_probs[0] - sorted_probs[1])


def select_evenly_spaced(rows: List[dict], max_chunks: int):
    if max_chunks is None or len(rows) <= max_chunks:
        return rows

    indices = np.linspace(0, len(rows) - 1, num=max_chunks, dtype=int)
    return [rows[int(i)] for i in indices]


def load_grouped_supp_rows():
    grouped = defaultdict(list)

    for row in iter_ndjson(bertweet_v35_supp_tokenized_path):
        cid = str(row["celebrity_id"])
        grouped[cid].append(row)

    for cid in grouped:
        grouped[cid] = sorted(grouped[cid], key=lambda r: int(r.get("chunk_id", 0)))

    print(f"[INFO] Supplement celebrities: {len(grouped)}")
    print(f"[INFO] Supplement chunks: {sum(len(v) for v in grouped.values())}")

    return grouped


class ChunkPredictDataset(Dataset):
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


def predict_logits(model, rows: List[dict], batch_size: int):
    dataset = ChunkPredictDataset(rows)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    all_logits = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting supplement chunks"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits.detach().cpu().numpy()
            all_logits.append(logits)

    return np.concatenate(all_logits, axis=0)


def predict_supplement(model_dir: str, target: str, conf_thresholds: dict, margin_thresholds: dict):
    labels = LABEL_ORDERS[target]

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Loading model: {model_dir}")
    print(f"[INFO] Target: {target}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

    grouped = load_grouped_supp_rows()

    all_predictions = []
    pseudo_candidates = []

    for cid in tqdm(
            sorted(grouped.keys(), key=lambda x: int(x) if x.isdigit() else x),
            desc="Aggregating supplement celebrities",
    ):
        rows = select_evenly_spaced(grouped[cid], MAX_PREDICT_CHUNKS_PER_CELEB)

        logits = predict_logits(
            model=model,
            rows=rows,
            batch_size=PREDICT_BATCH_SIZE,
        )

        probs = softmax_np(logits)
        final_probs = probs.mean(axis=0)

        pred_idx = int(np.argmax(final_probs))
        pred_label = labels[pred_idx]
        confidence = float(np.max(final_probs))
        margin = top_margin(final_probs.tolist())
        ent = entropy(final_probs.tolist())

        supp_label = None
        if "occupation" in rows[0]:
            supp_label = str(rows[0]["occupation"])

        pred_row = {
            "celebrity_id": cid,
            "target": target,
            "split": "supplement",
            "labels": labels,
            "bertweet_v35_probabilities": final_probs.tolist(),
            "bertweet_v35_pred_label": pred_label,
            "confidence": confidence,
            "margin": margin,
            "entropy": ent,
            "num_chunks": len(rows),
            "version": VERSION,
            "model_name": MODEL_NAME,
            "supplement_label_if_available": supp_label,
        }

        all_predictions.append(pred_row)

        conf_threshold = conf_thresholds.get(pred_label, 0.90)
        margin_threshold = margin_thresholds.get(pred_label, 0.25)

        if confidence >= conf_threshold and margin >= margin_threshold:
            pseudo_candidates.append({
                **pred_row,
                "pseudo_label": pred_label,
                "pseudo_source": "bertweet_v35_controlled_sampling",
                "confidence_threshold": conf_threshold,
                "margin_threshold": margin_threshold,
            })

    return all_predictions, pseudo_candidates


def build_balanced_candidates(candidates: List[dict], max_per_class: int):
    by_class = defaultdict(list)

    for row in candidates:
        by_class[row["pseudo_label"]].append(row)

    balanced = []

    for label, rows in by_class.items():
        rows_sorted = sorted(
            rows,
            key=lambda r: (r["confidence"], r["margin"]),
            reverse=True,
        )

        balanced.extend(rows_sorted[:max_per_class])

    balanced = sorted(
        balanced,
        key=lambda r: int(r["celebrity_id"]) if str(r["celebrity_id"]).isdigit() else str(r["celebrity_id"]),
    )

    return balanced


def build_report(predictions: List[dict], candidates: List[dict], balanced: List[dict]) -> str:
    pred_counts = Counter(row["bertweet_v35_pred_label"] for row in predictions)
    candidate_counts = Counter(row["pseudo_label"] for row in candidates)
    balanced_counts = Counter(row["pseudo_label"] for row in balanced)

    lines = []
    lines.append("# Supplement Pseudo-Label Audit — BERTweet V3.5")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Supplement celebrities predicted: `{len(predictions)}`")
    lines.append(f"- High-confidence pseudo-label candidates: `{len(candidates)}`")
    lines.append(f"- Balanced pseudo-label candidates: `{len(balanced)}`")
    lines.append("")
    lines.append("## Prediction distribution")
    lines.append("")
    lines.append("| predicted_label | count |")
    lines.append("|---|---:|")
    for label, count in sorted(pred_counts.items()):
        lines.append(f"| {label} | {count} |")
    lines.append("")
    lines.append("## High-confidence candidate distribution")
    lines.append("")
    lines.append("| pseudo_label | count |")
    lines.append("|---|---:|")
    for label, count in sorted(candidate_counts.items()):
        lines.append(f"| {label} | {count} |")
    lines.append("")
    lines.append("## Balanced candidate distribution")
    lines.append("")
    lines.append("| pseudo_label | count |")
    lines.append("|---|---:|")
    for label, count in sorted(balanced_counts.items()):
        lines.append(f"| {label} | {count} |")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    lines.append("| label | confidence_threshold | margin_threshold |")
    lines.append("|---|---:|---:|")
    for label in sorted(CONF_THRESHOLDS.keys()):
        lines.append(
            f"| {label} | {CONF_THRESHOLDS[label]:.2f} | {MARGIN_THRESHOLDS[label]:.2f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Do not train on all supplement labels directly.")
    lines.append(
        "- Use only high-confidence pseudo-label candidates if the class distribution is not dominated by performer.")
    lines.append(
        "- If creator has very few high-confidence candidates, supplement is probably not directly useful for creator supervised training.")
    lines.append("- If candidate counts are highly imbalanced, use the balanced candidate file only.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Predict supplement data with BERTweet V3.5 and extract pseudo-label candidates."
    )

    parser.add_argument(
        "--target",
        choices=["occupation"],
        default="occupation",
    )

    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
    )

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=500,
        help="Max pseudo-label candidates per predicted class for balanced candidate set.",
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.70,
        help="Default confidence threshold for pseudo-label candidates.",
    )

    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.10,
        help="Default top1-top2 margin threshold for pseudo-label candidates.",
    )

    parser.add_argument(
        "--creator-conf-threshold",
        type=float,
        default=None,
        help="Optional separate confidence threshold for creator.",
    )

    parser.add_argument(
        "--performer-conf-threshold",
        type=float,
        default=None,
        help="Optional separate confidence threshold for performer.",
    )

    args = parser.parse_args()

    ensure_dir(OUT_DIR)

    conf_thresholds, margin_thresholds = make_thresholds(
        conf=args.conf_threshold,
        margin=args.margin_threshold,
        creator_conf=args.creator_conf_threshold,
        performer_conf=args.performer_conf_threshold,
    )

    predictions, candidates = predict_supplement(
        model_dir=args.model_dir,
        target=args.target,
        conf_thresholds=conf_thresholds,
        margin_thresholds=margin_thresholds,
    )

    balanced = build_balanced_candidates(
        candidates=candidates,
        max_per_class=args.max_per_class,
    )

    threshold_tag = (
        f"conf{str(args.conf_threshold).replace('.', 'p')}"
        f"_margin{str(args.margin_threshold).replace('.', 'p')}"
    )

    if args.creator_conf_threshold is not None:
        threshold_tag += f"_creator{str(args.creator_conf_threshold).replace('.', 'p')}"

    if args.performer_conf_threshold is not None:
        threshold_tag += f"_performer{str(args.performer_conf_threshold).replace('.', 'p')}"

    pred_path = os.path.join(
        OUT_DIR,
        f"{args.target}_supplement_bertweet_v35_predictions_{threshold_tag}.json",
    )
    candidate_path = os.path.join(
        OUT_DIR,
        f"{args.target}_supplement_bertweet_v35_pseudo_candidates_{threshold_tag}.json",
    )
    balanced_path = os.path.join(
        OUT_DIR,
        f"{args.target}_supplement_bertweet_v35_pseudo_candidates_balanced_{threshold_tag}.json",
    )
    report_path = os.path.join(
        OUT_DIR,
        f"{args.target}_supplement_bertweet_v35_pseudo_label_audit_{threshold_tag}.md",
    )
    report_json_path = os.path.join(
        OUT_DIR,
        f"{args.target}_supplement_bertweet_v35_pseudo_label_audit_{threshold_tag}.json",
    )

    save_json(predictions, pred_path)
    save_json(candidates, candidate_path)
    save_json(balanced, balanced_path)

    report = build_report(predictions, candidates, balanced)
    save_text(report, report_path)

    save_json(
        {
            "target": args.target,
            "model_dir": args.model_dir,
            "num_predictions": len(predictions),
            "num_candidates": len(candidates),
            "num_balanced_candidates": len(balanced),
            "prediction_distribution": dict(Counter(row["bertweet_v35_pred_label"] for row in predictions)),
            "candidate_distribution": dict(Counter(row["pseudo_label"] for row in candidates)),
            "balanced_distribution": dict(Counter(row["pseudo_label"] for row in balanced)),
            "confidence_thresholds": conf_thresholds,
            "margin_thresholds": margin_thresholds,
            "max_per_class": args.max_per_class,
        },
        report_json_path,
    )

    print(f"[OK] Saved predictions: {pred_path}")
    print(f"[OK] Saved candidates:  {candidate_path}")
    print(f"[OK] Saved balanced:    {balanced_path}")
    print(f"[OK] Saved report:      {report_path}")


if __name__ == "__main__":
    main()
