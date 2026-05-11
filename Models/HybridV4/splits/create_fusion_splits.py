import argparse
import json
import os
import sys
from typing import Dict, Iterable, List

from sklearn.model_selection import train_test_split


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
    train_label_path,
    hybrid_v4_splits_dir,
)

from Models.HybridV4.feature.config_features import (
    RANDOM_SEED,
    TARGETS,
    LABEL_ORDERS,
)


DEFAULT_VAL_RATIO = 0.2


def ensure_dirs():
    os.makedirs(hybrid_v4_splits_dir, exist_ok=True)


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


def write_ndjson(rows: Iterable[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"[OK] Saved {count} rows: {path}")


def get_celebrity_id(row: dict) -> str:
    if "id" in row:
        return str(row["id"])
    if "celebrity_id" in row:
        return str(row["celebrity_id"])

    raise KeyError(
        f"No celebrity id found. Available keys: {sorted(row.keys())}"
    )


def map_birthyear_to_bucket(year) -> str:
    year = int(year)
    buckets = [int(x) for x in LABEL_ORDERS["birthyear"]]
    nearest = min(buckets, key=lambda b: abs(year - b))
    return str(nearest)


def get_target_label(row: dict, target: str) -> str:
    if target == "birthyear":
        return map_birthyear_to_bucket(row["birthyear"])

    return str(row[target])


def load_label_examples(target: str) -> List[dict]:
    examples = []

    for row in iter_ndjson(train_label_path):
        cid = get_celebrity_id(row)
        label = get_target_label(row, target)

        examples.append({
            "celebrity_id": cid,
            "target": target,
            "label": label,
        })

    return examples


def validate_labels(examples: List[dict], target: str):
    expected = set(LABEL_ORDERS[target])
    observed = {ex["label"] for ex in examples}

    unknown = observed - expected
    if unknown:
        raise ValueError(
            f"Unknown labels for target={target}: {sorted(unknown)}. "
            f"Expected: {sorted(expected)}"
        )

    print(f"[INFO] Target={target}")
    print(f"[INFO] Label order: {LABEL_ORDERS[target]}")

    for label in LABEL_ORDERS[target]:
        count = sum(1 for ex in examples if ex["label"] == label)
        print(f"[INFO]   {label}: {count}")


def create_split_rows(
    target: str,
    val_ratio: float,
    seed: int,
) -> List[dict]:
    examples = load_label_examples(target)
    validate_labels(examples, target)

    ids = [ex["celebrity_id"] for ex in examples]
    labels = [ex["label"] for ex in examples]

    train_ids, val_ids = train_test_split(
        ids,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )

    train_id_set = set(train_ids)
    val_id_set = set(val_ids)

    rows = []

    for ex in examples:
        cid = ex["celebrity_id"]

        if cid in train_id_set:
            split = "fusion_train"
        elif cid in val_id_set:
            split = "fusion_val"
        else:
            raise RuntimeError(f"ID not assigned to any split: {cid}")

        rows.append({
            "celebrity_id": cid,
            "target": target,
            "split": split,
            "label": ex["label"],
            "seed": seed,
            "val_ratio": val_ratio,
        })

    train_count = sum(1 for row in rows if row["split"] == "fusion_train")
    val_count = sum(1 for row in rows if row["split"] == "fusion_val")

    print(f"[INFO] fusion_train rows: {train_count}")
    print(f"[INFO] fusion_val rows:   {val_count}")

    return rows


def save_split(target: str, rows: List[dict]):
    output_path = os.path.join(
        hybrid_v4_splits_dir,
        f"{target}_fusion_split.ndjson",
    )

    # Stable ordering: train/val mixed by original label-file order.
    write_ndjson(rows, output_path)


def resolve_targets(target: str) -> List[str]:
    if target == "all":
        return TARGETS
    return [target]


def main():
    parser = argparse.ArgumentParser(
        description="Create shared HybridV4 fusion splits in NDJSON format."
    )
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear", "all"],
        default="all",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")

    ensure_dirs()

    for target in resolve_targets(args.target):
        print(f"\n========== Creating fusion split: {target} ==========")
        rows = create_split_rows(
            target=target,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        save_split(target, rows)


if __name__ == "__main__":
    main()