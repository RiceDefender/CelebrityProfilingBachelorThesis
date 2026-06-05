import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


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
    test_label_path,
    supp_label_path,
    bertweet_train_tokenized_path,
    bertweet_test_tokenized_path,
    bertweet_v34_train_tokenized_path,
    bertweet_v34_test_tokenized_path,
    hybrid_v4_output_dir,
    hybrid_v4_splits_dir,
)


LABEL_KEYS = ["occupation", "gender", "birthyear"]
OCCUPATION_ORDER = ["sports", "performer", "creator", "politics"]
GENDER_ORDER = ["male", "female"]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def iter_ndjson(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid NDJSON in {path} at line {line_idx}: {e}") from e


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_text(text: str, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def get_id(row: dict) -> str:
    for key in ["celebrity_id", "id", "author_id"]:
        if key in row:
            return str(row[key])
    raise KeyError(f"No celebrity id key found in row keys={list(row.keys())}")


def load_labels(path: str) -> Dict[str, dict]:
    labels = {}
    for row in iter_ndjson(path):
        cid = get_id(row)
        labels[cid] = row
    return labels


def make_age_bin(birthyear) -> str:
    year = int(birthyear)

    if year < 1950:
        return "1940-1949"
    if year < 1960:
        return "1950-1959"
    if year < 1970:
        return "1960-1969"
    if year < 1980:
        return "1970-1979"
    if year < 1990:
        return "1980-1989"
    return "1990-1999"


def count_values(labels: Dict[str, dict], key: str) -> Counter:
    c = Counter()
    for row in labels.values():
        if key == "age_bin":
            c[make_age_bin(row["birthyear"])] += 1
        else:
            c[str(row[key])] += 1
    return c


def ordered_counter_items(counter: Counter, order: List[str] = None):
    if order:
        return [(label, counter.get(label, 0)) for label in order]
    return sorted(counter.items(), key=lambda x: x[0])


def plot_bar(counter: Counter, title: str, out_path: str, order: List[str] = None):
    items = ordered_counter_items(counter, order)
    labels = [x[0] for x in items]
    values = [x[1] for x in items]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_hist(values: List[float], title: str, out_path: str, bins: int = 40):
    if not values:
        return

    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_box_by_class(
    values_by_class: Dict[str, List[float]],
    title: str,
    out_path: str,
    order: List[str] = None,
):
    if order is None:
        order = sorted(values_by_class.keys())

    data = [values_by_class.get(label, []) for label in order]
    non_empty = [(label, vals) for label, vals in zip(order, data) if len(vals) > 0]

    if not non_empty:
        return

    labels = [x[0] for x in non_empty]
    data = [x[1] for x in non_empty]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def summarize_numeric(values: List[float]) -> dict:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p10": None,
            "p90": None,
        }

    arr = np.array(values, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def load_token_stats(tokenized_path: str, labels: Dict[str, dict]) -> dict:
    celeb_stats = defaultdict(
        lambda: {
            "num_chunks": 0,
            "attention_mask_sums": [],
            "seq_lens": [],
            "padding_ratios": [],
            "full_attention_chunks": 0,
        }
    )

    max_seq_len_seen = 0

    for row in iter_ndjson(tokenized_path):
        cid = str(row.get("celebrity_id", row.get("id", "")))

        if cid not in labels:
            continue

        input_ids = row.get("input_ids", [])
        attention_mask = row.get("attention_mask", [])

        if not input_ids or not attention_mask:
            continue

        seq_len = len(input_ids)
        mask_sum = int(sum(attention_mask))
        padding_ratio = 1.0 - (mask_sum / max(seq_len, 1))

        max_seq_len_seen = max(max_seq_len_seen, seq_len)

        celeb_stats[cid]["num_chunks"] += 1
        celeb_stats[cid]["attention_mask_sums"].append(mask_sum)
        celeb_stats[cid]["seq_lens"].append(seq_len)
        celeb_stats[cid]["padding_ratios"].append(padding_ratio)

        if mask_sum == seq_len:
            celeb_stats[cid]["full_attention_chunks"] += 1

    per_celeb = {}

    for cid, s in celeb_stats.items():
        num_chunks = s["num_chunks"]
        mask_sums = s["attention_mask_sums"]
        seq_lens = s["seq_lens"]
        padding_ratios = s["padding_ratios"]

        per_celeb[cid] = {
            "celebrity_id": cid,
            "num_chunks": int(num_chunks),
            "mean_attention_mask_sum": float(np.mean(mask_sums)) if mask_sums else 0.0,
            "median_attention_mask_sum": float(np.median(mask_sums)) if mask_sums else 0.0,
            "mean_seq_len": float(np.mean(seq_lens)) if seq_lens else 0.0,
            "mean_padding_ratio": float(np.mean(padding_ratios)) if padding_ratios else 0.0,
            "full_attention_chunk_ratio": float(
                s["full_attention_chunks"] / max(num_chunks, 1)
            ),
        }

    return {
        "max_seq_len_seen": int(max_seq_len_seen),
        "per_celeb": per_celeb,
    }


def summarize_token_stats_by_class(
    token_stats: dict,
    labels: Dict[str, dict],
    target_key: str,
) -> dict:
    by_class = defaultdict(lambda: defaultdict(list))

    for cid, stat in token_stats["per_celeb"].items():
        label_row = labels.get(cid)
        if not label_row:
            continue

        if target_key == "age_bin":
            cls = make_age_bin(label_row["birthyear"])
        else:
            cls = str(label_row[target_key])

        for metric in [
            "num_chunks",
            "mean_attention_mask_sum",
            "median_attention_mask_sum",
            "mean_seq_len",
            "mean_padding_ratio",
            "full_attention_chunk_ratio",
        ]:
            by_class[cls][metric].append(stat[metric])

    summary = {}

    for cls, metrics in by_class.items():
        summary[cls] = {
            metric: summarize_numeric(values)
            for metric, values in metrics.items()
        }

    return summary


def load_existing_fusion_split(target: str) -> dict:
    path = os.path.join(hybrid_v4_splits_dir, f"{target}_fusion_split.ndjson")

    if not os.path.exists(path):
        return {
            "exists": False,
            "path": path,
            "splits": {},
        }

    splits = defaultdict(list)

    for row in iter_ndjson(path):
        if str(row.get("target")) != target:
            continue
        splits[str(row["split"])].append(str(row["celebrity_id"]))

    return {
        "exists": True,
        "path": path,
        "splits": {k: sorted(v) for k, v in splits.items()},
    }


def split_distribution(labels: Dict[str, dict], ids: List[str]) -> dict:
    selected = {cid: labels[cid] for cid in ids if cid in labels}

    return {
        "n": len(selected),
        "occupation": dict(count_values(selected, "occupation")),
        "gender": dict(count_values(selected, "gender")),
        "age_bin": dict(count_values(selected, "age_bin")),
    }


def make_strata(labels: Dict[str, dict], mode: str) -> Dict[str, str]:
    raw = {}

    for cid, row in labels.items():
        occupation = str(row["occupation"])
        gender = str(row["gender"])
        age_bin = make_age_bin(row["birthyear"])

        if mode == "full":
            raw[cid] = f"{occupation}__{gender}__{age_bin}"
        elif mode == "occupation_gender":
            raw[cid] = f"{occupation}__{gender}"
        elif mode == "occupation":
            raw[cid] = occupation
        elif mode == "gender":
            raw[cid] = gender
        elif mode == "age_bin":
            raw[cid] = age_bin
        else:
            raise ValueError(f"Unknown strata mode: {mode}")

    return raw


def build_balanced_split(
    labels: Dict[str, dict],
    val_size: float,
    seed: int,
) -> dict:
    from sklearn.model_selection import train_test_split

    ids = sorted(labels.keys(), key=lambda x: int(x) if x.isdigit() else x)

    modes = ["full", "occupation_gender", "occupation"]

    last_error = None

    for mode in modes:
        strata_map = make_strata(labels, mode)
        strata = [strata_map[cid] for cid in ids]
        counts = Counter(strata)

        if min(counts.values()) < 2:
            continue

        try:
            train_ids, val_ids = train_test_split(
                ids,
                test_size=val_size,
                random_state=seed,
                shuffle=True,
                stratify=strata,
            )

            return {
                "strategy": mode,
                "seed": seed,
                "val_size": val_size,
                "train_ids": sorted(train_ids, key=lambda x: int(x) if x.isdigit() else x),
                "val_ids": sorted(val_ids, key=lambda x: int(x) if x.isdigit() else x),
                "strata_counts": dict(counts),
            }

        except ValueError as e:
            last_error = str(e)

    np.random.seed(seed)
    shuffled = ids[:]
    np.random.shuffle(shuffled)

    n_val = int(round(len(shuffled) * val_size))
    val_ids = sorted(shuffled[:n_val], key=lambda x: int(x) if x.isdigit() else x)
    train_ids = sorted(shuffled[n_val:], key=lambda x: int(x) if x.isdigit() else x)

    return {
        "strategy": "random_fallback",
        "seed": seed,
        "val_size": val_size,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "last_stratified_error": last_error,
    }


def write_split_ndjson(split_result: dict, out_path: str):
    ensure_dir(os.path.dirname(out_path))

    with open(out_path, "w", encoding="utf-8") as f:
        for cid in split_result["train_ids"]:
            f.write(
                json.dumps(
                    {
                        "target": "multi_target",
                        "celebrity_id": cid,
                        "split": "fusion_train",
                        "split_version": "balanced_multitarget_v1",
                        "seed": split_result["seed"],
                        "strategy": split_result["strategy"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        for cid in split_result["val_ids"]:
            f.write(
                json.dumps(
                    {
                        "target": "multi_target",
                        "celebrity_id": cid,
                        "split": "fusion_val",
                        "split_version": "balanced_multitarget_v1",
                        "seed": split_result["seed"],
                        "strategy": split_result["strategy"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def markdown_counter_table(title: str, counter: Counter, order: List[str] = None) -> str:
    items = ordered_counter_items(counter, order)

    lines = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append("| label | count |")
    lines.append("|---|---:|")

    for label, count in items:
        lines.append(f"| {label} | {count} |")

    lines.append("")
    return "\n".join(lines)


def markdown_split_distribution(name: str, dist: dict) -> str:
    lines = []
    lines.append(f"### {name}")
    lines.append("")
    lines.append(f"n = `{dist['n']}`")
    lines.append("")

    for key in ["occupation", "gender", "age_bin"]:
        lines.append(f"#### {key}")
        lines.append("")
        lines.append("| label | count |")
        lines.append("|---|---:|")

        for label, count in sorted(dist[key].items(), key=lambda x: x[0]):
            lines.append(f"| {label} | {count} |")

        lines.append("")

    return "\n".join(lines)


def build_report(
    train_labels: Dict[str, dict],
    test_labels: Dict[str, dict],
    supp_labels: Dict[str, dict],
    bertweet_stats: dict,
    bertweet_v34_stats: dict,
    split_reports: dict,
    split_result: dict = None,
) -> str:
    lines = []

    lines.append("# HybridV4 Dataset Audit")
    lines.append("")
    lines.append("## Dataset sizes")
    lines.append("")
    lines.append("| dataset | celebrities |")
    lines.append("|---|---:|")
    lines.append(f"| train | {len(train_labels)} |")
    lines.append(f"| test | {len(test_labels)} |")
    lines.append(f"| supplement | {len(supp_labels)} |")
    lines.append("")

    lines.append("## Train label distribution")
    lines.append("")
    lines.append(markdown_counter_table("Occupation", count_values(train_labels, "occupation"), OCCUPATION_ORDER))
    lines.append(markdown_counter_table("Gender", count_values(train_labels, "gender"), GENDER_ORDER))
    lines.append(markdown_counter_table("Age bins", count_values(train_labels, "age_bin")))

    lines.append("## Existing HybridV4 fusion split files")
    lines.append("")

    for target, report in split_reports.items():
        lines.append(f"### {target}")
        lines.append("")
        lines.append(f"path: `{report['path']}`")
        lines.append("")
        lines.append(f"exists: `{report['exists']}`")
        lines.append("")

        if report["exists"]:
            for split_name, ids in report["splits"].items():
                dist = split_distribution(train_labels, ids)
                lines.append(markdown_split_distribution(f"{target} / {split_name}", dist))

    lines.append("## Tokenized input statistics")
    lines.append("")
    lines.append(f"BERTweet max sequence length seen: `{bertweet_stats['max_seq_len_seen']}`")
    lines.append("")
    lines.append(f"BERTweet V3.4 max sequence length seen: `{bertweet_v34_stats['max_seq_len_seen']}`")
    lines.append("")

    if split_result is not None:
        lines.append("## Proposed balanced multi-target split")
        lines.append("")
        lines.append(f"strategy: `{split_result['strategy']}`")
        lines.append("")
        lines.append(f"seed: `{split_result['seed']}`")
        lines.append("")
        lines.append(f"val_size: `{split_result['val_size']}`")
        lines.append("")
        lines.append(markdown_split_distribution(
            "balanced_multitarget / fusion_train",
            split_distribution(train_labels, split_result["train_ids"]),
        ))
        lines.append(markdown_split_distribution(
            "balanced_multitarget / fusion_val",
            split_distribution(train_labels, split_result["val_ids"]),
        ))

    return "\n".join(lines)


def run(args):
    out_dir = os.path.join(hybrid_v4_output_dir, "data_audit")
    plots_dir = os.path.join(out_dir, "plots")
    split_out_dir = os.path.join(hybrid_v4_splits_dir, "audit_candidates")

    ensure_dir(out_dir)
    ensure_dir(plots_dir)
    ensure_dir(split_out_dir)

    print("[INFO] Loading labels...")
    train_labels = load_labels(train_label_path)
    test_labels = load_labels(test_label_path)
    supp_labels = load_labels(supp_label_path)

    print(f"[INFO] Train labels: {len(train_labels)}")
    print(f"[INFO] Test labels:  {len(test_labels)}")
    print(f"[INFO] Supp labels:  {len(supp_labels)}")

    print("[INFO] Plotting label distributions...")
    plot_bar(
        count_values(train_labels, "occupation"),
        "Train occupation distribution",
        os.path.join(plots_dir, "train_occupation_distribution.png"),
        OCCUPATION_ORDER,
    )
    plot_bar(
        count_values(train_labels, "gender"),
        "Train gender distribution",
        os.path.join(plots_dir, "train_gender_distribution.png"),
        GENDER_ORDER,
    )
    plot_bar(
        count_values(train_labels, "age_bin"),
        "Train age-bin distribution",
        os.path.join(plots_dir, "train_age_bin_distribution.png"),
    )

    print("[INFO] Loading BERTweet token statistics...")
    bertweet_stats = load_token_stats(bertweet_train_tokenized_path, train_labels)
    bertweet_v34_stats = load_token_stats(bertweet_v34_train_tokenized_path, train_labels)

    print("[INFO] Summarizing token statistics...")
    token_summary = {
        "bertweet_v3": {
            "max_seq_len_seen": bertweet_stats["max_seq_len_seen"],
            "by_occupation": summarize_token_stats_by_class(bertweet_stats, train_labels, "occupation"),
            "by_gender": summarize_token_stats_by_class(bertweet_stats, train_labels, "gender"),
            "by_age_bin": summarize_token_stats_by_class(bertweet_stats, train_labels, "age_bin"),
        },
        "bertweet_v34": {
            "max_seq_len_seen": bertweet_v34_stats["max_seq_len_seen"],
            "by_occupation": summarize_token_stats_by_class(bertweet_v34_stats, train_labels, "occupation"),
            "by_gender": summarize_token_stats_by_class(bertweet_v34_stats, train_labels, "gender"),
            "by_age_bin": summarize_token_stats_by_class(bertweet_v34_stats, train_labels, "age_bin"),
        },
    }

    save_json(token_summary, os.path.join(out_dir, "token_statistics_summary.json"))

    print("[INFO] Plotting token statistics...")

    for version_name, stats in [
        ("bertweet_v3", bertweet_stats),
        ("bertweet_v34", bertweet_v34_stats),
    ]:
        for target_key, order in [
            ("occupation", OCCUPATION_ORDER),
            ("gender", GENDER_ORDER),
            ("age_bin", None),
        ]:
            values_by_class = defaultdict(list)

            for cid, s in stats["per_celeb"].items():
                row = train_labels.get(cid)
                if not row:
                    continue

                if target_key == "age_bin":
                    cls = make_age_bin(row["birthyear"])
                else:
                    cls = str(row[target_key])

                values_by_class[cls].append(s["mean_attention_mask_sum"])

            plot_box_by_class(
                values_by_class,
                f"{version_name}: mean attention mask sum by {target_key}",
                os.path.join(plots_dir, f"{version_name}_mean_attention_mask_sum_by_{target_key}.png"),
                order=order,
            )

            values_by_class = defaultdict(list)

            for cid, s in stats["per_celeb"].items():
                row = train_labels.get(cid)
                if not row:
                    continue

                if target_key == "age_bin":
                    cls = make_age_bin(row["birthyear"])
                else:
                    cls = str(row[target_key])

                values_by_class[cls].append(s["mean_padding_ratio"])

            plot_box_by_class(
                values_by_class,
                f"{version_name}: mean padding ratio by {target_key}",
                os.path.join(plots_dir, f"{version_name}_mean_padding_ratio_by_{target_key}.png"),
                order=order,
            )

    print("[INFO] Checking existing fusion splits...")
    split_reports = {
        target: load_existing_fusion_split(target)
        for target in ["occupation", "gender", "birthyear"]
    }

    split_result = None

    if args.make_split:
        print("[INFO] Building balanced multi-target split...")
        split_result = build_balanced_split(
            train_labels,
            val_size=args.val_size,
            seed=args.seed,
        )

        split_path = os.path.join(
            split_out_dir,
            f"balanced_multitarget_split_seed{args.seed}_val{str(args.val_size).replace('.', 'p')}.ndjson",
        )

        write_split_ndjson(split_result, split_path)

        save_json(
            {
                "strategy": split_result["strategy"],
                "seed": split_result["seed"],
                "val_size": split_result["val_size"],
                "train_count": len(split_result["train_ids"]),
                "val_count": len(split_result["val_ids"]),
                "train_distribution": split_distribution(train_labels, split_result["train_ids"]),
                "val_distribution": split_distribution(train_labels, split_result["val_ids"]),
                "split_path": split_path,
            },
            os.path.join(
                split_out_dir,
                f"balanced_multitarget_split_report_seed{args.seed}.json",
            ),
        )

        print(f"[OK] Saved candidate split: {split_path}")

    report_json = {
        "dataset_sizes": {
            "train": len(train_labels),
            "test": len(test_labels),
            "supplement": len(supp_labels),
        },
        "train_distribution": {
            "occupation": dict(count_values(train_labels, "occupation")),
            "gender": dict(count_values(train_labels, "gender")),
            "age_bin": dict(count_values(train_labels, "age_bin")),
        },
        "existing_fusion_splits": {
            target: {
                "exists": report["exists"],
                "path": report["path"],
                "split_sizes": {
                    split_name: len(ids)
                    for split_name, ids in report["splits"].items()
                },
                "split_distributions": {
                    split_name: split_distribution(train_labels, ids)
                    for split_name, ids in report["splits"].items()
                },
            }
            for target, report in split_reports.items()
        },
        "token_statistics_summary_path": os.path.join(out_dir, "token_statistics_summary.json"),
    }

    if split_result is not None:
        report_json["balanced_multitarget_candidate"] = {
            "strategy": split_result["strategy"],
            "seed": split_result["seed"],
            "val_size": split_result["val_size"],
            "train_count": len(split_result["train_ids"]),
            "val_count": len(split_result["val_ids"]),
            "train_distribution": split_distribution(train_labels, split_result["train_ids"]),
            "val_distribution": split_distribution(train_labels, split_result["val_ids"]),
        }

    report_md = build_report(
        train_labels=train_labels,
        test_labels=test_labels,
        supp_labels=supp_labels,
        bertweet_stats=bertweet_stats,
        bertweet_v34_stats=bertweet_v34_stats,
        split_reports=split_reports,
        split_result=split_result,
    )

    save_json(report_json, os.path.join(out_dir, "dataset_audit_report.json"))
    save_text(report_md, os.path.join(out_dir, "dataset_audit_report.md"))

    print(f"[OK] Saved report: {os.path.join(out_dir, 'dataset_audit_report.md')}")
    print(f"[OK] Saved plots:  {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="HybridV4 dataset audit and balanced validation split candidate builder"
    )

    parser.add_argument(
        "--make-split",
        action="store_true",
        help="Create a candidate balanced multi-target split. Existing split files are not overwritten.",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation size for candidate split.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for candidate split.",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()