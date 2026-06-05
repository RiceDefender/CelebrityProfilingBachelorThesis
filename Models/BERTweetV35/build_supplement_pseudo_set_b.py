import json
import os
import sys
from collections import Counter, defaultdict


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from _constants import outputs_dir


INPUT_PATH = os.path.join(
    outputs_dir,
    "hybrid_v4",
    "supplement_audit",
    "occupation_supplement_bertweet_v35_predictions.json",
)

OUT_DIR = os.path.join(
    outputs_dir,
    "hybrid_v4",
    "supplement_audit",
)

OUTPUT_PATH = os.path.join(
    OUT_DIR,
    "occupation_supplement_setB_conf0p45_margin0p10_max250_no_politics.json",
)

REPORT_PATH = os.path.join(
    OUT_DIR,
    "occupation_supplement_setB_conf0p45_margin0p10_max250_no_politics_report.md",
)


CONF_THRESHOLD = 0.45
MARGIN_THRESHOLD = 0.10
MAX_PER_CLASS = 250
EXCLUDE_LABELS = {"politics"}
TARGET_LABELS = ["sports", "performer", "creator"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_text(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def build_set_b(predictions):
    candidates_by_class = defaultdict(list)

    for row in predictions:
        pred_label = row["bertweet_v35_pred_label"]
        confidence = float(row["confidence"])
        margin = float(row["margin"])

        if pred_label in EXCLUDE_LABELS:
            continue

        if pred_label not in TARGET_LABELS:
            continue

        if confidence < CONF_THRESHOLD:
            continue

        if margin < MARGIN_THRESHOLD:
            continue

        out_row = {
            "celebrity_id": str(row["celebrity_id"]),
            "target": "occupation",
            "pseudo_label": pred_label,
            "pseudo_source": "bertweet_v35_controlled_sampling_setB",
            "confidence": confidence,
            "margin": margin,
            "entropy": float(row["entropy"]),
            "num_chunks": int(row["num_chunks"]),
            "bertweet_v35_probabilities": row["bertweet_v35_probabilities"],
            "bertweet_v35_pred_label": pred_label,
            "supplement_label_if_available": row.get("supplement_label_if_available"),
            "selection_config": {
                "conf_threshold": CONF_THRESHOLD,
                "margin_threshold": MARGIN_THRESHOLD,
                "max_per_class": MAX_PER_CLASS,
                "exclude_labels": sorted(EXCLUDE_LABELS),
            },
        }

        candidates_by_class[pred_label].append(out_row)

    selected = []

    for label in TARGET_LABELS:
        rows = candidates_by_class[label]

        rows = sorted(
            rows,
            key=lambda r: (r["confidence"], r["margin"]),
            reverse=True,
        )

        selected.extend(rows[:MAX_PER_CLASS])

    selected = sorted(
        selected,
        key=lambda r: int(r["celebrity_id"]) if str(r["celebrity_id"]).isdigit() else str(r["celebrity_id"]),
    )

    return selected, candidates_by_class


def build_report(predictions, selected, candidates_by_class):
    pred_counts = Counter(row["bertweet_v35_pred_label"] for row in predictions)
    selected_counts = Counter(row["pseudo_label"] for row in selected)
    available_counts = {
        label: len(candidates_by_class.get(label, []))
        for label in TARGET_LABELS
    }

    label_agreement = defaultdict(lambda: {"same": 0, "total": 0})

    for row in selected:
        pseudo = row["pseudo_label"]
        supp_label = row.get("supplement_label_if_available")

        if supp_label is None:
            continue

        label_agreement[pseudo]["total"] += 1
        if pseudo == supp_label:
            label_agreement[pseudo]["same"] += 1

    lines = []
    lines.append("# Supplement Pseudo Set B")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- confidence threshold: `{CONF_THRESHOLD}`")
    lines.append(f"- margin threshold: `{MARGIN_THRESHOLD}`")
    lines.append(f"- max per class: `{MAX_PER_CLASS}`")
    lines.append(f"- excluded labels: `{sorted(EXCLUDE_LABELS)}`")
    lines.append("")
    lines.append("## Prediction distribution before filtering")
    lines.append("")
    lines.append("| label | count |")
    lines.append("|---|---:|")
    for label, count in sorted(pred_counts.items()):
        lines.append(f"| {label} | {count} |")
    lines.append("")
    lines.append("## Available candidates after thresholding")
    lines.append("")
    lines.append("| label | available | selected |")
    lines.append("|---|---:|---:|")
    for label in TARGET_LABELS:
        lines.append(
            f"| {label} | {available_counts.get(label, 0)} | {selected_counts.get(label, 0)} |"
        )
    lines.append("")
    lines.append("## Agreement with supplement labels, only as noisy plausibility check")
    lines.append("")
    lines.append("| pseudo_label | same_as_supp_label | total | agreement |")
    lines.append("|---|---:|---:|---:|")
    for label in TARGET_LABELS:
        same = label_agreement[label]["same"]
        total = label_agreement[label]["total"]
        agreement = same / total if total > 0 else 0.0
        lines.append(f"| {label} | {same} | {total} | {agreement:.4f} |")
    lines.append("")
    lines.append("## Selected set size")
    lines.append("")
    lines.append(f"`{len(selected)}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- This set excludes politics because supplement labels do not provide a reliable politics reference.")
    lines.append("- Use this set only as pseudo-labeled augmentation, not as clean supervised data.")
    lines.append("- Recommended training: clean train + Set B, with 1 or 2 epochs maximum.")
    lines.append("")

    return "\n".join(lines)


def main():
    print(f"[INFO] Loading predictions: {INPUT_PATH}")
    predictions = load_json(INPUT_PATH)

    selected, candidates_by_class = build_set_b(predictions)
    report = build_report(predictions, selected, candidates_by_class)

    save_json(selected, OUTPUT_PATH)
    save_text(report, REPORT_PATH)

    print(f"[OK] Saved Set B:  {OUTPUT_PATH}")
    print(f"[OK] Saved report: {REPORT_PATH}")

    print("[INFO] Selected distribution:")
    for label, count in Counter(row["pseudo_label"] for row in selected).items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()