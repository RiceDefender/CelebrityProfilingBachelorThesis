"""
Build a consolidated semantic meta-feature table for HybridV4 error analysis.

Script version: semantic-meta-feature-table-v1

This script merges the previously generated semantic analysis outputs into one
profile-level table per task. It is intentionally an export/diagnostic step, not
final model training.

Expected inputs, if available:
  outputs/hybrid_v4/error_analysis/semantic_signals/{task}_semantic_signal_report.json
  outputs/hybrid_v4/error_analysis/semantic_signal_ratios/{task}_semantic_ratio_report.json
  outputs/hybrid_v4/error_analysis/synthetic_semantic_features/{task}_synthetic_semantic_features.json
  outputs/hybrid_v4/error_analysis/text_derived_semantic_features/{task}_text_derived_semantic_features.json

Outputs:
  outputs/hybrid_v4/error_analysis/semantic_meta_features/{task}_semantic_meta_features.csv
  outputs/hybrid_v4/error_analysis/semantic_meta_features/{task}_semantic_meta_features.json
  outputs/hybrid_v4/error_analysis/semantic_meta_features/{task}_semantic_meta_feature_summary.md
  outputs/hybrid_v4/error_analysis/semantic_meta_features/semantic_meta_feature_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_VERSION = "semantic-meta-feature-table-v1"

# Make project-root imports work both with `python -m ...` and direct execution.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from _constants import hybrid_v4_output_dir
except Exception as exc:  # pragma: no cover - user-facing failure path
    raise SystemExit(
        "Could not import hybrid_v4_output_dir from _constants.py. "
        "Run this from the project environment where _constants.py is available."
    ) from exc

TASKS = ("gender", "occupation")
ID_KEYS = ("id", "celebrity_id", "profile_id", "author", "author_id")
TRUE_KEYS = ("true", "true_label", "label", "gold", "y_true")
PRED_KEYS = ("pred", "prediction", "predicted_label", "fusion_pred_label", "y_pred")
EXCLUDE_NUMERIC_KEYS = {
    "id", "celebrity_id", "profile_id", "tokens", "token_count", "n", "count", "rank",
    "changed", "improved", "harmed", "still_wrong", "accuracy", "macro_f1",
}
META_KEYS = {
    "id", "celebrity_id", "profile_id", "task", "target", "true", "true_label", "label",
    "gold", "pred", "prediction", "predicted_label", "fusion_pred_label", "y_true", "y_pred",
    "group", "error_group", "is_correct", "tokens", "token_count", "top_keyword_hits",
    "top_text_derived_hits", "keyword_hits", "text", "features", "ratios", "synthetic_features",
    "text_derived_features", "scores",
}


def safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, str):
        try:
            out = float(value)
        except ValueError:
            return None
        return out if math.isfinite(out) else None
    return None


def read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_value(row: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def normalize_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip()


def infer_group(true_label: Optional[str], pred_label: Optional[str]) -> Optional[str]:
    if not true_label or not pred_label:
        return None
    true_label = str(true_label)
    pred_label = str(pred_label)
    if true_label == pred_label:
        return f"correct_{true_label}"
    return f"{true_label}_to_{pred_label}"


def flatten_numeric_features(prefix: str, row: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric features from common flat/nested report row shapes."""
    out: Dict[str, float] = {}

    def add(name: str, value: Any) -> None:
        v = safe_float(value)
        if v is not None:
            out[name] = v

    for key, value in row.items():
        if key in EXCLUDE_NUMERIC_KEYS or key in META_KEYS:
            continue
        if isinstance(value, (dict, list)):
            continue
        add(f"{prefix}{key}", value)

    for nested_key in ("features", "ratios", "synthetic_features", "text_derived_features", "scores"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            for key, value in nested.items():
                if key in EXCLUDE_NUMERIC_KEYS:
                    continue
                add(f"{prefix}{key}", value)

    return out


def find_profile_rows(data: Any) -> Tuple[List[Dict[str, Any]], str]:
    """Find profile-level rows in report JSONs with several fallback schemas."""
    if isinstance(data, list):
        rows = [x for x in data if isinstance(x, dict)]
        return rows, "top-level-list"

    if not isinstance(data, dict):
        return [], "unsupported"

    for key in ("profiles", "rows", "profile_rows", "examples", "data"):
        value = data.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value, key

    # Some reports may store examples by group/rule. Only use this as a fallback,
    # because it may not contain all 400 profiles.
    for key in ("top_examples", "examples_by_rule", "changed_examples"):
        value = data.get(key)
        if isinstance(value, dict):
            rows: List[Dict[str, Any]] = []
            for group_value in value.values():
                if isinstance(group_value, list):
                    rows.extend(x for x in group_value if isinstance(x, dict))
                elif isinstance(group_value, dict):
                    for sub_value in group_value.values():
                        if isinstance(sub_value, list):
                            rows.extend(x for x in sub_value if isinstance(x, dict))
            if rows:
                return rows, key

    return [], "not-found"


def merge_report(
    table: Dict[str, Dict[str, Any]],
    task: str,
    report_path: Path,
    prefix: str,
    require_base_meta: bool = False,
) -> Dict[str, Any]:
    data = read_json(report_path)
    if data is None:
        return {"path": str(report_path), "exists": False, "rows": 0, "source": None}

    rows, source = find_profile_rows(data)
    used = 0
    missing_id = 0

    for row in rows:
        cid = normalize_id(first_value(row, ID_KEYS))
        if not cid:
            missing_id += 1
            continue
        rec = table.setdefault(cid, {"id": cid, "task": task})

        true_label = first_value(row, TRUE_KEYS)
        pred_label = first_value(row, PRED_KEYS)
        if true_label is not None and "true_label" not in rec:
            rec["true_label"] = str(true_label)
        if pred_label is not None and "pred_label" not in rec:
            rec["pred_label"] = str(pred_label)
        if "tokens" not in rec:
            tokens = first_value(row, ("tokens", "token_count"))
            if tokens is not None:
                val = safe_float(tokens)
                if val is not None:
                    rec["tokens"] = int(val)

        group = row.get("group") or row.get("error_group")
        if group and "group" not in rec:
            rec["group"] = str(group)

        for key, value in flatten_numeric_features(prefix, row).items():
            # Avoid accidental overwrites by later reports with same raw names.
            rec[key] = value
        used += 1

    return {
        "path": str(report_path),
        "exists": True,
        "rows": len(rows),
        "used": used,
        "missing_id": missing_id,
        "source": source,
    }


def finalize_records(table: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(table.values())
    for rec in rows:
        true_label = rec.get("true_label")
        pred_label = rec.get("pred_label")
        rec["is_correct"] = int(true_label == pred_label) if true_label and pred_label else ""
        if "group" not in rec:
            group = infer_group(true_label, pred_label)
            if group:
                rec["group"] = group
    rows.sort(key=lambda r: str(r.get("id", "")))
    return rows


def numeric_feature_names(rows: List[Dict[str, Any]]) -> List[str]:
    names = set()
    for row in rows:
        for key, value in row.items():
            if key in {"id", "task", "true_label", "pred_label", "group", "is_correct", "tokens"}:
                continue
            if safe_float(value) is not None:
                names.add(key)
    return sorted(names)


def accuracy(rows: List[Dict[str, Any]]) -> Optional[float]:
    valid = [r for r in rows if r.get("true_label") and r.get("pred_label")]
    if not valid:
        return None
    return sum(1 for r in valid if r["true_label"] == r["pred_label"]) / len(valid)


def macro_f1(rows: List[Dict[str, Any]]) -> Optional[float]:
    labels = sorted({str(r.get("true_label")) for r in rows if r.get("true_label")})
    if not labels:
        return None
    f1s = []
    for label in labels:
        tp = sum(1 for r in rows if r.get("true_label") == label and r.get("pred_label") == label)
        fp = sum(1 for r in rows if r.get("true_label") != label and r.get("pred_label") == label)
        fn = sum(1 for r in rows if r.get("true_label") == label and r.get("pred_label") != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def cohen_d(a: List[float], b: List[float]) -> Optional[float]:
    if not a or not b:
        return None
    ma, mb = mean(a), mean(b)
    if ma is None or mb is None:
        return None
    va, vb = stdev(a), stdev(b)
    pooled = math.sqrt((va * va + vb * vb) / 2.0)
    if pooled == 0:
        return 0.0
    return (ma - mb) / pooled


def group_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("group", "unknown"))] += 1
    return dict(sorted(counts.items()))


def group_feature_means(rows: List[Dict[str, Any]], features: List[str]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("group", "unknown"))].append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for group, group_rows in sorted(groups.items()):
        rec = {"n": len(group_rows)}
        for feature in features:
            vals = [safe_float(r.get(feature)) for r in group_rows]
            vals2 = [v for v in vals if v is not None]
            if vals2:
                rec[feature] = mean(vals2)
        out[group] = rec
    return out


def rank_features_for_groups(
    rows: List[Dict[str, Any]],
    features: List[str],
    max_items: int = 30,
) -> List[Dict[str, Any]]:
    """Rank features that separate each error group from the predicted-label correct group."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("group", "unknown"))].append(row)

    ranked: List[Dict[str, Any]] = []
    for group, a_rows in sorted(groups.items()):
        if group.startswith("correct_") or "_to_" not in group:
            continue
        true_label, pred_label = group.split("_to_", 1)
        pred_group = f"correct_{pred_label}"
        true_group = f"correct_{true_label}"
        for compare_group in (pred_group, true_group):
            b_rows = groups.get(compare_group, [])
            if not b_rows:
                continue
            for feature in features:
                a_vals = [safe_float(r.get(feature)) for r in a_rows]
                b_vals = [safe_float(r.get(feature)) for r in b_rows]
                a_vals = [v for v in a_vals if v is not None]
                b_vals = [v for v in b_vals if v is not None]
                if len(a_vals) < 3 or len(b_vals) < 3:
                    continue
                ma, mb = mean(a_vals), mean(b_vals)
                d = cohen_d(a_vals, b_vals)
                if ma is None or mb is None or d is None:
                    continue
                ranked.append({
                    "error_group": group,
                    "compare_group": compare_group,
                    "feature": feature,
                    "mean_error": ma,
                    "mean_compare": mb,
                    "difference": ma - mb,
                    "cohen_d": d,
                    "abs_d": abs(d),
                })
    ranked.sort(key=lambda x: x["abs_d"], reverse=True)
    return ranked[:max_items]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fixed = ["id", "task", "true_label", "pred_label", "group", "is_correct", "tokens"]
    others = sorted({k for r in rows for k in r.keys() if k not in fixed})
    fieldnames = fixed + others
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_task_markdown(
    path: Path,
    task: str,
    rows: List[Dict[str, Any]],
    features: List[str],
    report_meta: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
) -> None:
    acc = accuracy(rows)
    f1 = macro_f1(rows)
    counts = group_counts(rows)
    lines: List[str] = []
    lines.append(f"# Semantic Meta-Feature Table: {task}\n")
    lines.append(f"Script version: `{SCRIPT_VERSION}`\n")
    lines.append("> This file consolidates semantic, ratio, synthetic, and text-derived features. It is a diagnostic export, not a final tuned model.\n")
    lines.append("## Loaded reports\n")
    lines.append("| report | exists | rows | source |")
    lines.append("| --- | ---: | ---: | --- |")
    for meta in report_meta:
        name = Path(meta["path"]).name
        lines.append(f"| `{name}` | {meta.get('exists')} | {meta.get('rows', 0)} | `{meta.get('source')}` |")
    lines.append("\n## Global table summary\n")
    lines.append(f"Rows: **{len(rows)}**\n")
    lines.append(f"Numeric features: **{len(features)}**\n")
    lines.append(f"Baseline accuracy from merged labels/predictions: **{fmt(acc)}**, macro-F1: **{fmt(f1)}**.\n")
    lines.append("## Group counts\n")
    lines.append("| group | n |")
    lines.append("| --- | ---: |")
    for group, n in counts.items():
        lines.append(f"| `{group}` | {n} |")
    lines.append("\n## Strongest feature separations\n")
    lines.append("These compare each error group to both its predicted-label correct group and its true-label correct group.\n")
    lines.append("| error group | compare group | feature | mean error | mean compare | diff | d |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for item in ranked[:25]:
        lines.append(
            f"| `{item['error_group']}` | `{item['compare_group']}` | `{item['feature']}` | "
            f"{fmt(item['mean_error'])} | {fmt(item['mean_compare'])} | {fmt(item['difference'])} | {fmt(item['cohen_d'])} |"
        )
    lines.append("\n## Recommended use\n")
    lines.append(
        "Use the CSV/JSON output as a feature matrix for a future validation-based meta-model. "
        "Features that were mined from the analysed split, especially `text_*` features, should not be used as final test-tuned rules without re-mining them on a separate training/validation split."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_task(task: str, base_dir: Path, out_dir: Path) -> Dict[str, Any]:
    table: Dict[str, Dict[str, Any]] = {}
    report_specs = [
        (base_dir / "semantic_signals" / f"{task}_semantic_signal_report.json", "sig_"),
        (base_dir / "semantic_signal_ratios" / f"{task}_semantic_ratio_report.json", "ratio_"),
        (base_dir / "synthetic_semantic_features" / f"{task}_synthetic_semantic_features.json", "syn_"),
        (base_dir / "text_derived_semantic_features" / f"{task}_text_derived_semantic_features.json", "text_"),
    ]
    report_meta = []
    for path, prefix in report_specs:
        report_meta.append(merge_report(table, task, path, prefix))

    rows = finalize_records(table)
    features = numeric_feature_names(rows)
    ranked = rank_features_for_groups(rows, features)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{task}_semantic_meta_features.csv"
    json_path = out_dir / f"{task}_semantic_meta_features.json"
    md_path = out_dir / f"{task}_semantic_meta_feature_summary.md"

    write_csv(csv_path, rows)
    payload = {
        "script_version": SCRIPT_VERSION,
        "task": task,
        "rows": rows,
        "features": features,
        "report_meta": report_meta,
        "accuracy": accuracy(rows),
        "macro_f1": macro_f1(rows),
        "group_counts": group_counts(rows),
        "strongest_feature_separations": ranked,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_task_markdown(md_path, task, rows, features, report_meta, ranked)

    return {
        "task": task,
        "n_rows": len(rows),
        "n_features": len(features),
        "accuracy": accuracy(rows),
        "macro_f1": macro_f1(rows),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "md_path": str(md_path),
        "report_meta": report_meta,
    }


def write_global_summary(out_dir: Path, results: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Semantic Meta-Feature Export Summary\n")
    lines.append(f"Script version: `{SCRIPT_VERSION}`\n")
    lines.append("| task | rows | numeric features | accuracy | macro-F1 | CSV |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for r in results:
        csv_name = Path(r["csv_path"]).name
        lines.append(
            f"| `{r['task']}` | {r['n_rows']} | {r['n_features']} | {fmt(r['accuracy'])} | {fmt(r['macro_f1'])} | `{csv_name}` |"
        )
    lines.append("\n## Interpretation\n")
    lines.append(
        "This export consolidates the previous semantic analyses into one profile-level table per task. "
        "It is the bridge from post-hoc analysis to a proper validation-based meta-model: learn any lexicons, thresholds, or classifier weights only on training/validation data, then evaluate once on test."
    )
    (out_dir / "semantic_meta_feature_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consolidated semantic meta-feature tables.")
    parser.add_argument("--task", choices=TASKS, default=None, help="Optional single task to process.")
    args = parser.parse_args()

    base_dir = Path(hybrid_v4_output_dir) / "error_analysis"
    out_dir = base_dir / "semantic_meta_features"
    tasks = [args.task] if args.task else list(TASKS)

    print(f"[INFO] Running {SCRIPT_VERSION}")
    print(f"[INFO] Base dir: {base_dir}")
    print(f"[INFO] Output dir: {out_dir}")

    results = []
    for task in tasks:
        print(f"\n========== {task} ==========")
        result = build_task(task, base_dir, out_dir)
        results.append(result)
        print(f"[INFO] Rows: {result['n_rows']}")
        print(f"[INFO] Numeric features: {result['n_features']}")
        print(f"[INFO] CSV: {result['csv_path']}")

    write_global_summary(out_dir, results)
    print(f"\n[INFO] Summary: {out_dir / 'semantic_meta_feature_summary.md'}")


if __name__ == "__main__":
    main()
