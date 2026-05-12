"""
Analyze semantic correctability candidates for HybridV4 error groups.

Run from project root:
    python -m Models.HybridV4.analysis.analyze_correctability_candidates

This script reads the semantic signal reports created by
`analyze_semantic_error_signals.py`, builds centroid profiles for correctly
classified classes, and checks whether misclassified profiles look semantically
closer to their true class or to the predicted class.

Version: correctability-candidates-v1
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from _constants import hybrid_v4_output_dir  # noqa: E402


TASKS = ["gender", "occupation"]

SIGNAL_REPORT_DIR = os.path.join(
    hybrid_v4_output_dir,
    "error_analysis",
    "semantic_signals",
)

OUTPUT_DIR = os.path.join(
    hybrid_v4_output_dir,
    "error_analysis",
    "correctability_candidates",
)


GENDER_RATIO_FEATURES = {
    "female_minus_male_role": "female_role_signal - male_role_signal",
    "relationship_minus_male_role": "relationship_signal - male_role_signal",
    "family_minus_male_role": "family_signal - male_role_signal",
    "female_context_minus_male_role": (
        "family_signal + relationship_signal + female_role_signal + "
        "violence_against_women_signal - male_role_signal"
    ),
    "violence_social_combo": (
        "violence_against_women_signal + emotion_social_issue_signal"
    ),
}

OCCUPATION_RATIO_FEATURES = {
    "performer_minus_creator": "performer_signal - creator_signal",
    "creator_minus_performer": "creator_signal - performer_signal",
    "fan_minus_creator": "fan_community_signal - creator_signal",
    "performer_fan_minus_creator": (
        "performer_signal + fan_community_signal - creator_signal"
    ),
    "sports_minus_performer": "sports_signal - performer_signal",
    "politics_minus_creator": "politics_signal - creator_signal",
}


TASK_FEATURES = {
    "gender": [
        "female_minus_male_role",
        "relationship_minus_male_role",
        "family_minus_male_role",
        "female_context_minus_male_role",
        "violence_social_combo",
    ],
    "occupation": [
        "performer_minus_creator",
        "creator_minus_performer",
        "fan_minus_creator",
        "performer_fan_minus_creator",
        "sports_minus_performer",
        "politics_minus_creator",
    ],
}


FOCUS_COMPARISONS = {
    "gender": [
        ("female_to_male", "correct_female", "correct_male"),
        ("male_to_female", "correct_male", "correct_female"),
    ],
    "occupation": [
        ("creator_to_performer", "correct_creator", "correct_performer"),
        ("creator_to_politics", "correct_creator", "correct_politics"),
        ("creator_to_sports", "correct_creator", "correct_sports"),
        ("sports_to_performer", "correct_sports", "correct_performer"),
        ("politics_to_creator", "correct_politics", "correct_creator"),
    ],
}


TOP_K = 20


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def top_hits_text(row: Dict[str, Any], limit: int = 8) -> str:
    hits = row.get("top_keyword_hits", [])
    parts = []
    for item in hits[:limit]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            parts.append(f"{item[0]}:{item[1]}")
    return ", ".join(parts)


def has_profile_shape(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    return (
        "group" in obj
        and ("true_label" in obj or "true" in obj)
        and ("pred_label" in obj or "pred" in obj)
        and ("signal_values" in obj or any(k.endswith("_signal") for k in obj))
    )


def iter_profile_rows(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Recursively extract profile-level rows from semantic report JSON.

    The current semantic report usually stores rows in top_examples, but this
    function also supports future keys such as records/examples/profiles.
    """
    if has_profile_shape(obj):
        yield obj
        return

    if isinstance(obj, dict):
        for value in obj.values():
            yield from iter_profile_rows(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_profile_rows(value)


def normalize_profile_row(row: Dict[str, Any], task: str) -> Dict[str, Any]:
    signal_values = row.get("signal_values", {})
    if not isinstance(signal_values, dict):
        signal_values = {}

    flat = {
        "celebrity_id": str(row.get("celebrity_id", row.get("id", ""))),
        "task": task,
        "true_label": str(row.get("true_label", row.get("true", ""))),
        "pred_label": str(row.get("pred_label", row.get("pred", ""))),
        "group": str(row.get("group", "")),
        "tweet_count": int(safe_float(row.get("tweet_count", 0), 0)),
        "token_count": int(safe_float(row.get("token_count", row.get("tokens", 0)), 0)),
        "signal_sum": safe_float(row.get("signal_sum", 0.0)),
        "top_keyword_hits": row.get("top_keyword_hits", []),
    }

    for key, value in signal_values.items():
        flat[key] = safe_float(value)

    # Some future report variants may already contain flat signal values.
    for key, value in row.items():
        if isinstance(key, str) and key.endswith("_signal"):
            flat[key] = safe_float(value)

    add_ratio_features(flat, task)
    return flat


def add_ratio_features(row: Dict[str, Any], task: str) -> None:
    if task == "gender":
        female = safe_float(row.get("female_role_signal"))
        male = safe_float(row.get("male_role_signal"))
        relationship = safe_float(row.get("relationship_signal"))
        family = safe_float(row.get("family_signal"))
        violence = safe_float(row.get("violence_against_women_signal"))
        social = safe_float(row.get("emotion_social_issue_signal"))

        row["female_minus_male_role"] = female - male
        row["relationship_minus_male_role"] = relationship - male
        row["family_minus_male_role"] = family - male
        row["female_context_minus_male_role"] = (
            family + relationship + female + violence - male
        )
        row["violence_social_combo"] = violence + social

    elif task == "occupation":
        creator = safe_float(row.get("creator_signal"))
        performer = safe_float(row.get("performer_signal"))
        fan = safe_float(row.get("fan_community_signal"))
        sports = safe_float(row.get("sports_signal"))
        politics = safe_float(row.get("politics_signal"))

        row["performer_minus_creator"] = performer - creator
        row["creator_minus_performer"] = creator - performer
        row["fan_minus_creator"] = fan - creator
        row["performer_fan_minus_creator"] = performer + fan - creator
        row["sports_minus_performer"] = sports - performer
        row["politics_minus_creator"] = politics - creator


def unique_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for row in rows:
        key = (
            row.get("task"),
            row.get("celebrity_id"),
            row.get("group"),
            row.get("true_label"),
            row.get("pred_label"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 1.0
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    s = math.sqrt(var)
    return s if s > 1e-12 else 1.0


def build_scaler(rows: List[Dict[str, Any]], features: List[str]) -> Dict[str, Tuple[float, float]]:
    scaler = {}
    for feat in features:
        vals = [safe_float(row.get(feat)) for row in rows]
        scaler[feat] = (mean(vals), std(vals))
    return scaler


def vectorize(
    row: Dict[str, Any],
    features: List[str],
    scaler: Dict[str, Tuple[float, float]],
) -> List[float]:
    vec = []
    for feat in features:
        mu, sigma = scaler[feat]
        vec.append((safe_float(row.get(feat)) - mu) / sigma)
    return vec


def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def centroid(
    rows: List[Dict[str, Any]],
    features: List[str],
    scaler: Dict[str, Tuple[float, float]],
) -> Optional[List[float]]:
    if not rows:
        return None
    vectors = [vectorize(row, features, scaler) for row in rows]
    return [mean([vec[i] for vec in vectors]) for i in range(len(features))]


def group_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    return dict(groups)


def summarize_features(rows: List[Dict[str, Any]], features: List[str]) -> Dict[str, float]:
    return {feat: mean([safe_float(row.get(feat)) for row in rows]) for feat in features}


def analyze_task(task: str) -> Dict[str, Any]:
    report_path = os.path.join(SIGNAL_REPORT_DIR, f"{task}_semantic_signal_report.json")
    data = load_json(report_path)

    raw_rows = [normalize_profile_row(row, task) for row in iter_profile_rows(data)]
    rows = unique_rows(raw_rows)
    features = TASK_FEATURES[task]
    groups = group_rows(rows)
    scaler = build_scaler(rows, features)

    correct_centroids = {}
    correct_group_sizes = {}
    for group, items in groups.items():
        if not group.startswith("correct_"):
            continue
        label = group.replace("correct_", "", 1)
        correct_centroids[label] = centroid(items, features, scaler)
        correct_group_sizes[label] = len(items)

    candidate_rows = []
    comparison_summaries = []

    for error_group, true_group, pred_group in FOCUS_COMPARISONS.get(task, []):
        error_items = groups.get(error_group, [])
        true_label = true_group.replace("correct_", "", 1)
        pred_label = pred_group.replace("correct_", "", 1)
        true_centroid = correct_centroids.get(true_label)
        pred_centroid = correct_centroids.get(pred_label)

        if not error_items or true_centroid is None or pred_centroid is None:
            comparison_summaries.append({
                "error_group": error_group,
                "n": len(error_items),
                "status": "missing_group_or_centroid",
            })
            continue

        analyzed = []
        for row in error_items:
            vec = vectorize(row, features, scaler)
            dist_true = euclidean(vec, true_centroid)
            dist_pred = euclidean(vec, pred_centroid)
            margin = dist_pred - dist_true
            closer_to = "true_label" if margin > 0 else "pred_label"

            out = dict(row)
            out.update({
                "error_group": error_group,
                "correct_true_group": true_group,
                "correct_pred_group": pred_group,
                "distance_to_true_centroid": dist_true,
                "distance_to_pred_centroid": dist_pred,
                "correctability_margin": margin,
                "closer_to": closer_to,
            })
            analyzed.append(out)
            candidate_rows.append(out)

        n_true = sum(1 for row in analyzed if row["closer_to"] == "true_label")
        n_pred = len(analyzed) - n_true
        comparison_summaries.append({
            "error_group": error_group,
            "n": len(analyzed),
            "true_group": true_group,
            "pred_group": pred_group,
            "closer_to_true_n": n_true,
            "closer_to_true_rate": n_true / len(analyzed) if analyzed else 0.0,
            "closer_to_pred_n": n_pred,
            "mean_distance_to_true": mean([r["distance_to_true_centroid"] for r in analyzed]),
            "mean_distance_to_pred": mean([r["distance_to_pred_centroid"] for r in analyzed]),
            "mean_correctability_margin": mean([r["correctability_margin"] for r in analyzed]),
            "feature_means_error_group": summarize_features(analyzed, features),
            "feature_means_true_group": summarize_features(groups.get(true_group, []), features),
            "feature_means_pred_group": summarize_features(groups.get(pred_group, []), features),
        })

    candidate_rows.sort(key=lambda r: r.get("correctability_margin", -999), reverse=True)

    return {
        "task": task,
        "source_report_path": report_path,
        "n_extracted_profile_rows": len(rows),
        "features": features,
        "feature_definitions": GENDER_RATIO_FEATURES if task == "gender" else OCCUPATION_RATIO_FEATURES,
        "group_sizes": {group: len(items) for group, items in sorted(groups.items())},
        "correct_group_sizes": correct_group_sizes,
        "comparison_summaries": comparison_summaries,
        "candidate_rows": candidate_rows,
        "note": (
            "Rows are extracted recursively from the semantic report JSON. If the semantic report "
            "contains only top_examples rather than all profiles, this analysis is qualitative."
        ),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(result: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    task = result["task"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Correctability Candidate Analysis: {task}\n\n")
        f.write(f"Source semantic report: `{result['source_report_path']}`\n\n")
        f.write(f"Extracted profile rows: **{result['n_extracted_profile_rows']}**\n\n")
        f.write(
            "> Interpretation: positive `correctability_margin` means the error case is closer "
            "to the centroid of correctly classified true-label examples than to the centroid "
            "of correctly classified predicted-label examples.\n\n"
        )

        f.write("## Features\n\n")
        f.write("| feature | definition |\n")
        f.write("| --- | --- |\n")
        definitions = result.get("feature_definitions", {})
        for feat in result["features"]:
            f.write(f"| `{feat}` | {definitions.get(feat, '')} |\n")
        f.write("\n")

        f.write("## Group sizes\n\n")
        f.write("| group | n |\n")
        f.write("| --- | ---: |\n")
        for group, n in result["group_sizes"].items():
            f.write(f"| `{group}` | {n} |\n")
        f.write("\n")

        f.write("## Correctability summary\n\n")
        f.write(
            "| error_group | n | closer_to_true | rate | mean_dist_true | "
            "mean_dist_pred | mean_margin |\n"
        )
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for summary in result["comparison_summaries"]:
            if summary.get("status"):
                f.write(
                    f"| `{summary['error_group']}` | {summary['n']} | - | - | - | - | - |\n"
                )
                continue
            f.write(
                f"| `{summary['error_group']}` "
                f"| {summary['n']} "
                f"| {summary['closer_to_true_n']} "
                f"| {fmt(summary['closer_to_true_rate'])} "
                f"| {fmt(summary['mean_distance_to_true'])} "
                f"| {fmt(summary['mean_distance_to_pred'])} "
                f"| {fmt(summary['mean_correctability_margin'])} |\n"
            )
        f.write("\n")

        f.write("## Feature means by comparison\n\n")
        for summary in result["comparison_summaries"]:
            if summary.get("status"):
                continue
            f.write(f"### `{summary['error_group']}`\n\n")
            f.write("| feature | error_group | correct_true | correct_pred |\n")
            f.write("| --- | ---: | ---: | ---: |\n")
            for feat in result["features"]:
                f.write(
                    f"| `{feat}` "
                    f"| {fmt(summary['feature_means_error_group'].get(feat, 0.0))} "
                    f"| {fmt(summary['feature_means_true_group'].get(feat, 0.0))} "
                    f"| {fmt(summary['feature_means_pred_group'].get(feat, 0.0))} |\n"
                )
            f.write("\n")

        f.write("## Top correction candidates\n\n")
        f.write(
            "These are errors whose ratio-feature vector is closest to the correct true-label centroid.\n\n"
        )
        f.write(
            "| id | group | true | pred | margin | dist_true | dist_pred | tokens | key ratios | top keyword hits |\n"
        )
        f.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")

        for row in result["candidate_rows"][:TOP_K]:
            key_ratios = ", ".join(
                f"{feat}={fmt(row.get(feat, 0.0), 2)}" for feat in result["features"][:3]
            )
            f.write(
                f"| {row.get('celebrity_id', '')} "
                f"| `{row.get('error_group', row.get('group', ''))}` "
                f"| {row.get('true_label', '')} "
                f"| {row.get('pred_label', '')} "
                f"| {fmt(row.get('correctability_margin', 0.0))} "
                f"| {fmt(row.get('distance_to_true_centroid', 0.0))} "
                f"| {fmt(row.get('distance_to_pred_centroid', 0.0))} "
                f"| {row.get('token_count', 0)} "
                f"| {key_ratios} "
                f"| {top_hits_text(row)} |\n"
            )

        f.write("\n## Short interpretation\n\n")
        if task == "gender":
            f.write(
                "For gender, pay special attention to `relationship_minus_male_role`. "
                "If uncertain cases move toward the opposite label mainly through relationship language, "
                "this is useful as an explanatory signal, but it should be treated carefully before being used "
                "as a correction rule.\n"
            )
        else:
            f.write(
                "For occupation, `creator_to_performer` should be checked against both `correct_creator` "
                "and `correct_performer`. If candidates are closer to `correct_performer`, the error is "
                "semantically hard to correct with these features alone; if closer to `correct_creator`, "
                "they are promising qualitative correction candidates.\n"
            )


def write_json(result: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze semantic correctability candidates for HybridV4."
    )
    parser.add_argument(
        "--task",
        choices=["all", "gender", "occupation"],
        default="all",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tasks = TASKS if args.task == "all" else [args.task]
    print("[INFO] Running correctability candidate analysis")
    print(f"[INFO] input_dir={SIGNAL_REPORT_DIR}")
    print(f"[INFO] output_dir={OUTPUT_DIR}")

    for task in tasks:
        print(f"\n========== {task} ==========")
        result = analyze_task(task)
        json_path = os.path.join(OUTPUT_DIR, f"{task}_correctability_candidates.json")
        md_path = os.path.join(OUTPUT_DIR, f"{task}_correctability_candidates.md")
        write_json(result, json_path)
        write_markdown(result, md_path)
        print(f"[OK] extracted rows: {result['n_extracted_profile_rows']}")
        print(f"[OK] saved: {md_path}")
        print(f"[OK] saved: {json_path}")


if __name__ == "__main__":
    main()
