"""
clean-semantic-ratios-v1

Analyze semantic signal ratios/differences from the Semantic Signal Reports.

Expected usage from project root:
    python -m Models.HybridV4.analysis.analyze_semantic_signal_ratios

The script reads the JSON reports created by analyze_semantic_error_signals.py:
    outputs/hybrid_v4/error_analysis/semantic_signals/gender_semantic_signal_report.json
    outputs/hybrid_v4/error_analysis/semantic_signals/occupation_semantic_signal_report.json

It writes ratio/difference reports to:
    outputs/hybrid_v4/error_analysis/semantic_signal_ratios/
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from _constants import hybrid_v4_output_dir
except Exception as exc:  # pragma: no cover - helpful project setup error
    raise RuntimeError(
        "Could not import hybrid_v4_output_dir from _constants.py. "
        "Run this from the project root or check PROJECT_ROOT resolution."
    ) from exc


EPS = 1e-9
DEFAULT_INPUT_DIR = os.path.join(
    hybrid_v4_output_dir,
    "error_analysis",
    "semantic_signals",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    hybrid_v4_output_dir,
    "error_analysis",
    "semantic_signal_ratios",
)


# Ratio/difference features to compute from per-profile semantic signal values.
# Formula operations supported below: diff, ratio, sum_minus.
RATIO_CONFIG = {
    "gender": {
        "features": [
            {
                "name": "female_minus_male_role",
                "kind": "diff",
                "a": "female_role_signal",
                "b": "male_role_signal",
                "description": "female_role_signal - male_role_signal",
            },
            {
                "name": "relationship_minus_male_role",
                "kind": "diff",
                "a": "relationship_signal",
                "b": "male_role_signal",
                "description": "relationship_signal - male_role_signal",
            },
            {
                "name": "family_minus_male_role",
                "kind": "diff",
                "a": "family_signal",
                "b": "male_role_signal",
                "description": "family_signal - male_role_signal",
            },
            {
                "name": "female_role_over_male_role",
                "kind": "ratio",
                "a": "female_role_signal",
                "b": "male_role_signal",
                "description": "female_role_signal / male_role_signal",
            },
            {
                "name": "relationship_over_male_role",
                "kind": "ratio",
                "a": "relationship_signal",
                "b": "male_role_signal",
                "description": "relationship_signal / male_role_signal",
            },
            {
                "name": "female_context_minus_male_role",
                "kind": "sum_minus",
                "sum": [
                    "family_signal",
                    "relationship_signal",
                    "female_role_signal",
                    "violence_against_women_signal",
                ],
                "minus": ["male_role_signal"],
                "description": "family + relationship + female_role + violence_against_women - male_role",
            },
            {
                "name": "violence_social_combo",
                "kind": "sum_minus",
                "sum": [
                    "violence_against_women_signal",
                    "emotion_social_issue_signal",
                ],
                "minus": [],
                "description": "violence_against_women_signal + emotion_social_issue_signal",
            },
        ],
        "comparisons": [
            ("female_to_male", "correct_male"),
            ("female_to_male", "correct_female"),
            ("male_to_female", "correct_female"),
            ("male_to_female", "correct_male"),
        ],
        "focus_groups": ["female_to_male", "male_to_female"],
        "focus_features": [
            "female_context_minus_male_role",
            "female_minus_male_role",
            "relationship_minus_male_role",
            "violence_social_combo",
        ],
    },
    "occupation": {
        "features": [
            {
                "name": "performer_minus_creator",
                "kind": "diff",
                "a": "performer_signal",
                "b": "creator_signal",
                "description": "performer_signal - creator_signal",
            },
            {
                "name": "creator_minus_performer",
                "kind": "diff",
                "a": "creator_signal",
                "b": "performer_signal",
                "description": "creator_signal - performer_signal",
            },
            {
                "name": "fan_minus_creator",
                "kind": "diff",
                "a": "fan_community_signal",
                "b": "creator_signal",
                "description": "fan_community_signal - creator_signal",
            },
            {
                "name": "performer_fan_minus_creator",
                "kind": "sum_minus",
                "sum": ["performer_signal", "fan_community_signal"],
                "minus": ["creator_signal"],
                "description": "performer_signal + fan_community_signal - creator_signal",
            },
            {
                "name": "sports_minus_performer",
                "kind": "diff",
                "a": "sports_signal",
                "b": "performer_signal",
                "description": "sports_signal - performer_signal",
            },
            {
                "name": "politics_minus_creator",
                "kind": "diff",
                "a": "politics_signal",
                "b": "creator_signal",
                "description": "politics_signal - creator_signal",
            },
            {
                "name": "performer_over_creator",
                "kind": "ratio",
                "a": "performer_signal",
                "b": "creator_signal",
                "description": "performer_signal / creator_signal",
            },
            {
                "name": "fan_over_creator",
                "kind": "ratio",
                "a": "fan_community_signal",
                "b": "creator_signal",
                "description": "fan_community_signal / creator_signal",
            },
            {
                "name": "sports_over_performer",
                "kind": "ratio",
                "a": "sports_signal",
                "b": "performer_signal",
                "description": "sports_signal / performer_signal",
            },
            {
                "name": "politics_over_creator",
                "kind": "ratio",
                "a": "politics_signal",
                "b": "creator_signal",
                "description": "politics_signal / creator_signal",
            },
        ],
        "comparisons": [
            ("creator_to_performer", "correct_creator"),
            ("creator_to_performer", "correct_performer"),
            ("creator_to_politics", "correct_creator"),
            ("creator_to_sports", "correct_creator"),
            ("sports_to_performer", "correct_sports"),
            ("sports_to_performer", "correct_performer"),
            ("politics_to_creator", "correct_politics"),
        ],
        "focus_groups": [
            "creator_to_performer",
            "creator_to_politics",
            "creator_to_sports",
            "sports_to_performer",
            "politics_to_creator",
        ],
        "focus_features": [
            "performer_fan_minus_creator",
            "performer_minus_creator",
            "fan_minus_creator",
            "sports_minus_performer",
            "politics_minus_creator",
        ],
    },
}


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_get_signal(profile: dict, key: str) -> float:
    values = profile.get("signal_values", {}) or {}
    value = values.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compute_feature(profile: dict, spec: dict) -> float:
    kind = spec["kind"]

    if kind == "diff":
        return safe_get_signal(profile, spec["a"]) - safe_get_signal(profile, spec["b"])

    if kind == "ratio":
        numerator = safe_get_signal(profile, spec["a"])
        denominator = safe_get_signal(profile, spec["b"])
        return numerator / (denominator + EPS)

    if kind == "sum_minus":
        plus = sum(safe_get_signal(profile, key) for key in spec.get("sum", []))
        minus = sum(safe_get_signal(profile, key) for key in spec.get("minus", []))
        return plus - minus

    raise ValueError(f"Unknown feature kind: {kind}")


def enrich_profiles(profiles: Iterable[dict], feature_specs: List[dict]) -> List[dict]:
    enriched = []

    for profile in profiles:
        row = dict(profile)
        ratio_values = {}
        for spec in feature_specs:
            ratio_values[spec["name"]] = compute_feature(profile, spec)
        row["ratio_values"] = ratio_values
        enriched.append(row)

    return enriched


def group_profiles(profiles: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for profile in profiles:
        grouped[str(profile.get("group", "unknown"))].append(profile)
    return dict(grouped)


def summarize_group(rows: List[dict], feature_names: List[str]) -> dict:
    summary = {"n": len(rows)}
    for name in feature_names:
        vals = [float(row.get("ratio_values", {}).get(name, 0.0)) for row in rows]
        summary[name] = mean(vals) if vals else 0.0
        summary[f"{name}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
    return summary


def compute_group_stats(grouped: Dict[str, List[dict]], feature_names: List[str]) -> Dict[str, dict]:
    return {
        group: summarize_group(rows, feature_names)
        for group, rows in sorted(grouped.items(), key=lambda item: item[0])
    }


def pooled_std(values_a: List[float], values_b: List[float]) -> float:
    if not values_a or not values_b:
        return 0.0
    std_a = pstdev(values_a) if len(values_a) > 1 else 0.0
    std_b = pstdev(values_b) if len(values_b) > 1 else 0.0
    return math.sqrt((std_a ** 2 + std_b ** 2) / 2.0)


def compare_groups(
    grouped: Dict[str, List[dict]],
    comparisons: List[Tuple[str, str]],
    feature_names: List[str],
) -> List[dict]:
    results = []
    for group_a, group_b in comparisons:
        rows_a = grouped.get(group_a, [])
        rows_b = grouped.get(group_b, [])
        result = {
            "group_a": group_a,
            "group_b": group_b,
            "n_a": len(rows_a),
            "n_b": len(rows_b),
            "features": {},
        }

        for name in feature_names:
            vals_a = [float(row.get("ratio_values", {}).get(name, 0.0)) for row in rows_a]
            vals_b = [float(row.get("ratio_values", {}).get(name, 0.0)) for row in rows_b]
            mean_a = mean(vals_a) if vals_a else 0.0
            mean_b = mean(vals_b) if vals_b else 0.0
            diff = mean_a - mean_b
            sd = pooled_std(vals_a, vals_b)
            result["features"][name] = {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "difference_a_minus_b": diff,
                "cohens_d_approx": diff / (sd + EPS) if sd else 0.0,
            }
        results.append(result)
    return results


def top_examples(
    grouped: Dict[str, List[dict]],
    focus_groups: List[str],
    focus_features: List[str],
    top_k: int,
) -> Dict[str, Dict[str, List[dict]]]:
    output: Dict[str, Dict[str, List[dict]]] = {}

    for group in focus_groups:
        rows = grouped.get(group, [])
        output[group] = {}
        for feature in focus_features:
            sorted_rows = sorted(
                rows,
                key=lambda row: float(row.get("ratio_values", {}).get(feature, 0.0)),
                reverse=True,
            )[:top_k]
            output[group][feature] = [compact_profile(row, ratio_feature=feature) for row in sorted_rows]
    return output


def compact_profile(profile: dict, ratio_feature: str = None) -> dict:
    top_hits = profile.get("top_keyword_hits", []) or []
    compact = {
        "celebrity_id": profile.get("celebrity_id"),
        "true_label": profile.get("true_label"),
        "pred_label": profile.get("pred_label"),
        "group": profile.get("group"),
        "token_count": profile.get("token_count", 0),
        "signal_sum": profile.get("signal_sum", 0.0),
        "ratio_values": profile.get("ratio_values", {}),
        "top_keyword_hits": top_hits[:12],
    }
    if ratio_feature:
        compact["ranking_feature"] = ratio_feature
        compact["ranking_value"] = compact["ratio_values"].get(ratio_feature, 0.0)
    return compact


def fmt(value: float, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0.0000"


def hits_to_text(hits: List) -> str:
    parts = []
    for item in hits[:8]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            parts.append(f"{item[0]}:{item[1]}")
    return ", ".join(parts)


def write_markdown_report(
    path: str,
    task: str,
    source_report_path: str,
    n_profiles: int,
    feature_specs: List[dict],
    group_stats: Dict[str, dict],
    comparisons: List[dict],
    top_examples_by_group: Dict[str, Dict[str, List[dict]]],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    feature_names = [spec["name"] for spec in feature_specs]

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Semantic Signal Ratio Analysis: {task}\n\n")
        f.write(f"Source semantic report: `{source_report_path}`\n\n")
        f.write(f"Profiles: **{n_profiles}**\n\n")

        f.write("## Ratio-/Differenzfeatures\n\n")
        f.write("| feature | definition |\n")
        f.write("| --- | --- |\n")
        for spec in feature_specs:
            f.write(f"| `{spec['name']}` | {spec.get('description', '')} |\n")

        f.write("\n## Gruppendurchschnitte\n\n")
        f.write("| group | n | " + " | ".join(feature_names) + " |\n")
        f.write("| --- | ---: | " + " | ".join(["---:" for _ in feature_names]) + " |\n")
        for group, stats in group_stats.items():
            values = " | ".join(fmt(stats.get(name, 0.0)) for name in feature_names)
            f.write(f"| {group} | {stats.get('n', 0)} | {values} |\n")

        f.write("\n## Vergleich relevanter Fehlergruppen\n\n")
        for comp in comparisons:
            f.write(
                f"### `{comp['group_a']}` vs `{comp['group_b']}` "
                f"(n={comp['n_a']} vs n={comp['n_b']})\n\n"
            )
            f.write("| feature | mean A | mean B | A - B | approx. Cohen d |\n")
            f.write("| --- | ---: | ---: | ---: | ---: |\n")
            # Sort by absolute difference so the most interesting features are at the top.
            items = sorted(
                comp["features"].items(),
                key=lambda item: abs(item[1]["difference_a_minus_b"]),
                reverse=True,
            )
            for name, values in items:
                f.write(
                    f"| `{name}` | {fmt(values['mean_a'])} | {fmt(values['mean_b'])} | "
                    f"{fmt(values['difference_a_minus_b'])} | {fmt(values['cohens_d_approx'])} |\n"
                )
            f.write("\n")

        f.write("## Top-Beispiele nach Ratio-/Differenzfeatures\n\n")
        for group, feature_map in top_examples_by_group.items():
            f.write(f"### {group}\n\n")
            for feature, rows in feature_map.items():
                f.write(f"#### Ranking nach `{feature}`\n\n")
                if not rows:
                    f.write("Keine Beispiele gefunden.\n\n")
                    continue
                f.write("| id | true | pred | tokens | value | signal_sum | top keyword hits |\n")
                f.write("| --- | --- | --- | ---: | ---: | ---: | --- |\n")
                for row in rows:
                    f.write(
                        f"| {row.get('celebrity_id')} | {row.get('true_label')} | {row.get('pred_label')} | "
                        f"{row.get('token_count', 0)} | {fmt(row.get('ranking_value', 0.0))} | "
                        f"{fmt(row.get('signal_sum', 0.0))} | {hits_to_text(row.get('top_keyword_hits', []))} |\n"
                    )
                f.write("\n")

        f.write("## Kurze Interpretation\n\n")
        if task == "occupation":
            f.write(
                "Für `creator_to_performer` sind besonders `performer_minus_creator`, "
                "`fan_minus_creator` und `performer_fan_minus_creator` relevant. "
                "Wenn diese Werte klar über `correct_creator` liegen, stützt das die These, "
                "dass Creator mit performer-/fan-naher Follower-Sprache in Richtung Performer kippen.\n"
            )
        elif task == "gender":
            f.write(
                "Für `female_to_male` sind besonders `female_context_minus_male_role`, "
                "`female_minus_male_role` und `relationship_minus_male_role` relevant. "
                "Wenn diese Werte nicht klar über den Vergleichsgruppen liegen, sind einfache Keyword-Signale "
                "für Gender wahrscheinlich eher erklärend als direkt korrigierend.\n"
            )


def analyze_task(task: str, input_dir: str, output_dir: str, top_k: int) -> dict:
    config = RATIO_CONFIG[task]
    input_path = os.path.join(input_dir, f"{task}_semantic_signal_report.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Missing semantic signal report for {task}: {input_path}. "
            "Run analyze_semantic_error_signals.py first."
        )

    report = load_json(input_path)
    profiles = report.get("profiles", [])
    if not profiles:
        raise ValueError(
            f"No per-profile rows found in {input_path}. "
            "Expected key 'profiles' from analyze_semantic_error_signals.py."
        )

    feature_specs = config["features"]
    feature_names = [spec["name"] for spec in feature_specs]
    enriched = enrich_profiles(profiles, feature_specs)
    grouped = group_profiles(enriched)
    group_stats = compute_group_stats(grouped, feature_names)
    comparisons = compare_groups(grouped, config["comparisons"], feature_names)
    top = top_examples(
        grouped=grouped,
        focus_groups=config["focus_groups"],
        focus_features=config["focus_features"],
        top_k=top_k,
    )

    result = {
        "task": task,
        "source_report_path": input_path,
        "n_profiles": len(enriched),
        "feature_definitions": feature_specs,
        "group_stats": group_stats,
        "comparisons": comparisons,
        "top_examples": top,
        "profiles": [compact_profile(row) for row in enriched],
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{task}_semantic_ratio_report.json")
    md_path = os.path.join(output_dir, f"{task}_semantic_ratio_report.md")

    write_json(result, json_path)
    write_markdown_report(
        path=md_path,
        task=task,
        source_report_path=input_path,
        n_profiles=len(enriched),
        feature_specs=feature_specs,
        group_stats=group_stats,
        comparisons=comparisons,
        top_examples_by_group=top,
    )

    print(f"[OK] Saved {task} JSON: {json_path}")
    print(f"[OK] Saved {task} Markdown: {md_path}")
    return result


def resolve_tasks(task_arg: str) -> List[str]:
    if task_arg == "all":
        return ["gender", "occupation"]
    return [task_arg]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze ratio/difference features from HybridV4 semantic signal reports."
    )
    parser.add_argument(
        "--task",
        choices=["gender", "occupation", "all"],
        default="all",
        help="Which semantic report to analyze.",
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing *_semantic_signal_report.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for semantic ratio reports.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top examples per focus group/feature.",
    )
    args = parser.parse_args()

    print("[INFO] Running semantic signal ratio analysis")
    print(f"[INFO] input_dir={args.input_dir}")
    print(f"[INFO] output_dir={args.output_dir}")

    for task in resolve_tasks(args.task):
        print(f"\n========== {task} ==========")
        analyze_task(
            task=task,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
