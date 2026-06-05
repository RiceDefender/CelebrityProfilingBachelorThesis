import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

# -------------------------------------------------------------------
# Project import setup
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from _constants import (  # noqa: E402
    test_feeds_path,
    hybrid_v4_output_dir,
    hybrid_v4_fusion_predictions_dir,
)

SCRIPT_VERSION = "2026-05-11-clean-semantic-signals-v1"

TOKEN_RE = re.compile(r"[#@]?[A-Za-z0-9_]{2,}", re.UNICODE)

DEFAULT_PREDICTION_FILES = {
    "gender": "gender_test_fair_weighted_fusion_predictions_best_val_w_0p28_0p56_0p17_alpha_0p000.json",
    "occupation": "occupation_test_fair_weighted_fusion_predictions_best_val_w_0p64_0p21_0p14_alpha_0p000.json",
}

GENDER_SIGNALS = {
    "family_signal": [
        "mother", "father", "daughter", "son", "sister", "brother", "family",
        "child", "children", "mom", "dad",
    ],
    "relationship_signal": [
        "wife", "husband", "marriage", "married", "boyfriend", "girlfriend",
        "relationship", "love", "cheating", "divorce",
    ],
    "female_role_signal": [
        "woman", "women", "girl", "girls", "lady", "ladies", "female", "feminist",
        "mother", "daughter", "wife",
    ],
    "violence_against_women_signal": [
        "abuse", "abused", "rape", "raped", "assault", "beaten", "domestic violence",
        "harassment", "victim",
    ],
    "emotion_social_issue_signal": [
        "shame", "respect", "disgusting", "heartbreaking", "outrage", "justice",
        "protect", "support",
    ],
    "male_role_signal": [
        "man", "men", "male", "husband", "father", "son", "brother", "boyfriend",
    ],
}

OCCUPATION_SIGNALS = {
    "creator_signal": [
        "writer", "writing", "author", "book", "blog", "blogger", "design", "designer",
        "art", "artist", "photography", "content", "creator", "handmade", "vintage",
        "shop", "etsy",
    ],
    "performer_signal": [
        "actor", "actress", "singer", "music", "movie", "film", "show", "tv", "drama",
        "concert", "album", "celebrity", "fan", "fans", "fandom",
    ],
    "sports_signal": [
        "match", "game", "team", "player", "coach", "league", "cup", "goal", "football",
        "basketball", "cricket", "tennis", "olympic", "tournament",
    ],
    "politics_signal": [
        "election", "vote", "government", "minister", "president", "parliament", "party",
        "policy", "campaign", "democracy", "bjp", "congress", "trump", "brexit",
    ],
    "business_branding_signal": [
        "startup", "marketing", "branding", "ceo", "founder", "growth", "business",
        "entrepreneur", "digital", "social media",
    ],
    "fan_community_signal": [
        "fan", "fans", "army", "fandom", "stan", "vote", "support", "idol", "bias",
        "comeback", "bbmas", "mtv",
    ],
}

DISPLAY_GROUPS = {
    "gender": [
        "correct_female", "correct_male", "female_to_male", "male_to_female",
    ],
    "occupation": [
        "correct_creator", "creator_to_performer", "creator_to_politics", "creator_to_sports",
        "correct_performer", "performer_to_creator", "sports_to_performer", "politics_to_creator",
    ],
}

PRED_KEYS = ["fusion_pred_label", "pred_label", "prediction", "pred", "predicted", "predicted_label", "y_pred"]
TRUE_KEYS = ["true_label", "gold_label", "label", "y_true"]
ID_KEYS = ["celebrity_id", "id"]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_first(row: dict, keys: List[str]):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def celebrity_id(row: dict) -> str:
    value = get_first(row, ID_KEYS)
    if value is None:
        raise KeyError(f"No celebrity id field found. Keys: {sorted(row.keys())}")
    return str(value)


def prediction_label(row: dict) -> str:
    value = get_first(row, PRED_KEYS)
    if value is None:
        raise KeyError(f"No prediction label field found. Keys: {sorted(row.keys())}")
    return str(value)


def true_label(row: dict) -> str:
    value = get_first(row, TRUE_KEYS)
    if value is None:
        raise KeyError(f"No true label field found. Keys: {sorted(row.keys())}")
    return str(value)


def normalize_prediction_rows(data: Any) -> List[dict]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    if isinstance(data, dict):
        for key in ["predictions", "rows", "data", "results", "items"]:
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]

        if all(isinstance(v, dict) for v in data.values()):
            rows = []
            for key, value in data.items():
                row = dict(value)
                row.setdefault("celebrity_id", key)
                rows.append(row)
            return rows

    raise ValueError(f"Unsupported prediction JSON shape: {type(data)}")


def load_predictions(path: str) -> List[dict]:
    rows = normalize_prediction_rows(load_json(path))
    usable = []
    skipped = Counter()

    for row in rows:
        cid = get_first(row, ID_KEYS)
        gold = get_first(row, TRUE_KEYS)
        pred = get_first(row, PRED_KEYS)

        if cid is None:
            skipped["missing_id"] += 1
            continue
        if gold is None:
            skipped["missing_true_label"] += 1
            continue
        if pred is None:
            skipped["missing_prediction"] += 1
            continue

        new_row = dict(row)
        new_row["celebrity_id"] = str(cid)
        new_row["true_label"] = str(gold)
        new_row["pred_label"] = str(pred)
        usable.append(new_row)

    print(f"[INFO] Loaded predictions: {len(usable)} from {path}")
    if skipped:
        print(f"[INFO] Skipped prediction rows: {dict(skipped)}")
    if not usable:
        sample = rows[:3]
        raise ValueError(
            "No usable predictions. Need celebrity_id/id, true_label/label and "
            f"fusion_pred_label/prediction. First rows: {sample}"
        )
    return usable


def load_pan_ndjson(path: str) -> Dict[str, dict]:
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row:
                raise KeyError(f"No id in PAN row line {line_idx}. Keys: {sorted(row.keys())}")
            rows[str(row["id"])] = row
    return rows


def flatten_text(value: Any) -> List[str]:
    tweets = []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        for item in value:
            tweets.extend(flatten_text(item))
    elif isinstance(value, dict):
        for key in ["text", "tweet", "content"]:
            if key in value:
                tweets.extend(flatten_text(value[key]))
    return tweets


def tokenize(tweets: List[str]) -> List[str]:
    tokens = []
    for tweet in tweets:
        for tok in TOKEN_RE.findall(tweet.lower()):
            if tok.startswith("@"):
                continue
            if tok.startswith("#"):
                tok = tok[1:]
            tokens.append(tok)
    return tokens


def count_keyword_hits(tokens: List[str], keywords: List[str]) -> Tuple[int, Dict[str, int]]:
    token_counter = Counter(tokens)
    joined = " " + " ".join(tokens) + " "
    total = 0
    hits = {}

    for keyword in keywords:
        key = keyword.lower().strip()
        if " " in key:
            count = joined.count(" " + key + " ")
        else:
            count = token_counter[key]
        if count:
            hits[keyword] = count
            total += count

    return total, hits


def signal_config(task: str) -> Dict[str, List[str]]:
    if task == "gender":
        return GENDER_SIGNALS
    if task == "occupation":
        return OCCUPATION_SIGNALS
    raise ValueError(f"Unsupported task: {task}")


def group_name(gold: str, pred: str) -> str:
    return f"correct_{gold}" if gold == pred else f"{gold}_to_{pred}"


def build_analysis(task: str, prediction_path: str, pan_path: str, max_examples: int) -> dict:
    predictions = load_predictions(prediction_path)
    pan_rows = load_pan_ndjson(pan_path)
    signals = signal_config(task)

    profiles = []
    profiles_with_text = 0
    matched_feed_rows = 0

    for row in predictions:
        cid = celebrity_id(row)
        gold = true_label(row)
        pred = prediction_label(row)
        group = group_name(gold, pred)

        pan_row = pan_rows.get(cid)
        tweets = []
        if pan_row is not None:
            matched_feed_rows += 1
            tweets = flatten_text(pan_row.get("text", []))

        tokens = tokenize(tweets)
        if tokens:
            profiles_with_text += 1

        token_count = len(tokens)
        signal_values = {}
        keyword_hits_by_signal = {}
        all_hits = Counter()

        for signal_name, keywords in signals.items():
            hit_count, hits = count_keyword_hits(tokens, keywords)
            keyword_hits_by_signal[signal_name] = hits
            signal_values[signal_name] = (hit_count / token_count * 1000.0) if token_count else 0.0
            all_hits.update(hits)

        profiles.append({
            "celebrity_id": cid,
            "task": task,
            "true_label": gold,
            "pred_label": pred,
            "group": group,
            "tweet_count": len(tweets),
            "token_count": token_count,
            "signal_values": signal_values,
            "signal_sum": sum(signal_values.values()),
            "keyword_hits_by_signal": keyword_hits_by_signal,
            "top_keyword_hits": all_hits.most_common(12),
        })

    groups = defaultdict(list)
    for profile in profiles:
        groups[profile["group"]].append(profile)

    signal_names = list(signals.keys())
    group_stats = {}
    for group, items in sorted(groups.items()):
        stats = {"n": len(items)}
        for signal_name in signal_names:
            stats[signal_name] = sum(p["signal_values"][signal_name] for p in items) / len(items)
        stats["signal_sum"] = sum(p["signal_sum"] for p in items) / len(items)
        group_stats[group] = stats

    relevant = DISPLAY_GROUPS[task]
    top_examples = {}
    for group in relevant:
        items = sorted(groups.get(group, []), key=lambda x: x["signal_sum"], reverse=True)
        top_examples[group] = items[:max_examples]

    false_high_signal = sorted(
        [p for p in profiles if p["true_label"] != p["pred_label"]],
        key=lambda x: x["signal_sum"],
        reverse=True,
    )[:max_examples * 2]

    return {
        "task": task,
        "prediction_path": prediction_path,
        "pan_path": pan_path,
        "n_predictions": len(predictions),
        "n_pan_rows": len(pan_rows),
        "matched_feed_rows": matched_feed_rows,
        "profiles_with_follower_text": profiles_with_text,
        "signal_names": signal_names,
        "group_stats": group_stats,
        "top_examples": top_examples,
        "false_high_signal": false_high_signal,
        "profiles": profiles,
    }


def format_hits(hits: List[Tuple[str, int]]) -> str:
    return ", ".join(f"{k}:{v}" for k, v in hits[:8])


def write_markdown_report(analysis: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    task = analysis["task"]
    signal_names = analysis["signal_names"]

    with open(output_path, "w", encoding="utf-8") as f:
        title = "Gender" if task == "gender" else "Occupation"
        f.write(f"# {title} Semantic Signal Report\n\n")
        f.write(f"Predictions: **{analysis['n_predictions']}**\n")
        f.write(f"PAN feed rows: **{analysis['n_pan_rows']}**\n")
        f.write(f"Matched feed rows: **{analysis['matched_feed_rows']}**\n")
        f.write(f"Profiles with follower text: **{analysis['profiles_with_follower_text']}**\n\n")

        f.write("## Gruppengrößen\n")
        f.write("| group | n |\n| --- | ---: |\n")
        for group, stats in analysis["group_stats"].items():
            f.write(f"| {group} | {stats['n']} |\n")

        f.write("\n## Durchschnittliche Signalwerte pro 1000 Tokens\n")
        f.write("| group | n | " + " | ".join(signal_names) + " | signal_sum |\n")
        f.write("| --- | ---: | " + " | ".join(["---:"] * (len(signal_names) + 1)) + " |\n")
        for group, stats in analysis["group_stats"].items():
            values = [f"{stats[name]:.4f}" for name in signal_names]
            f.write(f"| {group} | {stats['n']} | " + " | ".join(values) + f" | {stats['signal_sum']:.4f} |\n")

        f.write("\n## Top-Beispiele pro relevanter Fehlergruppe\n")
        for group, items in analysis["top_examples"].items():
            f.write(f"\n### {group}\n")
            f.write("| id | true | pred | tokens | signal_sum | top keyword hits |\n")
            f.write("| --- | --- | --- | ---: | ---: | --- |\n")
            for p in items:
                f.write(
                    f"| {p['celebrity_id']} | {p['true_label']} | {p['pred_label']} | "
                    f"{p['token_count']} | {p['signal_sum']:.4f} | {format_hits(p['top_keyword_hits'])} |\n"
                )

        f.write("\n## IDs mit hoher Signalstärke, aber falscher Vorhersage\n")
        f.write("| id | group | true | pred | tokens | signal_sum | top keyword hits |\n")
        f.write("| --- | --- | --- | --- | ---: | ---: | --- |\n")
        for p in analysis["false_high_signal"]:
            f.write(
                f"| {p['celebrity_id']} | {p['group']} | {p['true_label']} | {p['pred_label']} | "
                f"{p['token_count']} | {p['signal_sum']:.4f} | {format_hits(p['top_keyword_hits'])} |\n"
            )

        f.write("\n## Kurze Interpretation\n")
        if task == "gender":
            f.write(
                "Prüfe besonders, ob `female_to_male` höhere `female_role_signal`, "
                "`family_signal`, `relationship_signal` oder `violence_against_women_signal` "
                "Werte als `correct_male` zeigt. Falls ja, deuten die Fehler auf semantische "
                "Signale hin, die durch die Fusion nicht ausreichend genutzt wurden.\n"
            )
        else:
            f.write(
                "Prüfe besonders, ob `creator_to_performer` gleichzeitig hohe `creator_signal` "
                "und `performer_signal` Werte zeigt. Das würde erklären, warum Creator mit "
                "fan-/performer-naher Follower-Sprache in Richtung Performer kippen.\n"
            )


def strip_profiles_for_json(analysis: dict) -> dict:
    # Full profile list is useful but can be large. Keep it in JSON, but all values are simple.
    return analysis


def resolve_prediction_path(task: str, args) -> str:
    if task == "gender" and args.gender_predictions:
        return args.gender_predictions
    if task == "occupation" and args.occupation_predictions:
        return args.occupation_predictions
    return os.path.join(hybrid_v4_fusion_predictions_dir, DEFAULT_PREDICTION_FILES[task])


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic error signal analysis for HybridV4.")
    parser.add_argument("--target", choices=["gender", "occupation", "all"], default="all")
    parser.add_argument("--pan", default=test_feeds_path)
    parser.add_argument("--gender-predictions", default=None)
    parser.add_argument("--occupation-predictions", default=None)
    parser.add_argument("--output-dir", default=os.path.join(hybrid_v4_output_dir, "error_analysis", "semantic_signals"))
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    print(f"[INFO] analyze_semantic_error_signals version: {SCRIPT_VERSION}")
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] PAN feeds: {args.pan}")
    print(f"[INFO] Output dir: {args.output_dir}")

    tasks = ["gender", "occupation"] if args.target == "all" else [args.target]
    os.makedirs(args.output_dir, exist_ok=True)

    for task in tasks:
        print(f"\n========== {task} ==========")
        prediction_path = resolve_prediction_path(task, args)
        analysis = build_analysis(
            task=task,
            prediction_path=prediction_path,
            pan_path=args.pan,
            max_examples=args.max_examples,
        )

        md_path = os.path.join(args.output_dir, f"{task}_semantic_signal_report.md")
        json_path = os.path.join(args.output_dir, f"{task}_semantic_signal_report.json")
        write_markdown_report(analysis, md_path)
        write_json(strip_profiles_for_json(analysis), json_path)
        print(f"[OK] Saved Markdown: {md_path}")
        print(f"[OK] Saved JSON:     {json_path}")
        print(f"[INFO] Profiles with follower text: {analysis['profiles_with_follower_text']} / {analysis['n_predictions']}")


if __name__ == "__main__":
    main()
