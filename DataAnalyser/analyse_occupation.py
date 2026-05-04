import json
from pathlib import Path
from typing import Any, List, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Wichtig: Das Predictionfile muss zum gleichen Split gehören wie TEST_FEEDS_PATH.
PRED_PATH = PROJECT_ROOT / "outputs" / "bertweet_v3" / "predictions" / "occupation_test_predictions.json"

TEST_FEEDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "pan20-celebrity-profiling-test-dataset-2020-02-28"
    / "pan20-celebrity-profiling-test-dataset-2020-02-28"
    / "follower-feeds.ndjson"
)

OUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "bertweet_v3"
    / "analysis"
    / "creator_performer_error_tweets_test.json"
)


def flatten_text(value: Any) -> List[str]:
    texts = []

    if isinstance(value, str):
        texts.append(value)

    elif isinstance(value, list):
        for item in value:
            texts.extend(flatten_text(item))

    elif isinstance(value, dict):
        if "text" in value:
            texts.extend(flatten_text(value["text"]))
        else:
            for v in value.values():
                texts.extend(flatten_text(v))

    return texts


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_ndjson(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def select_creator_to_performer_cases(preds: List[dict], top_k: int = 5):
    cases = []

    for row in preds:
        if row["true_label"] == "creator" and row["pred_label"] == "performer":
            probs = row["probabilities"]

            cases.append({
                "celebrity_id": str(row["celebrity_id"]),
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "p_sports": probs[0],
                "p_performer": probs[1],
                "p_creator": probs[2],
                "p_politics": probs[3],
                "performer_minus_creator": probs[1] - probs[2],
            })

    uncertain = sorted(cases, key=lambda x: x["performer_minus_creator"])[:top_k]
    confident_wrong = sorted(cases, key=lambda x: x["performer_minus_creator"], reverse=True)[:top_k]

    return uncertain, confident_wrong, cases


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    preds = load_json(PRED_PATH)

    uncertain, confident_wrong, all_cases = select_creator_to_performer_cases(
        preds,
        top_k=5,
    )

    selected_ids = {row["celebrity_id"] for row in uncertain + confident_wrong}

    print(f"[INFO] Total creator->performer cases: {len(all_cases)}")
    print(f"[INFO] Selected IDs: {sorted(selected_ids)}")

    feeds_by_id: Dict[str, dict] = {}

    for row in iter_ndjson(TEST_FEEDS_PATH):
        cid = str(row["id"])

        if cid not in selected_ids:
            continue

        texts = flatten_text(row["text"])

        feeds_by_id[cid] = {
            "num_texts_found": len(texts),
            "sample_texts": texts[:100],
        }

    result = {
        "prediction_file": str(PRED_PATH),
        "feeds_file": str(TEST_FEEDS_PATH),
        "total_creator_to_performer_cases": len(all_cases),
        "uncertain_creator_to_performer": uncertain,
        "confident_wrong_creator_to_performer": confident_wrong,
        "tweets_by_celebrity_id": feeds_by_id,
        "missing_ids": sorted(selected_ids - set(feeds_by_id.keys())),
    }

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {OUT_PATH}")

    print("[INFO] Found IDs:")
    for cid, data in feeds_by_id.items():
        print(f"  {cid}: {data['num_texts_found']} tweets")

    missing = sorted(selected_ids - set(feeds_by_id.keys()))
    if missing:
        print(f"[WARN] Missing IDs: {missing}")
        print("[WARN] This usually means prediction file and feed file are from different splits/runs.")


if __name__ == "__main__":
    main()