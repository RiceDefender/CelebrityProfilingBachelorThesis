import argparse
import json
import os
import sys

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from _constants import (
    train_label_path,
    train_feeds_path,
    supp_label_path,
    supp_feeds_path,
    test_label_path,
    test_feeds_path,
)

from Preprocessing.io_utils import load_ndjson
from Preprocessing.normalize import TWEET_SEP, FOLLOWER_SEP
from Preprocessing.pan_parser import extract_label_record, aggregate_feeds_by_id
from Preprocessing.build_hf_dataset import build_dataset_examples
from Preprocessing.vectorizers.sbert.config_sbert import (
    MODEL_NAME,
    MAX_FOLLOWERS,
    MAX_TWEETS_PER_FOLLOWER,
    MAX_CHARS,
    OUTPUT_DIRNAME,
    NORMALIZE_EMBEDDINGS,
    USE_CHUNKING,
    TWEETS_PER_CHUNK,
)

def resolve_paths(split: str):
    split = split.lower()
    if split == "train":
        return train_label_path, train_feeds_path
    if split == "supp":
        return supp_label_path, supp_feeds_path
    if split == "test":
        return test_label_path, test_feeds_path
    raise ValueError(f"Unknown split: {split}")


def main():
    parser = argparse.ArgumentParser(description="Create SBERT embeddings")
    parser.add_argument("--split", choices=["train", "supp", "test"], default="train")
    args = parser.parse_args()

    label_path, feed_path = resolve_paths(args.split)

    label_rows = load_ndjson(label_path)
    feed_rows = load_ndjson(feed_path)

    labels = [extract_label_record(r) for r in label_rows]
    feed_map = aggregate_feeds_by_id(feed_rows)

    examples = build_dataset_examples(
        labels=labels,
        feed_map=feed_map,
        tweet_sep=TWEET_SEP,
        follower_sep=FOLLOWER_SEP,
        use_chunking=USE_CHUNKING,
        tweets_per_chunk=TWEETS_PER_CHUNK,
        max_followers=MAX_FOLLOWERS,
        max_tweets_per_follower=MAX_TWEETS_PER_FOLLOWER,
        max_chars=MAX_CHARS,
    )

    print(f"[{args.split}] Examples built: {len(examples)}")

    model = SentenceTransformer(MODEL_NAME)

    texts = [ex["text"] for ex in examples]
    embeddings = model.encode(
        texts,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        show_progress_bar=True,
    )

    output = []
    for ex, emb in zip(examples, embeddings):
        output.append(
            {
                "celebrity_id": ex.get("celebrity_id", ex.get("id")),
                "chunk_id": ex.get("chunk_id"),
                "embedding": emb.tolist(),
                "birthyear": ex["birthyear"],
                "gender": ex["gender"],
                "occupation": ex["occupation"],
            }
        )

    output_dir = os.path.join(PROJECT_ROOT, "data", OUTPUT_DIRNAME)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{args.split}_vectors.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"[{args.split}] Saved to: {output_file}")


if __name__ == "__main__":
    main()