import argparse
import os
import sys
from typing import Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

from Preprocessing.io_utils import load_ndjson, save_jsonl
from Preprocessing.normalize import TWEET_SEP, FOLLOWER_SEP
from Preprocessing.pan_parser import extract_label_record, aggregate_feeds_by_id
from Preprocessing.aggregation import build_examples
from Preprocessing.chunking import build_chunked_examples


def resolve_paths(split: str) -> Tuple[str, str]:
    split = split.lower()
    if split == "train":
        return train_label_path, train_feeds_path
    if split == "supp":
        return supp_label_path, supp_feeds_path
    if split == "test":
        return test_label_path, test_feeds_path
    raise ValueError(f"Unbekannter split: {split}")

def build_dataset_examples(
    labels,
    feed_map,
    tweet_sep,
    follower_sep,
    use_chunking=False,
    tweets_per_chunk=12,
    max_followers=None,
    max_tweets_per_follower=None,
    max_chars=None,
):
    if use_chunking:
        return build_chunked_examples(
            labels=labels,
            feed_map=feed_map,
            tweet_sep=tweet_sep,
            tweets_per_chunk=tweets_per_chunk,
            max_followers=max_followers,
            max_tweets_per_follower=max_tweets_per_follower,
            max_chars=max_chars,
        )

    return build_examples(
        labels=labels,
        feed_map=feed_map,
        tweet_sep=tweet_sep,
        follower_sep=follower_sep,
        max_followers=max_followers,
        max_tweets_per_follower=max_tweets_per_follower,
        max_chars=max_chars,
    )


def main():
    parser = argparse.ArgumentParser(description="Build HF-ready dataset v2")
    parser.add_argument("--split", choices=["train", "supp", "test"], default="train")
    parser.add_argument("--max-followers", type=int, default=3)
    parser.add_argument("--max-tweets-per-follower", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=10000)
    parser.add_argument("--save-jsonl", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--chunk", action="store_true", help="Build chunked examples instead of one text per celebrity")
    parser.add_argument("--tweets-per-chunk", type=int, default=12)
    args = parser.parse_args()

    label_path, feed_path = resolve_paths(args.split)

    label_rows = load_ndjson(label_path)
    feed_rows = load_ndjson(feed_path)

    if args.debug:
        print("\n[DEBUG] Erste Label-Zeile:")
        print(label_rows[0] if label_rows else "Keine Labels")

        print("\n[DEBUG] Erste Feed-Zeile:")
        print(feed_rows[0] if feed_rows else "Keine Feeds")

    labels = [extract_label_record(r) for r in label_rows]
    feed_map = aggregate_feeds_by_id(feed_rows)

    print(f"\n[{args.split}] Labels geladen: {len(label_rows)}")
    print(f"[{args.split}] Feed-Zeilen geladen: {len(feed_rows)}")
    print(f"[{args.split}] Feed-IDs erkannt: {len(feed_map)}")

    examples = build_dataset_examples(
        labels=labels,
        feed_map=feed_map,
        tweet_sep=TWEET_SEP,
        follower_sep=FOLLOWER_SEP,
        use_chunking=args.chunk,
        tweets_per_chunk=args.tweets_per_chunk,
        max_followers=args.max_followers,
        max_tweets_per_follower=args.max_tweets_per_follower,
        max_chars=args.max_chars,
    )

    print(f"[{args.split}] Beispiele gebaut: {len(examples)}")

    # Debug-Ausgabe des ersten Beispiels
    if examples:
        print("\n[DEBUG] Erstes Beispiel:")
        preview = {
            "text_preview": examples[0]["text"][:500],
            "num_chars": examples[0]["num_chars"],
        }

        for key in ["id", "celebrity_id", "chunk_id", "birthyear", "gender", "occupation", "num_followers_used",
                    "num_tweets_in_chunk"]:
            if key in examples[0]:
                preview[key] = examples[0][key]

        print(preview)

    # Optional: Speichern als JSONL
    if args.save_jsonl:
        save_jsonl(args.save_jsonl, examples)
        print(f"\nGespeichert: {args.save_jsonl}")


if __name__ == "__main__":
    main()