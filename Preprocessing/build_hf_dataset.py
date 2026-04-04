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

from io_utils import load_ndjson, save_jsonl
from normalize import TWEET_SEP, FOLLOWER_SEP
from pan_parser import extract_label_record, aggregate_feeds_by_id
from aggregation import build_examples


def resolve_paths(split: str) -> Tuple[str, str]:
    split = split.lower()
    if split == "train":
        return train_label_path, train_feeds_path
    if split == "supp":
        return supp_label_path, supp_feeds_path
    if split == "test":
        return test_label_path, test_feeds_path
    raise ValueError(f"Unbekannter split: {split}")


def main():
    parser = argparse.ArgumentParser(description="Build HF-ready dataset v2")
    parser.add_argument("--split", choices=["train", "supp", "test"], default="train")
    parser.add_argument("--max-followers", type=int, default=3)
    parser.add_argument("--max-tweets-per-follower", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=10000)
    parser.add_argument("--save-jsonl", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
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

    examples = build_examples(
        labels=labels,
        feed_map=feed_map,
        tweet_sep=TWEET_SEP,
        follower_sep=FOLLOWER_SEP,
        max_followers=args.max_followers,
        max_tweets_per_follower=args.max_tweets_per_follower,
        max_chars=args.max_chars,
    )

    print(f"[{args.split}] Beispiele gebaut: {len(examples)}")

    if examples:
        print("\n[DEBUG] Erstes Beispiel:")
        print({
            "id": examples[0]["id"],
            "birthyear": examples[0]["birthyear"],
            "gender": examples[0]["gender"],
            "occupation": examples[0]["occupation"],
            "num_followers_used": examples[0]["num_followers_used"],
            "num_chars": examples[0]["num_chars"],
            "text_preview": examples[0]["text"][:500],
        })

    if args.save_jsonl:
        save_jsonl(args.save_jsonl, examples)
        print(f"\nGespeichert: {args.save_jsonl}")


if __name__ == "__main__":
    main()