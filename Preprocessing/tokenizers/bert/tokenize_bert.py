import argparse
import json
import os
import sys
from transformers import AutoTokenizer

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
from Preprocessing.normalize import TWEET_SEP, FOLLOWER_SEP, URL_TOKEN, MENTION_TOKEN
from Preprocessing.pan_parser import extract_label_record, aggregate_feeds_by_id
from Preprocessing.build_hf_dataset import build_examples, build_dataset_examples
from Preprocessing.tokenizers.bert.config_bert import (
    MODEL_NAME,
    MAX_LENGTH,
    MAX_FOLLOWERS,
    MAX_TWEETS_PER_FOLLOWER,
    MAX_CHARS,
    OUTPUT_DIRNAME,
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
    parser = argparse.ArgumentParser(description="Tokenize dataset with BERT")
    parser.add_argument("--split", choices=["train", "supp", "test"], default="train")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    added = tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                URL_TOKEN,
                MENTION_TOKEN,
                TWEET_SEP,
                FOLLOWER_SEP,
            ]
        }
    )
    print(f"Added {added} special tokens")

    tokenized_examples = []
    trunc_count = 0

    for ex in examples:
        raw_ids = tokenizer(ex["text"], truncation=False, padding=False)["input_ids"]
        was_truncated = len(raw_ids) > args.max_length
        if was_truncated:
            trunc_count += 1

        tokens = tokenizer(
            ex["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

        tokenized_examples.append(
            {
                "celebrity_id": ex.get("celebrity_id", ex.get("id")),
                "chunk_id": ex.get("chunk_id"),
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "birthyear": ex["birthyear"],
                "gender": ex["gender"],
                "occupation": ex["occupation"],
                "raw_length": len(raw_ids),
                "truncated": was_truncated,
            }
        )

    output_dir = os.path.join(PROJECT_ROOT, "data", OUTPUT_DIRNAME)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{args.split}_tokenized.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tokenized_examples, f, ensure_ascii=False)

    print(f"[{args.split}] Saved to: {output_file}")
    print(f"[{args.split}] Truncated: {trunc_count}/{len(tokenized_examples)}")


if __name__ == "__main__":
    main()