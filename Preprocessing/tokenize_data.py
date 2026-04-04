import argparse
import os
import sys
import json
from transformers import BertTokenizer

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

from io_utils import load_ndjson
from normalize import TWEET_SEP, FOLLOWER_SEP
from pan_parser import extract_label_record, aggregate_feeds_by_id
from aggregation import build_examples

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
    # Parse command-line arguments the limit of the token is 512
    parser = argparse.ArgumentParser(description="Tokenize dataset with google-bert/bert-base-uncased")
    parser.add_argument("--split", choices=["train", "supp", "test"], default="test", help="Dataset split to tokenize")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length for the tokenizer")
    args = parser.parse_args()

    label_path, feed_path = resolve_paths(args.split)
    
    print(f"[{args.split}] Loading labels from: {label_path}")
    label_rows = load_ndjson(label_path)
    
    print(f"[{args.split}] Loading feeds from: {feed_path}")
    feed_rows = load_ndjson(feed_path)
    
    labels = [extract_label_record(r) for r in label_rows]
    feed_map = aggregate_feeds_by_id(feed_rows)
    
    # We rely on the preprocessing modules to build the text
    examples = build_examples(
        labels=labels,
        feed_map=feed_map,
        tweet_sep=TWEET_SEP,
        follower_sep=FOLLOWER_SEP,
        max_followers=3,
        max_tweets_per_follower=20,
        max_chars=10000
    )

    # Load the tokenizer
    print(f"[{args.split}] Loading BertTokenizer (google-bert/bert-base-uncased)...")
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    
    # Add custom tokens from the normalization step to avoid splitting them up
    special_tokens_dict = {'additional_special_tokens': [TWEET_SEP, FOLLOWER_SEP, '[URL]', '[MENTION]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens to the tokenizer.")
    
    print(f"[{args.split}] Tokenizing {len(examples)} examples...")
    
    tokenized_examples = []
    for ex in examples:
        tokens = tokenizer(
            ex["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length
        )
        tokenized_examples.append({
            "id": ex["id"],
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "gender": ex["gender"],
            "occupation": ex["occupation"],
            "birthyear": ex["birthyear"]
        })

    # Save the tokenized data
    output_dir = os.path.join(PROJECT_ROOT, "data", "tokenized")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{args.split}_tokenized.json")
    print(f"Saving tokenized data to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tokenized_examples, f)
        
    print(f"Done! Output is in {output_file}")

if __name__ == "__main__":
    main()