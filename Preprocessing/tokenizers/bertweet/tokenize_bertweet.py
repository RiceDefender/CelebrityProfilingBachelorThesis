import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

from transformers import AutoTokenizer


# ---------------------------------------------------------
# Project root / imports
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from _constants import (
    train_label_path,
    train_feeds_path,
    test_label_path,
    test_feeds_path,
    supp_label_path,
    supp_feeds_path,
    bertweet_processed_dir,
    bertweet_train_tokenized_path,
    bertweet_test_tokenized_path,
    bertweet_supp_tokenized_path,
    bertweet_train_meta_path,
    bertweet_test_meta_path,
    bertweet_supp_meta_path,
)

from Preprocessing.tokenizers.bertweet.config_bertweet import (
    MODEL_NAME,
    MAX_LENGTH,
    STRIDE,
    MAX_CHUNKS_PER_CELEBRITY,
    MIN_TOKENS_PER_CHUNK,
    URL_TOKEN,
    MENTION_TOKEN,
    NORMALIZE_URLS,
    NORMALIZE_MENTIONS,
    MAX_TWEETS_PER_CELEBRITY,
    TWEET_SELECTION_STRATEGY,
)

# ---------------------------------------------------------
# IO helpers
# ---------------------------------------------------------
def ensure_dirs():
    os.makedirs(bertweet_processed_dir, exist_ok=True)


def read_ndjson(path: str) -> List[dict]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def write_ndjson_row(f, obj: dict):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def iter_ndjson(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_celebrity_id(row: dict):
    return (
        row.get("id")
        or row.get("celebrity_id")
        or row.get("author_id")
        or row.get("user_id")
    )

def select_tweets(tweets: List[str], max_tweets: int, strategy: str = "evenly_spaced") -> List[str]:
    if max_tweets is None:
        return tweets

    if len(tweets) <= max_tweets:
        return tweets

    if strategy == "first":
        return tweets[:max_tweets]

    if strategy == "evenly_spaced":
        indices = [
            round(i * (len(tweets) - 1) / (max_tweets - 1))
            for i in range(max_tweets)
        ]
        return [tweets[i] for i in indices]

    raise ValueError(f"Unknown tweet selection strategy: {strategy}")

# ---------------------------------------------------------
# Loading labels / feeds
# ---------------------------------------------------------
def load_labels(label_path: str) -> Dict[str, dict]:
    labels = {}

    for row in read_ndjson(label_path):
        celebrity_id = str(row["id"])

        labels[celebrity_id] = {
            "celebrity_id": celebrity_id,
            "occupation": row["occupation"],
            "gender": row["gender"],
            "birthyear": row["birthyear"],
        }

    return labels


def extract_texts_recursive(obj) -> List[str]:
    """
    Robustly extracts tweet texts from nested PAN follower-feed structures.

    Important:
    - Does NOT convert large lists/dicts to string.
    - Only returns actual string tweet texts.
    - Handles:
      - strings
      - lists of strings
      - lists of dicts
      - nested followers/tweets/text fields
    """

    texts = []

    if obj is None:
        return texts

    if isinstance(obj, str):
        cleaned = obj.strip()
        if cleaned:
            texts.append(cleaned)
        return texts

    if isinstance(obj, list):
        for item in obj:
            texts.extend(extract_texts_recursive(item))
        return texts

    if isinstance(obj, dict):
        # Most common tweet text keys
        for key in ["text", "full_text", "tweet", "content"]:
            value = obj.get(key)

            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    texts.append(cleaned)

            elif isinstance(value, (list, dict)):
                texts.extend(extract_texts_recursive(value))

        # Common nested keys in follower-feed data
        for key in ["tweets", "followers", "feed", "feeds", "items", "data"]:
            value = obj.get(key)
            if isinstance(value, (list, dict)):
                texts.extend(extract_texts_recursive(value))

        return texts

    # Do not cast unknown large objects to str.
    return texts


def extract_tweets_from_feed_row(row: dict) -> List[str]:
    """
    Extracts all tweet texts for one celebrity row.
    """
    tweets = []

    # Try common top-level containers first
    for key in ["tweets", "followers", "text", "feed", "feeds", "items", "data"]:
        if key in row:
            tweets.extend(extract_texts_recursive(row[key]))

    # Remove duplicates while preserving order
    seen = set()
    unique_tweets = []

    for tweet in tweets:
        if tweet not in seen:
            seen.add(tweet)
            unique_tweets.append(tweet)

    return unique_tweets


# ---------------------------------------------------------
# BERTweet normalization
# ---------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_bertweet(text: str) -> str:
    text = str(text)

    if NORMALIZE_URLS:
        text = _URL_RE.sub(URL_TOKEN, text)

    if NORMALIZE_MENTIONS:
        text = _MENTION_RE.sub(MENTION_TOKEN, text)

    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


# ---------------------------------------------------------
# Token-based chunking
# ---------------------------------------------------------
def build_token_chunks(
    tokenizer,
    tweets: List[str],
    max_length: int,
    stride: int,
    max_chunks: int,
    min_tokens_per_chunk: int,
) -> List[dict]:
    """
    Creates chunks before final encoding.

    Important:
    - We tokenize without special tokens first.
    - max_content_length = max_length - number of special tokens.
    - Then we convert each token chunk back to text.
    - Final tokenizer call pads to max_length.
    """

    normalized_tweets = [
        normalize_for_bertweet(t)
        for t in tweets
        if isinstance(t, str) and t.strip()
    ]

    joined_text = " ".join(normalized_tweets).strip()

    if not joined_text:
        return []

    token_ids = tokenizer.encode(
        joined_text,
        add_special_tokens=False,
        truncation=False,
    )

    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    max_content_length = max_length - num_special_tokens

    if max_content_length <= 0:
        raise ValueError(
            f"MAX_LENGTH={max_length} is too small for tokenizer special tokens."
        )

    step = max_content_length - stride

    if step <= 0:
        raise ValueError(
            f"Invalid STRIDE={stride}. It must be smaller than max_content_length={max_content_length}."
        )

    chunks = []



    for start in range(0, len(token_ids), step):
        end = start + max_content_length
        chunk_token_ids = token_ids[start:end]

        if len(chunk_token_ids) < min_tokens_per_chunk:
            continue

        chunk_text = tokenizer.decode(
            chunk_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        encoded = tokenizer(
            chunk_text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

        # Safety check: output must always be exactly MAX_LENGTH.
        assert len(encoded["input_ids"]) == max_length
        assert len(encoded["attention_mask"]) == max_length

        chunks.append({
            "chunk_text": chunk_text,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "num_content_tokens": len(chunk_token_ids),
        })

        if max_chunks is not None and len(chunks) >= max_chunks:
            break

    return chunks


# ---------------------------------------------------------
# Dataset tokenization
# ---------------------------------------------------------
def tokenize_split(
    split_name: str,
    label_path: str,
    feeds_path: str,
    output_path: str,
    meta_path: str,
    limit_celebrities: Optional[int] = None,
):
    print(f"\n========== Tokenizing BERTweet split: {split_name} ==========")
    print(f"[INFO] Labels: {label_path}")
    print(f"[INFO] Feeds:  {feeds_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Meta:   {meta_path}")

    labels = load_labels(label_path)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        normalization=True,
    )

    missing_labels = 0
    empty_chunks = 0
    processed_celebrities = 0
    written_celebrities = 0
    written_chunks = 0
    meta_rows = []

    with open(output_path, "w", encoding="utf-8") as out_f:
        for row in iter_ndjson(feeds_path):
            if limit_celebrities is not None and processed_celebrities >= limit_celebrities:
                break

            celebrity_id = get_celebrity_id(row)

            if celebrity_id is None:
                continue

            celebrity_id = str(celebrity_id)

            tweets = extract_tweets_from_feed_row(row)
            num_raw_tweets = len(tweets)

            tweets = select_tweets(
                tweets=tweets,
                max_tweets=MAX_TWEETS_PER_CELEBRITY,
                strategy=TWEET_SELECTION_STRATEGY,
            )

            if processed_celebrities < 3:
                print("\n[DEBUG] row keys:", list(row.keys()))
                print("[DEBUG] celebrity_id:", celebrity_id)
                print("[DEBUG] num raw tweets:", num_raw_tweets)
                print("[DEBUG] num used tweets:", len(tweets))

                if tweets:
                    lengths = [len(t) for t in tweets]
                    print("[DEBUG] min used tweet chars:", min(lengths))
                    print("[DEBUG] max used tweet chars:", max(lengths))
                    print("[DEBUG] avg used tweet chars:", sum(lengths) / len(lengths))
                    print("[DEBUG] first used tweet sample:", tweets[0][:300])

            processed_celebrities += 1

            if celebrity_id not in labels:
                missing_labels += 1
                continue

            chunks = build_token_chunks(
                tokenizer=tokenizer,
                tweets=tweets,
                max_length=MAX_LENGTH,
                stride=STRIDE,
                max_chunks=MAX_CHUNKS_PER_CELEBRITY,
                min_tokens_per_chunk=MIN_TOKENS_PER_CHUNK,
            )

            if not chunks:
                empty_chunks += 1
                continue

            label_info = labels[celebrity_id]

            for chunk_idx, chunk in enumerate(chunks):
                row = {
                    "celebrity_id": celebrity_id,
                    "chunk_id": chunk_idx,
                    "input_ids": chunk["input_ids"],
                    "attention_mask": chunk["attention_mask"],
                    "max_length": MAX_LENGTH,
                    "model_name": MODEL_NAME,
                    "num_content_tokens": chunk["num_content_tokens"],
                    "occupation": label_info["occupation"],
                    "gender": label_info["gender"],
                    "birthyear": label_info["birthyear"],
                }

                write_ndjson_row(out_f, row)
                written_chunks += 1

            meta_rows.append({
                "celebrity_id": celebrity_id,
                "num_raw_tweets": num_raw_tweets,
                "num_used_tweets": len(tweets),
                "num_chunks": len(chunks),
                "max_length": MAX_LENGTH,
                "stride": STRIDE,
                "model_name": MODEL_NAME,
                "occupation": label_info["occupation"],
                "gender": label_info["gender"],
                "birthyear": label_info["birthyear"],
                "max_chunks_per_celebrity": MAX_CHUNKS_PER_CELEBRITY,
            })

            written_celebrities += 1

            if written_celebrities % 50 == 0:
                print(
                    f"[PROGRESS] {split_name}: "
                    f"celebrities={written_celebrities} "
                    f"chunks={written_chunks}"
                )

    save_json(meta_rows, meta_path)

    print(f"[OK] Saved tokenized NDJSON: {output_path}")
    print(f"[OK] Saved meta JSON:        {meta_path}")
    print(f"[INFO] Processed celebrities: {processed_celebrities}")
    print(f"[INFO] Written celebrities:   {written_celebrities}")
    print(f"[INFO] Tokenized chunks:      {written_chunks}")
    print(f"[INFO] Missing labels:        {missing_labels}")
    print(f"[INFO] Empty chunks:          {empty_chunks}")

    if meta_rows:
        avg_chunks = sum(r["num_chunks"] for r in meta_rows) / len(meta_rows)
        print(f"[INFO] Avg chunks/celeb:      {avg_chunks:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize PAN Celebrity Profiling data for BERTweet."
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "supp", "all"],
        default="all",
    )

    parser.add_argument(
        "--limit-celebrities",
        type=int,
        default=None,
        help="Optional limit for quick tokenizer debugging.",
    )

    args = parser.parse_args()

    ensure_dirs()

    if args.split in ["train", "all"]:
        tokenize_split(
            split_name="train",
            label_path=train_label_path,
            feeds_path=train_feeds_path,
            output_path=bertweet_train_tokenized_path,
            meta_path=bertweet_train_meta_path,
            limit_celebrities=args.limit_celebrities,
        )

    if args.split in ["test", "all"]:
        tokenize_split(
            split_name="test",
            label_path=test_label_path,
            feeds_path=test_feeds_path,
            output_path=bertweet_test_tokenized_path,
            meta_path=bertweet_test_meta_path,
            limit_celebrities=args.limit_celebrities,
        )

    if args.split in ["supp", "all"]:
        tokenize_split(
            split_name="supp",
            label_path=supp_label_path,
            feeds_path=supp_feeds_path,
            output_path=bertweet_supp_tokenized_path,
            meta_path=bertweet_supp_meta_path,
            limit_celebrities=args.limit_celebrities,
        )


if __name__ == "__main__":
    main()