import argparse
import json
import os
import random
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
)

from Preprocessing.tokenizers.bertweet.config_bertweet_v35_controlled_sampling import (
    MODEL_NAME,
    VERSION,
    SAMPLING_NAME,
    MAX_FOLLOWERS_PER_CELEBRITY,
    MAX_TWEETS_PER_FOLLOWER,
    FOLLOWER_SELECTION_STRATEGY,
    TWEET_SELECTION_STRATEGY,
    RANDOM_SEED,
    MAX_LENGTH,
    STRIDE,
    MAX_CHUNKS_PER_CELEBRITY,
    MIN_TOKENS_PER_CHUNK,
    URL_TOKEN,
    MENTION_TOKEN,
    NORMALIZE_URLS,
    NORMALIZE_MENTIONS,
    bertweet_v35_processed_dir,
    bertweet_v35_train_tokenized_path,
    bertweet_v35_test_tokenized_path,
    bertweet_v35_supp_tokenized_path,
    bertweet_v35_train_meta_path,
    bertweet_v35_test_meta_path,
    bertweet_v35_supp_meta_path,
)


# ---------------------------------------------------------
# IO helpers
# ---------------------------------------------------------
def ensure_dirs():
    os.makedirs(bertweet_v35_processed_dir, exist_ok=True)


def read_ndjson(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_ndjson(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid NDJSON in {path} at line {line_idx}: {e}"
                ) from e


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_ndjson_row(f, obj: dict):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_celebrity_id(row: dict):
    return (
        row.get("id")
        or row.get("celebrity_id")
        or row.get("author_id")
        or row.get("user_id")
    )


# ---------------------------------------------------------
# Labels
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


# ---------------------------------------------------------
# Follower-block extraction
# ---------------------------------------------------------
def extract_tweets_recursive(obj) -> List[str]:
    tweets = []

    if obj is None:
        return tweets

    if isinstance(obj, str):
        cleaned = obj.strip()
        if cleaned:
            tweets.append(cleaned)
        return tweets

    if isinstance(obj, list):
        for item in obj:
            tweets.extend(extract_tweets_recursive(item))
        return tweets

    if isinstance(obj, dict):
        for key in ["text", "full_text", "tweet", "content"]:
            value = obj.get(key)
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    tweets.append(cleaned)
            elif isinstance(value, (list, dict)):
                tweets.extend(extract_tweets_recursive(value))

        for key in ["tweets", "feed", "feeds", "items", "data"]:
            value = obj.get(key)
            if isinstance(value, (list, dict)):
                tweets.extend(extract_tweets_recursive(value))

        return tweets

    return tweets


def deduplicate_preserve_order(tweets: List[str]) -> List[str]:
    seen = set()
    unique = []

    for tweet in tweets:
        if tweet not in seen:
            seen.add(tweet)
            unique.append(tweet)

    return unique


def extract_follower_blocks_from_feed_row(row: dict) -> Tuple[List[List[str]], str]:
    """
    Keeps PAN follower structure whenever possible.

    Expected:
        row["text"] = [
            [tweet, tweet, ...],
            [tweet, tweet, ...],
            ...
        ]

    Returns:
        follower_blocks, structure_type
    """
    for key in ["text", "followers", "tweets", "feed", "feeds", "items", "data"]:
        if key not in row:
            continue

        value = row[key]

        if isinstance(value, list) and all(isinstance(x, list) for x in value):
            follower_blocks = []
            for follower_block in value:
                tweets = extract_tweets_recursive(follower_block)
                tweets = deduplicate_preserve_order(tweets)
                if tweets:
                    follower_blocks.append(tweets)
            return follower_blocks, "nested_followers"

        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            tweets = deduplicate_preserve_order(
                [str(t).strip() for t in value if str(t).strip()]
            )
            return [tweets] if tweets else [], "flat_tweets"

        if isinstance(value, (list, dict)):
            tweets = extract_tweets_recursive(value)
            tweets = deduplicate_preserve_order(tweets)
            if tweets:
                return [tweets], "recursive_flattened"

    return [], "unknown"


# ---------------------------------------------------------
# Controlled sampling
# ---------------------------------------------------------
def select_items(items: List, max_items: Optional[int], strategy: str, rng: random.Random) -> List:
    if max_items is None:
        return list(items)

    if len(items) <= max_items:
        return list(items)

    if strategy == "first":
        return list(items[:max_items])

    if strategy == "random":
        return rng.sample(items, max_items)

    if strategy == "evenly_spaced":
        indices = [
            round(i * (len(items) - 1) / (max_items - 1))
            for i in range(max_items)
        ]
        return [items[i] for i in indices]

    raise ValueError(f"Unknown selection strategy: {strategy}")


def sample_follower_blocks(
    follower_blocks: List[List[str]],
    celebrity_id: str,
) -> Tuple[List[List[str]], dict]:
    """
    Controlled sampling:
    - choose up to MAX_FOLLOWERS_PER_CELEBRITY follower blocks
    - choose up to MAX_TWEETS_PER_FOLLOWER tweets per selected follower
    """
    non_empty = [block for block in follower_blocks if len(block) > 0]

    # deterministic per celebrity
    try:
        cid_int = int(celebrity_id)
    except ValueError:
        cid_int = sum(ord(c) for c in celebrity_id)

    rng = random.Random(RANDOM_SEED + cid_int)

    selected_followers = select_items(
        items=non_empty,
        max_items=MAX_FOLLOWERS_PER_CELEBRITY,
        strategy=FOLLOWER_SELECTION_STRATEGY,
        rng=rng,
    )

    sampled_blocks = []

    for follower_idx, block in enumerate(selected_followers):
        follower_rng = random.Random(RANDOM_SEED + cid_int * 1009 + follower_idx)

        selected_tweets = select_items(
            items=block,
            max_items=MAX_TWEETS_PER_FOLLOWER,
            strategy=TWEET_SELECTION_STRATEGY,
            rng=follower_rng,
        )

        if selected_tweets:
            sampled_blocks.append(selected_tweets)

    tweets_per_selected_follower = [len(block) for block in sampled_blocks]

    stats = {
        "sampling_name": SAMPLING_NAME,
        "num_raw_follower_blocks": len(follower_blocks),
        "num_non_empty_raw_followers": len(non_empty),
        "num_selected_followers": len(sampled_blocks),
        "num_raw_tweets": sum(len(block) for block in non_empty),
        "num_used_tweets": sum(tweets_per_selected_follower),
        "max_followers_per_celebrity": MAX_FOLLOWERS_PER_CELEBRITY,
        "max_tweets_per_follower": MAX_TWEETS_PER_FOLLOWER,
        "follower_selection_strategy": FOLLOWER_SELECTION_STRATEGY,
        "tweet_selection_strategy": TWEET_SELECTION_STRATEGY,
        "tweets_per_selected_follower_min": min(tweets_per_selected_follower)
        if tweets_per_selected_follower
        else 0,
        "tweets_per_selected_follower_max": max(tweets_per_selected_follower)
        if tweets_per_selected_follower
        else 0,
        "tweets_per_selected_follower_mean": (
            sum(tweets_per_selected_follower) / len(tweets_per_selected_follower)
            if tweets_per_selected_follower
            else 0.0
        ),
    }

    return sampled_blocks, stats


def flatten_blocks(follower_blocks: List[List[str]]) -> List[str]:
    return [tweet for block in follower_blocks for tweet in block]


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
    max_chunks: Optional[int],
    min_tokens_per_chunk: int,
) -> List[dict]:
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
    print(f"\n========== Tokenizing BERTweet split: {split_name} ({VERSION}) ==========")
    print(f"[INFO] Labels: {label_path}")
    print(f"[INFO] Feeds:  {feeds_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Meta:   {meta_path}")
    print(f"[INFO] Sampling: {SAMPLING_NAME}")
    print(f"[INFO] max_followers={MAX_FOLLOWERS_PER_CELEBRITY}")
    print(f"[INFO] max_tweets_per_follower={MAX_TWEETS_PER_FOLLOWER}")

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
    structure_counter = {}
    meta_rows = []

    with open(output_path, "w", encoding="utf-8") as out_f:
        for row in iter_ndjson(feeds_path):
            if limit_celebrities is not None and processed_celebrities >= limit_celebrities:
                break

            celebrity_id = get_celebrity_id(row)

            if celebrity_id is None:
                continue

            celebrity_id = str(celebrity_id)

            follower_blocks, structure_type = extract_follower_blocks_from_feed_row(row)
            structure_counter[structure_type] = structure_counter.get(structure_type, 0) + 1

            sampled_blocks, sampling_stats = sample_follower_blocks(
                follower_blocks=follower_blocks,
                celebrity_id=celebrity_id,
            )

            tweets = flatten_blocks(sampled_blocks)

            if processed_celebrities < 3:
                print("\n[DEBUG] row keys:", list(row.keys()))
                print("[DEBUG] celebrity_id:", celebrity_id)
                print("[DEBUG] structure_type:", structure_type)
                print("[DEBUG] raw followers:", sampling_stats["num_non_empty_raw_followers"])
                print("[DEBUG] selected followers:", sampling_stats["num_selected_followers"])
                print("[DEBUG] raw tweets:", sampling_stats["num_raw_tweets"])
                print("[DEBUG] used tweets:", sampling_stats["num_used_tweets"])
                if tweets:
                    print("[DEBUG] first sampled tweet:", tweets[0][:300])

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
                out_row = {
                    "celebrity_id": celebrity_id,
                    "chunk_id": chunk_idx,
                    "input_ids": chunk["input_ids"],
                    "attention_mask": chunk["attention_mask"],
                    "max_length": MAX_LENGTH,
                    "model_name": MODEL_NAME,
                    "version": VERSION,
                    "num_content_tokens": chunk["num_content_tokens"],
                    "occupation": label_info["occupation"],
                    "gender": label_info["gender"],
                    "birthyear": label_info["birthyear"],
                    "sampling": {
                        "sampling_name": SAMPLING_NAME,
                        "max_followers_per_celebrity": MAX_FOLLOWERS_PER_CELEBRITY,
                        "max_tweets_per_follower": MAX_TWEETS_PER_FOLLOWER,
                        "follower_selection_strategy": FOLLOWER_SELECTION_STRATEGY,
                        "tweet_selection_strategy": TWEET_SELECTION_STRATEGY,
                        "random_seed": RANDOM_SEED,
                    },
                }

                write_ndjson_row(out_f, out_row)
                written_chunks += 1

            meta_rows.append({
                "celebrity_id": celebrity_id,
                "structure_type": structure_type,
                "num_chunks": len(chunks),
                "max_length": MAX_LENGTH,
                "stride": STRIDE,
                "model_name": MODEL_NAME,
                "version": VERSION,
                "occupation": label_info["occupation"],
                "gender": label_info["gender"],
                "birthyear": label_info["birthyear"],
                "max_chunks_per_celebrity": MAX_CHUNKS_PER_CELEBRITY,
                "sampling_stats": sampling_stats,
            })

            written_celebrities += 1

            if written_celebrities % 50 == 0:
                print(
                    f"[PROGRESS] {split_name}: "
                    f"celebrities={written_celebrities} "
                    f"chunks={written_chunks}"
                )

    save_json({
        "version": VERSION,
        "sampling_name": SAMPLING_NAME,
        "structure_counter": structure_counter,
        "rows": meta_rows,
    }, meta_path)

    print(f"[OK] Saved tokenized NDJSON: {output_path}")
    print(f"[OK] Saved meta JSON:        {meta_path}")
    print(f"[INFO] Processed celebrities: {processed_celebrities}")
    print(f"[INFO] Written celebrities:   {written_celebrities}")
    print(f"[INFO] Tokenized chunks:      {written_chunks}")
    print(f"[INFO] Missing labels:        {missing_labels}")
    print(f"[INFO] Empty chunks:          {empty_chunks}")
    print(f"[INFO] Structure counter:     {structure_counter}")

    if meta_rows:
        avg_chunks = sum(r["num_chunks"] for r in meta_rows) / len(meta_rows)
        avg_used_tweets = (
            sum(r["sampling_stats"]["num_used_tweets"] for r in meta_rows)
            / len(meta_rows)
        )
        avg_selected_followers = (
            sum(r["sampling_stats"]["num_selected_followers"] for r in meta_rows)
            / len(meta_rows)
        )

        print(f"[INFO] Avg chunks/celeb:       {avg_chunks:.2f}")
        print(f"[INFO] Avg used tweets/celeb:  {avg_used_tweets:.2f}")
        print(f"[INFO] Avg followers/celeb:    {avg_selected_followers:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize PAN Celebrity Profiling data for BERTweet V3.5 controlled feed sampling."
    )

    parser.add_argument(
        "--split",
        choices=["train", "test", "supp", "all", "all-with-supp"],
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
            output_path=bertweet_v35_train_tokenized_path,
            meta_path=bertweet_v35_train_meta_path,
            limit_celebrities=args.limit_celebrities,
        )

    if args.split in ["test", "all"]:
        tokenize_split(
            split_name="test",
            label_path=test_label_path,
            feeds_path=test_feeds_path,
            output_path=bertweet_v35_test_tokenized_path,
            meta_path=bertweet_v35_test_meta_path,
            limit_celebrities=args.limit_celebrities,
        )

    if args.split in ["supp", "all-with-supp"]:
        tokenize_split(
            split_name="supp",
            label_path=supp_label_path,
            feeds_path=supp_feeds_path,
            output_path=bertweet_v35_supp_tokenized_path,
            meta_path=bertweet_v35_supp_meta_path,
            limit_celebrities=args.limit_celebrities,
        )


if __name__ == "__main__":
    main()