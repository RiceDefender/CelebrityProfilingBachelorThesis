import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

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

from Preprocessing.tokenizers.bertweet.config_bertweet_v34_stopwords import (
    MODEL_NAME,
    VERSION,
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
    REMOVE_STOPWORDS,
    KEEP_PRONOUNS,
    KEEP_NEGATIONS,
    REMOVE_SOCIAL_TOKENS,
    REMOVE_RT_ARTIFACTS,
    HARD_STOPWORDS,
    PRONOUNS_TO_KEEP,
    NEGATIONS_TO_KEEP,
    TWITTER_ARTIFACTS,
    bertweet_v34_processed_dir,
    bertweet_v34_train_tokenized_path,
    bertweet_v34_test_tokenized_path,
    bertweet_v34_supp_tokenized_path,
    bertweet_v34_train_meta_path,
    bertweet_v34_test_meta_path,
    bertweet_v34_supp_meta_path,
)

# ---------------------------------------------------------
# IO helpers
# ---------------------------------------------------------
def ensure_dirs():
    os.makedirs(bertweet_v34_processed_dir, exist_ok=True)


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
        for key in ["text", "full_text", "tweet", "content"]:
            value = obj.get(key)
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    texts.append(cleaned)
            elif isinstance(value, (list, dict)):
                texts.extend(extract_texts_recursive(value))
        for key in ["tweets", "followers", "feed", "feeds", "items", "data"]:
            value = obj.get(key)
            if isinstance(value, (list, dict)):
                texts.extend(extract_texts_recursive(value))
        return texts
    return texts


def extract_tweets_from_feed_row(row: dict) -> List[str]:
    tweets = []
    for key in ["tweets", "followers", "text", "feed", "feeds", "items", "data"]:
        if key in row:
            tweets.extend(extract_texts_recursive(row[key]))

    seen = set()
    unique_tweets = []
    for tweet in tweets:
        if tweet not in seen:
            seen.add(tweet)
            unique_tweets.append(tweet)
    return unique_tweets

# ---------------------------------------------------------
# BERTweet normalization + V3.4 stopword filtering
# ---------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")

# Captures BERTweet-normalized social tokens, hashtags, words, digits, punctuation, whitespace.
# Keeping whitespace tokens makes it possible to preserve a readable text while deleting selected words.
_TOKEN_RE = re.compile(
    r"HTTPURL|@USER|#[A-Za-z0-9_]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]|\s+",
    flags=re.UNICODE,
)


def normalize_for_bertweet(text: str) -> str:
    text = str(text)
    if NORMALIZE_URLS:
        text = _URL_RE.sub(URL_TOKEN, text)
    if NORMALIZE_MENTIONS:
        text = _MENTION_RE.sub(MENTION_TOKEN, text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_stopword_set(keep_pronouns: bool, keep_negations: bool, remove_rt_artifacts: bool) -> set:
    stopwords = set(HARD_STOPWORDS)
    if not keep_pronouns:
        stopwords.update(PRONOUNS_TO_KEEP)
    if not keep_negations:
        stopwords.update(NEGATIONS_TO_KEEP)
    if remove_rt_artifacts:
        stopwords.update(TWITTER_ARTIFACTS)
    return stopwords


def filter_text_for_v34(
    text: str,
    remove_stopwords: bool = REMOVE_STOPWORDS,
    keep_pronouns: bool = KEEP_PRONOUNS,
    keep_negations: bool = KEEP_NEGATIONS,
    remove_social_tokens: bool = REMOVE_SOCIAL_TOKENS,
    remove_rt_artifacts: bool = REMOVE_RT_ARTIFACTS,
) -> Tuple[str, dict]:
    """
    Normalizes Twitter text for BERTweet and optionally removes a conservative stopword set.

    Design choice:
    - This is an ablation tokenizer, not the default BERTweet tokenizer.
    - Mentions/URLs can be kept for BERTweet, but can also be removed via CLI.
    - Pronouns and negations are kept by default because they can carry author-profile/style signals.
    """
    normalized = normalize_for_bertweet(text)
    stopwords = build_stopword_set(
        keep_pronouns=keep_pronouns,
        keep_negations=keep_negations,
        remove_rt_artifacts=remove_rt_artifacts,
    )

    stats = {
        "stopwords_removed": 0,
        "social_tokens_removed": 0,
        "rt_artifacts_removed": 0,
    }

    if not remove_stopwords and not remove_social_tokens:
        return normalized, stats

    out = []
    for match in _TOKEN_RE.finditer(normalized):
        token = match.group(0)
        token_lower = token.lower()

        if token.isspace():
            out.append(" ")
            continue

        if remove_social_tokens and token in {URL_TOKEN, MENTION_TOKEN}:
            stats["social_tokens_removed"] += 1
            out.append(" ")
            continue

        # Hashtags are kept as signal. We do not stopword-filter inside hashtags here.
        if token.startswith("#"):
            out.append(token)
            continue

        # Stopword removal only applies to alphabetic tokens.
        if remove_stopwords and re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", token):
            normalized_token = token_lower.strip("'")
            if normalized_token in stopwords:
                stats["stopwords_removed"] += 1
                if normalized_token in TWITTER_ARTIFACTS:
                    stats["rt_artifacts_removed"] += 1
                out.append(" ")
                continue

        out.append(token)

    filtered = "".join(out)
    filtered = _WHITESPACE_RE.sub(" ", filtered).strip()

    # Remove spaces before common punctuation to keep chunk_text readable.
    filtered = re.sub(r"\s+([.,!?;:])", r"\1", filtered)

    return filtered, stats

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
    remove_stopwords: bool,
    keep_pronouns: bool,
    keep_negations: bool,
    remove_social_tokens: bool,
    remove_rt_artifacts: bool,
) -> Tuple[List[dict], dict]:
    processed_tweets = []
    aggregate_stats = {
        "stopwords_removed": 0,
        "social_tokens_removed": 0,
        "rt_artifacts_removed": 0,
        "num_empty_after_filter": 0,
    }

    for t in tweets:
        if not isinstance(t, str) or not t.strip():
            continue
        filtered, stats = filter_text_for_v34(
            t,
            remove_stopwords=remove_stopwords,
            keep_pronouns=keep_pronouns,
            keep_negations=keep_negations,
            remove_social_tokens=remove_social_tokens,
            remove_rt_artifacts=remove_rt_artifacts,
        )
        for k, v in stats.items():
            aggregate_stats[k] += v
        if filtered:
            processed_tweets.append(filtered)
        else:
            aggregate_stats["num_empty_after_filter"] += 1

    joined_text = " ".join(processed_tweets).strip()
    if not joined_text:
        return [], aggregate_stats

    token_ids = tokenizer.encode(
        joined_text,
        add_special_tokens=False,
        truncation=False,
    )

    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    max_content_length = max_length - num_special_tokens
    if max_content_length <= 0:
        raise ValueError(f"MAX_LENGTH={max_length} is too small for tokenizer special tokens.")

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

    return chunks, aggregate_stats

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
    remove_stopwords: bool = REMOVE_STOPWORDS,
    keep_pronouns: bool = KEEP_PRONOUNS,
    keep_negations: bool = KEEP_NEGATIONS,
    remove_social_tokens: bool = REMOVE_SOCIAL_TOKENS,
    remove_rt_artifacts: bool = REMOVE_RT_ARTIFACTS,
):
    print(f"\n========== Tokenizing BERTweet split: {split_name} ({VERSION}) ==========")
    print(f"[INFO] Labels: {label_path}")
    print(f"[INFO] Feeds:  {feeds_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Meta:   {meta_path}")
    print(f"[INFO] remove_stopwords={remove_stopwords}")
    print(f"[INFO] keep_pronouns={keep_pronouns}")
    print(f"[INFO] keep_negations={keep_negations}")
    print(f"[INFO] remove_social_tokens={remove_social_tokens}")
    print(f"[INFO] remove_rt_artifacts={remove_rt_artifacts}")

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

    total_removed = {
        "stopwords_removed": 0,
        "social_tokens_removed": 0,
        "rt_artifacts_removed": 0,
        "num_empty_after_filter": 0,
    }

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
                    sample_raw = tweets[0][:300]
                    sample_filtered, sample_stats = filter_text_for_v34(
                        tweets[0],
                        remove_stopwords=remove_stopwords,
                        keep_pronouns=keep_pronouns,
                        keep_negations=keep_negations,
                        remove_social_tokens=remove_social_tokens,
                        remove_rt_artifacts=remove_rt_artifacts,
                    )
                    print("[DEBUG] first raw tweet sample:", sample_raw)
                    print("[DEBUG] first filtered tweet sample:", sample_filtered[:300])
                    print("[DEBUG] first filtered stats:", sample_stats)

            processed_celebrities += 1

            if celebrity_id not in labels:
                missing_labels += 1
                continue

            chunks, filter_stats = build_token_chunks(
                tokenizer=tokenizer,
                tweets=tweets,
                max_length=MAX_LENGTH,
                stride=STRIDE,
                max_chunks=MAX_CHUNKS_PER_CELEBRITY,
                min_tokens_per_chunk=MIN_TOKENS_PER_CHUNK,
                remove_stopwords=remove_stopwords,
                keep_pronouns=keep_pronouns,
                keep_negations=keep_negations,
                remove_social_tokens=remove_social_tokens,
                remove_rt_artifacts=remove_rt_artifacts,
            )

            for k in total_removed:
                total_removed[k] += filter_stats.get(k, 0)

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
                    "preprocessing": {
                        "remove_stopwords": remove_stopwords,
                        "keep_pronouns": keep_pronouns,
                        "keep_negations": keep_negations,
                        "remove_social_tokens": remove_social_tokens,
                        "remove_rt_artifacts": remove_rt_artifacts,
                    },
                }
                write_ndjson_row(out_f, out_row)
                written_chunks += 1

            meta_rows.append({
                "celebrity_id": celebrity_id,
                "num_raw_tweets": num_raw_tweets,
                "num_used_tweets": len(tweets),
                "num_chunks": len(chunks),
                "max_length": MAX_LENGTH,
                "stride": STRIDE,
                "model_name": MODEL_NAME,
                "version": VERSION,
                "occupation": label_info["occupation"],
                "gender": label_info["gender"],
                "birthyear": label_info["birthyear"],
                "max_chunks_per_celebrity": MAX_CHUNKS_PER_CELEBRITY,
                "filter_stats": filter_stats,
                "preprocessing": {
                    "remove_stopwords": remove_stopwords,
                    "keep_pronouns": keep_pronouns,
                    "keep_negations": keep_negations,
                    "remove_social_tokens": remove_social_tokens,
                    "remove_rt_artifacts": remove_rt_artifacts,
                },
            })

            written_celebrities += 1
            if written_celebrities % 50 == 0:
                print(
                    f"[PROGRESS] {split_name}: "
                    f"celebrities={written_celebrities} chunks={written_chunks}"
                )

    save_json(meta_rows, meta_path)

    print(f"[OK] Saved tokenized NDJSON: {output_path}")
    print(f"[OK] Saved meta JSON:        {meta_path}")
    print(f"[INFO] Processed celebrities: {processed_celebrities}")
    print(f"[INFO] Written celebrities:   {written_celebrities}")
    print(f"[INFO] Tokenized chunks:      {written_chunks}")
    print(f"[INFO] Missing labels:        {missing_labels}")
    print(f"[INFO] Empty chunks:          {empty_chunks}")
    print(f"[INFO] Total filter stats:    {total_removed}")

    if meta_rows:
        avg_chunks = sum(r["num_chunks"] for r in meta_rows) / len(meta_rows)
        avg_removed = sum(r["filter_stats"].get("stopwords_removed", 0) for r in meta_rows) / len(meta_rows)
        print(f"[INFO] Avg chunks/celeb:      {avg_chunks:.2f}")
        print(f"[INFO] Avg stopwords removed: {avg_removed:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize PAN Celebrity Profiling data for BERTweet V3.4 with conservative stopword filtering."
    )
    parser.add_argument("--split", choices=["train", "test", "supp", "all"], default="all")
    parser.add_argument("--limit-celebrities", type=int, default=None)

    parser.add_argument("--no-stopwords", action="store_true", help="Disable stopword removal for debugging/ablation.")
    parser.add_argument("--drop-pronouns", action="store_true", help="Also remove pronouns. Default keeps pronouns.")
    parser.add_argument("--drop-negations", action="store_true", help="Also remove negations. Default keeps negations.")
    parser.add_argument("--remove-social-tokens", action="store_true", help="Remove @USER and HTTPURL before BERTweet tokenization.")
    parser.add_argument("--keep-rt-artifacts", action="store_true", help="Keep rt/via/amp artifacts. Default removes them.")

    args = parser.parse_args()
    ensure_dirs()

    remove_stopwords = not args.no_stopwords
    keep_pronouns = not args.drop_pronouns
    keep_negations = not args.drop_negations
    remove_social_tokens = args.remove_social_tokens
    remove_rt_artifacts = not args.keep_rt_artifacts

    if args.split in ["train", "all"]:
        tokenize_split(
            split_name="train",
            label_path=train_label_path,
            feeds_path=train_feeds_path,
            output_path=bertweet_v34_train_tokenized_path,
            meta_path=bertweet_v34_train_meta_path,
            limit_celebrities=args.limit_celebrities,
            remove_stopwords=remove_stopwords,
            keep_pronouns=keep_pronouns,
            keep_negations=keep_negations,
            remove_social_tokens=remove_social_tokens,
            remove_rt_artifacts=remove_rt_artifacts,
        )

    if args.split in ["test", "all"]:
        tokenize_split(
            split_name="test",
            label_path=test_label_path,
            feeds_path=test_feeds_path,
            output_path=bertweet_v34_test_tokenized_path,
            meta_path=bertweet_v34_test_meta_path,
            limit_celebrities=args.limit_celebrities,
            remove_stopwords=remove_stopwords,
            keep_pronouns=keep_pronouns,
            keep_negations=keep_negations,
            remove_social_tokens=remove_social_tokens,
            remove_rt_artifacts=remove_rt_artifacts,
        )

    if args.split in ["supp", "all"]:
        tokenize_split(
            split_name="supp",
            label_path=supp_label_path,
            feeds_path=supp_feeds_path,
            output_path=bertweet_v34_supp_tokenized_path,
            meta_path=bertweet_v34_supp_meta_path,
            limit_celebrities=args.limit_celebrities,
            remove_stopwords=remove_stopwords,
            keep_pronouns=keep_pronouns,
            keep_negations=keep_negations,
            remove_social_tokens=remove_social_tokens,
            remove_rt_artifacts=remove_rt_artifacts,
        )


if __name__ == "__main__":
    main()
