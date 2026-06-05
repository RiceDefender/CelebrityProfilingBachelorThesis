from typing import Any, Dict, List, Optional


def chunk_follower_groups_by_tweets(
    follower_groups: List[List[str]],
    tweets_per_chunk: int = 12,
    max_followers: Optional[int] = None,
    max_tweets_per_follower: Optional[int] = None,
) -> List[List[str]]:
    """
    Erzeugt Chunks als Listen von Tweets.

    Strategie:
    - follower_groups = List[List[str]]   -> followers -> tweets
    - optional follower/tweet limits anwenden
    - alle Tweets in eine flache Liste bringen
    - in Blöcke von tweets_per_chunk aufteilen
    """
    selected_followers = follower_groups[:max_followers] if max_followers is not None else follower_groups

    flat_tweets: List[str] = []

    for tweets in selected_followers:
        selected_tweets = tweets[:max_tweets_per_follower] if max_tweets_per_follower is not None else tweets
        clean = [t for t in selected_tweets if isinstance(t, str) and t.strip()]
        flat_tweets.extend(clean)

    chunks: List[List[str]] = []
    for i in range(0, len(flat_tweets), tweets_per_chunk):
        chunk = flat_tweets[i:i + tweets_per_chunk]
        if chunk:
            chunks.append(chunk)

    return chunks


def build_text_from_tweet_chunk(
    tweet_chunk: List[str],
    tweet_sep: str,
    max_chars: Optional[int] = None,
) -> str:
    """
    Baut aus einem Tweet-Chunk einen Text.
    """
    parts: List[str] = []
    current_len = 0

    for idx, tweet in enumerate(tweet_chunk):
        if idx > 0:
            sep = f" {tweet_sep} "
            if max_chars is not None and current_len + len(sep) > max_chars:
                break
            parts.append(sep)
            current_len += len(sep)

        if max_chars is not None and current_len + len(tweet) > max_chars:
            remaining = max_chars - current_len
            if remaining > 0:
                parts.append(tweet[:remaining])
            break

        parts.append(tweet)
        current_len += len(tweet)

    return "".join(parts).strip()


def build_chunked_examples(
    labels: List[Dict[str, Any]],
    feed_map: Dict[str, List[List[str]]],
    tweet_sep: str,
    tweets_per_chunk: int = 12,
    max_followers: Optional[int] = None,
    max_tweets_per_follower: Optional[int] = None,
    max_chars: Optional[int] = 3000,
) -> List[Dict[str, Any]]:
    """
    Baut mehrere Beispiele pro Celebrity:
      celebrity_id + chunk_id + text + labels
    """
    examples: List[Dict[str, Any]] = []

    for label in labels:
        rec_id = label["id"]
        if rec_id is None:
            continue

        follower_groups = feed_map.get(rec_id, [])
        if not follower_groups:
            continue

        chunks = chunk_follower_groups_by_tweets(
            follower_groups=follower_groups,
            tweets_per_chunk=tweets_per_chunk,
            max_followers=max_followers,
            max_tweets_per_follower=max_tweets_per_follower,
        )

        for chunk_idx, tweet_chunk in enumerate(chunks):
            text = build_text_from_tweet_chunk(
                tweet_chunk=tweet_chunk,
                tweet_sep=tweet_sep,
                max_chars=max_chars,
            )

            if not text:
                continue

            examples.append(
                {
                    "celebrity_id": rec_id,
                    "chunk_id": f"{rec_id}_{chunk_idx}",
                    "text": text,
                    "birthyear": label["birthyear"],
                    "gender": label["gender"],
                    "occupation": label["occupation"],
                    "num_tweets_in_chunk": len(tweet_chunk),
                    "num_chars": len(text),
                }
            )

    return examples