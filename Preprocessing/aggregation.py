from typing import Any, Dict, List, Optional


def build_text_from_followers(
    follower_groups: List[List[str]],
    tweet_sep: str,
    follower_sep: str,
    max_followers: Optional[int] = None,
    max_tweets_per_follower: Optional[int] = None,
    max_chars: Optional[int] = 10000,
) -> str:
    parts: List[str] = []
    current_len = 0

    selected_followers = follower_groups[:max_followers] if max_followers is not None else follower_groups

    for follower_idx, tweets in enumerate(selected_followers):
        selected_tweets = tweets[:max_tweets_per_follower] if max_tweets_per_follower is not None else tweets
        clean_tweets = [t for t in selected_tweets if isinstance(t, str) and t.strip()]

        if not clean_tweets:
            continue

        if parts:
            sep = f" {follower_sep} "
            if max_chars is not None and current_len + len(sep) > max_chars:
                break
            parts.append(sep)
            current_len += len(sep)

        for tweet_idx, tweet in enumerate(clean_tweets):
            if tweet_idx > 0:
                sep = f" {tweet_sep} "
                if max_chars is not None and current_len + len(sep) > max_chars:
                    return "".join(parts).strip()
                parts.append(sep)
                current_len += len(sep)

            if max_chars is not None and current_len + len(tweet) > max_chars:
                remaining = max_chars - current_len
                if remaining > 0:
                    parts.append(tweet[:remaining])
                return "".join(parts).strip()

            parts.append(tweet)
            current_len += len(tweet)

    return "".join(parts).strip()


def build_examples(
    labels: List[Dict[str, Any]],
    feed_map: Dict[str, List[List[str]]],
    tweet_sep: str,
    follower_sep: str,
    max_followers: Optional[int] = None,
    max_tweets_per_follower: Optional[int] = None,
    max_chars: Optional[int] = 10000,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []

    for label in labels:
        rec_id = label["id"]
        if rec_id is None:
            continue

        follower_groups = feed_map.get(rec_id, [])
        if not follower_groups:
            continue

        text = build_text_from_followers(
            follower_groups=follower_groups,
            tweet_sep=tweet_sep,
            follower_sep=follower_sep,
            max_followers=max_followers,
            max_tweets_per_follower=max_tweets_per_follower,
            max_chars=max_chars,
        )

        if not text:
            continue

        examples.append(
            {
                "id": rec_id,
                "text": text,
                "birthyear": label["birthyear"],
                "gender": label["gender"],
                "occupation": label["occupation"],
                "num_followers_used": len(follower_groups) if max_followers is None else min(len(follower_groups), max_followers),
                "num_chars": len(text),
            }
        )

    return examples