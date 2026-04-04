from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from normalize import normalize_tweet


def first_present(record: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in record:
            return record[key]
    return None


def find_id(record: Dict[str, Any]) -> Optional[str]:
    value = first_present(
        record,
        [
            "author",
            "id",
            "author_id",
            "user_id",
            "celebrity_id",
            "profile_id",
            "twitter_id",
        ],
    )
    if value is None:
        return None
    return str(value)


def extract_label_record(record: Dict[str, Any]) -> Dict[str, Any]:
    rec_id = find_id(record)

    birthyear = first_present(record, ["birthyear", "birth_year", "year", "yob", "age"])
    gender = first_present(record, ["gender", "sex"])
    occupation = first_present(record, ["occupation", "job", "profession", "occ"])

    return {
        "id": rec_id,
        "birthyear": birthyear,
        "gender": gender,
        "occupation": occupation,
        "raw": record,
    }


def extract_texts_recursive(obj: Any) -> List[str]:
    texts: List[str] = []

    if obj is None:
        return texts

    if isinstance(obj, str):
        cleaned = obj.strip()
        return [cleaned] if cleaned else []

    if isinstance(obj, list):
        for item in obj:
            texts.extend(extract_texts_recursive(item))
        return texts

    if isinstance(obj, dict):
        for key in [
            "text",
            "full_text",
            "tweet",
            "content",
            "body",
            "tweets",
            "texts",
            "feed",
            "timeline",
            "followers",
            "items",
            "data",
        ]:
            if key in obj:
                texts.extend(extract_texts_recursive(obj[key]))
        return texts

    return texts


def infer_follower_groups(record: Dict[str, Any]) -> List[List[str]]:
    """
    Erwarteter PAN-Fall:
    {"id": ..., "text": [[tweet1, tweet2, ...], [tweet1, ...], ...]}
    """
    if "text" in record:
        value = record["text"]

        if isinstance(value, list):
            if value and all(isinstance(item, list) for item in value):
                grouped = []
                for follower_list in value:
                    tweets = [t.strip() for t in follower_list if isinstance(t, str) and t.strip()]
                    if tweets:
                        grouped.append(tweets)
                if grouped:
                    return grouped

            flat = [t.strip() for t in value if isinstance(t, str) and t.strip()]
            if flat:
                return [flat]

    flat_texts = extract_texts_recursive(record)
    flat_texts = [t for t in flat_texts if isinstance(t, str) and t.strip()]
    if flat_texts:
        return [flat_texts]

    return []


def aggregate_feeds_by_id(feed_rows: List[Dict[str, Any]]) -> Dict[str, List[List[str]]]:
    grouped: Dict[str, List[List[str]]] = defaultdict(list)

    for row in feed_rows:
        rec_id = find_id(row)
        if rec_id is None:
            continue

        follower_groups = infer_follower_groups(row)
        for follower_tweets in follower_groups:
            clean = [normalize_tweet(t) for t in follower_tweets if isinstance(t, str) and t.strip()]
            if clean:
                grouped[rec_id].append(clean)

    return grouped