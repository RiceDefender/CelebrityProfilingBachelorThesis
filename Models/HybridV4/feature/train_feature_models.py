import argparse
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.multiclass import OneVsRestClassifier


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
    hybrid_v4_feature_models_dir,
    hybrid_v4_feature_predictions_dir,
    hybrid_v4_feature_metrics_dir,
)

from Models.HybridV4.feature.config_features import (
    RANDOM_SEED,
    TARGETS,
    LABEL_ORDERS,
    WORD_NGRAM_RANGE,
    CHAR_NGRAM_RANGE,
    WORD_MAX_FEATURES,
    CHAR_MAX_FEATURES,
    MIN_DF,
    MAX_DF,
    USE_WORD_TFIDF,
    USE_CHAR_TFIDF,
    USE_STYLE_FEATURES,
    MAX_ITER,
    C,
    CLASS_WEIGHT,
    REMOVE_STOPWORDS_FOR_WORD_TFIDF,
    DROP_SOCIAL_TOKENS_FOR_WORD_TFIDF,
    NORMALIZE_HASHTAGS_FOR_WORD_TFIDF,
    SOCIAL_NGRAM_TOKENS,
    DEDUPLICATE_TWEETS_FOR_WORD_TFIDF,
    MAX_TWEETS_PER_CELEBRITY,
)


VAL_RATIO = 0.1

URL_RE = re.compile(r"(https?://\S+|www\.\S+|HTTPURL)")
MENTION_RE = re.compile(r"(@\w+|@USER)")
HASHTAG_RE = re.compile(r"#\w+")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)


def ensure_dirs():
    os.makedirs(hybrid_v4_feature_models_dir, exist_ok=True)
    os.makedirs(hybrid_v4_feature_predictions_dir, exist_ok=True)
    os.makedirs(hybrid_v4_feature_metrics_dir, exist_ok=True)


def iter_ndjson(path: str):
    print(f"[INFO] Streaming: {path}")
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


def load_ndjson(path: str) -> List[dict]:
    return list(iter_ndjson(path))


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def normalize_tweet_text(text: str) -> str:
    text = URL_RE.sub(" HTTPURL ", text)
    text = MENTION_RE.sub(" @USER ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_celebrity_id(row: dict) -> str:
    for key in ["id", "celebrity_id", "author_id"]:
        if key in row:
            return str(row[key])
    raise KeyError(f"No celebrity id key found in row: {sorted(row.keys())}")


def get_feed_celebrity_id(row: dict) -> str:
    for key in ["id", "celebrity_id", "author_id"]:
        if key in row:
            return str(row[key])
    raise KeyError(f"No celebrity id key found in feed row: {sorted(row.keys())}")

TOKEN_RE = re.compile(r"[#@]?[A-Za-z0-9_']+")


def clean_token_for_word_tfidf(token: str):
    token = token.lower().strip()

    if not token:
        return None

    if token.startswith("@"):
        return None

    if token.startswith("#"):
        if NORMALIZE_HASHTAGS_FOR_WORD_TFIDF:
            token = token[1:]
        else:
            token = token.replace("#", "hashtag_")

    if DROP_SOCIAL_TOKENS_FOR_WORD_TFIDF and token in SOCIAL_NGRAM_TOKENS:
        return None

    if REMOVE_STOPWORDS_FOR_WORD_TFIDF and token in ENGLISH_STOP_WORDS:
        return None

    if len(token) < 2:
        return None

    return token


def word_tfidf_tokenizer(text: str):
    tokens = TOKEN_RE.findall(text)
    cleaned = []

    for token in tokens:
        token = clean_token_for_word_tfidf(token)
        if token is not None:
            cleaned.append(token)

    return cleaned


def extract_tweets_from_feed_row(row: dict) -> List[str]:
    """
    PAN follower-feeds format:
    {
        "id": celebrity_id,
        "text": [
            [tweet_1, tweet_2, ...],   # follower 1
            [tweet_1, tweet_2, ...],   # follower 2
            ...
        ]
    }

    This function also supports flat lists as fallback.
    """
    tweets = []

    value = row.get("text")

    if isinstance(value, str):
        tweets.append(value)

    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                tweets.append(item)

            elif isinstance(item, list):
                for nested in item:
                    if isinstance(nested, str):
                        tweets.append(nested)
                    elif isinstance(nested, dict):
                        text = (
                            nested.get("text")
                            or nested.get("tweet")
                            or nested.get("content")
                            or nested.get("full_text")
                        )
                        if isinstance(text, str):
                            tweets.append(text)

            elif isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("tweet")
                    or item.get("content")
                    or item.get("full_text")
                )
                if isinstance(text, str):
                    tweets.append(text)

    return [
        normalize_tweet_text(t)
        for t in tweets
        if isinstance(t, str) and t.strip()
    ]


def load_labels(label_path: str) -> Dict[str, dict]:
    labels = {}
    for row in iter_ndjson(label_path):
        cid = get_celebrity_id(row)
        labels[cid] = row
    return labels


def build_examples(label_path: str, feeds_path: str) -> List[dict]:
    labels = load_labels(label_path)

    examples_by_id = {}
    for cid, label_row in labels.items():
        examples_by_id[cid] = {
            "celebrity_id": cid,
            "text": "",
            "tweet_count": 0,
            "unique_tweet_count": 0,
            "duplicate_tweet_ratio": 0.0,
            "top_tweet_share": 0.0,
            "occupation": str(label_row["occupation"]),
            "gender": str(label_row["gender"]),
            "birthyear": int(label_row["birthyear"]),
        }

    missing_text = 0

    for row in iter_ndjson(feeds_path):
        cid = get_feed_celebrity_id(row)
        if cid not in examples_by_id:
            continue

        tweets = extract_tweets_from_feed_row(row)

        total_tweets = len(tweets)
        unique_tweets = len(set(tweets))

        if total_tweets > 0:
            counts = defaultdict(int)
            for tweet in tweets:
                counts[tweet] += 1
            top_tweet_share = max(counts.values()) / total_tweets
            duplicate_tweet_ratio = (total_tweets - unique_tweets) / total_tweets
        else:
            top_tweet_share = 0.0
            duplicate_tweet_ratio = 0.0

        if DEDUPLICATE_TWEETS_FOR_WORD_TFIDF:
            tweets_for_text = list(dict.fromkeys(tweets))
        else:
            tweets_for_text = tweets

        if MAX_TWEETS_PER_CELEBRITY and MAX_TWEETS_PER_CELEBRITY > 0:
            tweets_for_text = tweets_for_text[:MAX_TWEETS_PER_CELEBRITY]

        text = "\n".join(tweets_for_text)

        examples_by_id[cid]["text"] = text
        examples_by_id[cid]["tweet_count"] = total_tweets
        examples_by_id[cid]["unique_tweet_count"] = unique_tweets
        examples_by_id[cid]["duplicate_tweet_ratio"] = duplicate_tweet_ratio
        examples_by_id[cid]["top_tweet_share"] = top_tweet_share

    examples = list(examples_by_id.values())

    for ex in examples:
        if not ex["text"].strip():
            missing_text += 1

    print(f"[INFO] Examples: {len(examples)}")
    print(f"[INFO] Missing/empty texts: {missing_text}")
    print(f"[INFO] Max tweets per celebrity for TF-IDF text: {MAX_TWEETS_PER_CELEBRITY}")

    return examples


def map_birthyear_to_bucket(year) -> str:
    year = int(year)
    buckets = [int(x) for x in LABEL_ORDERS["birthyear"]]
    nearest = min(buckets, key=lambda b: abs(year - b))
    return str(nearest)


def get_label(example: dict, target: str) -> str:
    if target == "birthyear":
        return map_birthyear_to_bucket(example["birthyear"])
    return str(example[target])


def split_train_val(examples: List[dict], target: str):
    labels = LABEL_ORDERS[target]
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    celebrity_ids = [ex["celebrity_id"] for ex in examples]
    example_by_id = {ex["celebrity_id"]: ex for ex in examples}
    y = [label_to_id[get_label(ex, target)] for ex in examples]

    train_ids, val_ids = train_test_split(
        celebrity_ids,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    train_examples = [example_by_id[cid] for cid in train_ids]
    val_examples = [example_by_id[cid] for cid in val_ids]

    return train_examples, val_examples


LOVE_WORDS = {
    "love", "loved", "lovely", "heart", "hearts", "cute", "beautiful",
    "amazing", "happy", "birthday", "miss", "queen", "king"
}

FAN_WORDS = {
    "fan", "fans", "fandom", "vote", "voted", "voting", "retweet",
    "follow", "followers", "stan", "idol", "army", "stream"
}

POLITICS_WORDS = {
    "vote", "election", "president", "minister", "government", "senate",
    "policy", "law", "rights", "democracy", "party", "campaign",
    "leader", "dictator", "communism", "freedom", "speech"
}

SPORTS_WORDS = {
    "game", "team", "match", "league", "season", "win", "won", "goal",
    "score", "coach", "player", "football", "soccer", "basketball",
    "tennis", "cricket", "baseball", "nfl", "nba", "fifa", "dota"
}

CREATOR_WORDS = {
    "youtube", "video", "stream", "live", "blog", "podcast", "photo",
    "facebook", "twitch", "tutorial", "content", "design", "art",
    "artist", "posted", "subscribe", "channel"
}

PERFORMER_WORDS = {
    "song", "album", "music", "concert", "tour", "movie", "actor",
    "actress", "band", "stage", "show", "performance", "dance",
    "singer", "artist", "soundcloud"
}


def count_keywords(tokens, vocab):
    return sum(1 for token in tokens if token in vocab)


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def extract_style_features_from_examples(examples: List[dict]) -> np.ndarray:
    rows = []

    for ex in examples:
        text = ex["text"]

        length_chars = len(text)
        tokens_raw = text.split()
        num_tokens = len(tokens_raw)

        clean_tokens = [
            t.lower().strip("#@.,!?;:()[]{}\"'")
            for t in tokens_raw
        ]
        clean_tokens = [t for t in clean_tokens if t]

        tweet_count = float(ex.get("tweet_count", 0))
        unique_tweet_count = float(ex.get("unique_tweet_count", 0))
        duplicate_tweet_ratio = float(ex.get("duplicate_tweet_ratio", 0.0))
        top_tweet_share = float(ex.get("top_tweet_share", 0.0))

        url_count = len(URL_RE.findall(text))
        mention_count = len(MENTION_RE.findall(text))
        hashtag_count = len(HASHTAG_RE.findall(text))
        emoji_count = len(EMOJI_RE.findall(text))

        exclamation_count = text.count("!")
        question_count = text.count("?")
        uppercase_chars = sum(1 for ch in text if ch.isupper())
        alpha_chars = sum(1 for ch in text if ch.isalpha())

        avg_token_length = (
            float(np.mean([len(t) for t in tokens_raw])) if tokens_raw else 0.0
        )

        avg_tokens_per_tweet = safe_div(num_tokens, tweet_count)

        rows.append([
            length_chars,
            num_tokens,
            tweet_count,
            unique_tweet_count,
            safe_div(unique_tweet_count, tweet_count),
            duplicate_tweet_ratio,
            top_tweet_share,

            url_count,
            mention_count,
            hashtag_count,
            emoji_count,

            safe_div(url_count, tweet_count),
            safe_div(mention_count, tweet_count),
            safe_div(hashtag_count, tweet_count),
            safe_div(emoji_count, tweet_count),

            exclamation_count,
            question_count,
            safe_div(exclamation_count, tweet_count),
            safe_div(question_count, tweet_count),

            uppercase_chars / max(alpha_chars, 1),
            avg_token_length,
            avg_tokens_per_tweet,

            count_keywords(clean_tokens, LOVE_WORDS),
            count_keywords(clean_tokens, FAN_WORDS),
            count_keywords(clean_tokens, POLITICS_WORDS),
            count_keywords(clean_tokens, SPORTS_WORDS),
            count_keywords(clean_tokens, CREATOR_WORDS),
            count_keywords(clean_tokens, PERFORMER_WORDS),
        ])

    return np.asarray(rows, dtype=np.float32)


def build_vectorizers():
    parts = []

    if USE_WORD_TFIDF:
        parts.append((
            "word_tfidf",
            TfidfVectorizer(
                analyzer="word",
                tokenizer=word_tfidf_tokenizer,
                token_pattern=None,
                ngram_range=WORD_NGRAM_RANGE,
                max_features=WORD_MAX_FEATURES,
                min_df=MIN_DF,
                max_df=MAX_DF,
                lowercase=False,
                sublinear_tf=True,
                strip_accents=None,
            ),
        ))

    if USE_CHAR_TFIDF:
        parts.append((
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=CHAR_NGRAM_RANGE,
                max_features=CHAR_MAX_FEATURES,
                min_df=MIN_DF,
                max_df=MAX_DF,
                lowercase=True,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ))

    if not parts:
        raise ValueError("At least one TF-IDF feature type must be enabled.")

    return FeatureUnion(parts)


def build_feature_matrix(examples: List[dict], vectorizer=None, scaler=None, fit: bool = False):
    texts = [ex["text"] for ex in examples]
    if fit:
        vectorizer = build_vectorizers()
        x_tfidf = vectorizer.fit_transform(texts)
    else:
        check_is_fitted(vectorizer)
        x_tfidf = vectorizer.transform(texts)

    if USE_STYLE_FEATURES:
        style = extract_style_features_from_examples(examples)

        if fit:
            scaler = StandardScaler()
            style_scaled = scaler.fit_transform(style)
        else:
            check_is_fitted(scaler)
            style_scaled = scaler.transform(style)

        x = hstack([x_tfidf, csr_matrix(style_scaled)], format="csr")
    else:
        x = x_tfidf
        scaler = None

    return x, vectorizer, scaler


def predict_examples(
    examples: List[dict],
    target: str,
    labels: List[str],
    clf,
    vectorizer,
    scaler,
    split: str,
):
    x, _, _ = build_feature_matrix(
        examples,
        vectorizer=vectorizer,
        scaler=scaler,
        fit=False,
    )

    probs = clf.predict_proba(x)
    pred_ids = np.argmax(probs, axis=1)
    id_to_label = {idx: label for idx, label in enumerate(labels)}
    pred_labels = [id_to_label[int(i)] for i in pred_ids]

    predictions = []
    for ex, prob, pred_label in zip(examples, probs, pred_labels):
        predictions.append({
            "celebrity_id": ex["celebrity_id"],
            "target": target,
            "split": split,
            "true_label": get_label(ex, target),
            "labels": labels,
            "feature_probabilities": prob.tolist(),
            "feature_pred_label": pred_label,
            "feature_model": "tfidf_word_char_style_logreg",
        })

    return predictions


def evaluate_predictions(predictions: List[dict], target: str, split: str):
    labels = LABEL_ORDERS[target]
    y_true = [row["true_label"] for row in predictions]
    y_pred = [row["feature_pred_label"] for row in predictions]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "target": target,
        "split": split,
        "model": "tfidf_word_char_style_logreg",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "labels": labels,
        "classification_report": report,
        "num_celebrities": len(predictions),
    }


def train_one_target(target: str, train_all_examples: List[dict], test_examples: List[dict]):
    print(f"\n========== HybridV4 feature model: target={target} ==========")

    labels = LABEL_ORDERS[target]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    train_examples, val_examples = split_train_val(train_all_examples, target)

    x_train_texts = [ex["text"] for ex in train_examples]
    y_train_labels = [get_label(ex, target) for ex in train_examples]
    y_train = np.asarray([label_to_id[label] for label in y_train_labels], dtype=np.int64)

    print(f"[INFO] Train celebrities: {len(train_examples)}")
    print(f"[INFO] Val celebrities:   {len(val_examples)}")
    print(f"[INFO] Test celebrities:  {len(test_examples)}")
    print(f"[INFO] Labels:            {label_to_id}")

    x_train, vectorizer, scaler = build_feature_matrix(
        train_examples,
        fit=True,
    )

    print(f"[INFO] Feature shape:     {x_train.shape}")

    base_clf = LogisticRegression(
        C=C,
        max_iter=MAX_ITER,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_SEED,
        solver="liblinear",
        verbose=0,
    )

    clf = OneVsRestClassifier(base_clf)

    clf.fit(x_train, y_train)

    val_predictions = predict_examples(
        examples=val_examples,
        target=target,
        labels=labels,
        clf=clf,
        vectorizer=vectorizer,
        scaler=scaler,
        split="val",
    )

    test_predictions = predict_examples(
        examples=test_examples,
        target=target,
        labels=labels,
        clf=clf,
        vectorizer=vectorizer,
        scaler=scaler,
        split="test",
    )

    val_metrics = evaluate_predictions(val_predictions, target, "val")
    test_metrics = evaluate_predictions(test_predictions, target, "test")

    model_payload = {
        "target": target,
        "labels": labels,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "classifier": clf,
    }

    model_path = os.path.join(
        hybrid_v4_feature_models_dir,
        f"{target}_feature_model.pkl",
    )
    val_pred_path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{target}_val_feature_probs.json",
    )
    test_pred_path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{target}_test_feature_probs.json",
    )
    val_metrics_path = os.path.join(
        hybrid_v4_feature_metrics_dir,
        f"{target}_val_feature_metrics.json",
    )
    test_metrics_path = os.path.join(
        hybrid_v4_feature_metrics_dir,
        f"{target}_test_feature_metrics.json",
    )

    save_pickle(model_payload, model_path)
    save_json(val_predictions, val_pred_path)
    save_json(test_predictions, test_pred_path)
    save_json(val_metrics, val_metrics_path)
    save_json(test_metrics, test_metrics_path)

    print(
        f"[RESULT] {target} "
        f"val_acc={val_metrics['accuracy']:.4f} "
        f"val_macro_f1={val_metrics['macro_f1']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}"
    )
    print(f"[OK] Saved model:        {model_path}")
    print(f"[OK] Saved val preds:    {val_pred_path}")
    print(f"[OK] Saved test preds:   {test_pred_path}")


def resolve_targets(target: str) -> List[str]:
    if target == "all":
        return TARGETS
    return [target]


def main():
    parser = argparse.ArgumentParser(
        description="Train HybridV4 classical TF-IDF/style feature models from raw PAN data."
    )
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear", "all"],
        default="all",
    )

    args = parser.parse_args()

    ensure_dirs()

    train_all_examples = build_examples(
        label_path=train_label_path,
        feeds_path=train_feeds_path,
    )
    test_examples = build_examples(
        label_path=test_label_path,
        feeds_path=test_feeds_path,
    )

    for target in resolve_targets(args.target):
        train_one_target(
            target=target,
            train_all_examples=train_all_examples,
            test_examples=test_examples,
        )


if __name__ == "__main__":
    main()