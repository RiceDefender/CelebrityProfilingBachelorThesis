# -------------------------------------------------------------------
# BERTweet V3.4 tokenizer config: stopword-filtered experiment
# -------------------------------------------------------------------

import os

from _constants import preprocessing_data_dir

# Keep the same base model and chunking assumptions as BERTweet V3.
MODEL_NAME = "vinai/bertweet-base"
VERSION = "bertweet_v3_4_stopwords"

MAX_TWEETS_PER_CELEBRITY = 5000
TWEET_SELECTION_STRATEGY = "evenly_spaced"

# BERTweet vinai/bertweet-base supports effective input length 128.
MAX_LENGTH = 128
STRIDE = 32

# None = keep all generated chunks.
MAX_CHUNKS_PER_CELEBRITY = None
MIN_TOKENS_PER_CHUNK = 16

# BERTweet normalization style
URL_TOKEN = "HTTPURL"
MENTION_TOKEN = "@USER"
NORMALIZE_URLS = True
NORMALIZE_MENTIONS = True

# -------------------------------------------------------------------
# V3.4 filtering switches
# -------------------------------------------------------------------
# Core experiment: remove high-frequency function words before BERTweet tokenization.
REMOVE_STOPWORDS = True

# Safer defaults: keep words that are often useful in author profiling.
KEEP_PRONOUNS = True
KEEP_NEGATIONS = True

# Social tokens are useful signals but often dominate N-grams.
# For this BERTweet experiment the safer default is to keep them.
# CLI flags in tokenize_bertweet_v34_stopwords.py can override this.
REMOVE_SOCIAL_TOKENS = False
REMOVE_RT_ARTIFACTS = True

# Keep Twitter-specific signals for BERTweet unless you intentionally ablate them.
KEEP_HASHTAGS = True
KEEP_EMOJIS = True
KEEP_PUNCTUATION = True
LOWERCASE = False

# -------------------------------------------------------------------
# Stopword policy
# -------------------------------------------------------------------
# Hard stopwords: mostly function words that often dominate N-grams.
# Deliberately excludes pronouns and negations; those are controlled by KEEP_* flags.
HARD_STOPWORDS = {
    "a", "an", "the",
    "and", "or", "but",
    "of", "in", "on", "at", "to", "from", "for", "with", "by", "as",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing",
    "have", "has", "had", "having",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "this", "that", "these", "those",
    "there", "here", "then", "than",
    "so", "very", "just", "also", "only", "even", "more", "most", "less", "least",
    "about", "into", "over", "under", "after", "before", "between", "through",
    "up", "down", "out", "off", "again", "once",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
}

PRONOUNS_TO_KEEP = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "we", "us", "our", "ours", "ourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves",
    "it", "its", "itself",
}

NEGATIONS_TO_KEEP = {
    "no", "not", "nor", "never", "none", "nothing", "nobody", "nowhere",
    "without", "cannot", "can't", "dont", "don't", "didnt", "didn't", "isnt", "isn't",
    "wasnt", "wasn't", "wont", "won't", "wouldnt", "wouldn't", "shouldnt", "shouldn't",
}

TWITTER_ARTIFACTS = {
    "rt", "via", "amp", "gt", "lt",
}

# -------------------------------------------------------------------
# Separate output directory for V3.4
# -------------------------------------------------------------------
bertweet_v34_processed_dir = os.path.join(
    preprocessing_data_dir,
    "bertweet_v3_4_stopwords_tokenized_chunked",
)

bertweet_v34_train_tokenized_path = os.path.join(
    bertweet_v34_processed_dir,
    "train_tokenized.ndjson",
)
bertweet_v34_test_tokenized_path = os.path.join(
    bertweet_v34_processed_dir,
    "test_tokenized.ndjson",
)
bertweet_v34_supp_tokenized_path = os.path.join(
    bertweet_v34_processed_dir,
    "supp_tokenized.ndjson",
)

bertweet_v34_train_meta_path = os.path.join(bertweet_v34_processed_dir, "train_meta.json")
bertweet_v34_test_meta_path = os.path.join(bertweet_v34_processed_dir, "test_meta.json")
bertweet_v34_supp_meta_path = os.path.join(bertweet_v34_processed_dir, "supp_meta.json")
