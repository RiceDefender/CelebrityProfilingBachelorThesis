# -------------------------------------------------------------------
# BERTweet V3.5 tokenizer config: controlled follower-feed sampling
# -------------------------------------------------------------------

import os

from _constants import preprocessing_data_dir

MODEL_NAME = "vinai/bertweet-base"
VERSION = "bertweet_v3_5_controlled_sampling"

# Main controlled sampling variant from simulation:
# balanced_20_followers_50_tweets
SAMPLING_NAME = "balanced_20_followers_50_tweets"
MAX_FOLLOWERS_PER_CELEBRITY = 20
MAX_TWEETS_PER_FOLLOWER = 50
FOLLOWER_SELECTION_STRATEGY = "random"
TWEET_SELECTION_STRATEGY = "random"
RANDOM_SEED = 42

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

KEEP_HASHTAGS = True
KEEP_EMOJIS = True
KEEP_PUNCTUATION = True
LOWERCASE = False

# -------------------------------------------------------------------
# Separate output directory for V3.5
# -------------------------------------------------------------------
bertweet_v35_processed_dir = os.path.join(
    preprocessing_data_dir,
    "bertweet_v3_5_controlled_sampling_tokenized_chunked",
)

bertweet_v35_train_tokenized_path = os.path.join(
    bertweet_v35_processed_dir,
    "train_tokenized.ndjson",
)
bertweet_v35_test_tokenized_path = os.path.join(
    bertweet_v35_processed_dir,
    "test_tokenized.ndjson",
)

bertweet_v35_train_meta_path = os.path.join(
    bertweet_v35_processed_dir,
    "train_meta.json",
)
bertweet_v35_test_meta_path = os.path.join(
    bertweet_v35_processed_dir,
    "test_meta.json",
)
bertweet_v35_supp_tokenized_path = os.path.join(
    bertweet_v35_processed_dir,
    "supp_tokenized.ndjson",
)
bertweet_v35_supp_meta_path = os.path.join(
    bertweet_v35_processed_dir,
    "supp_meta.json",
)