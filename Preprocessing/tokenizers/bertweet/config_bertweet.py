# -------------------------------------------------------------------
# BERTweet tokenizer config
# -------------------------------------------------------------------

MODEL_NAME = "vinai/bertweet-base"

MAX_TWEETS_PER_CELEBRITY = 5000
TWEET_SELECTION_STRATEGY = "evenly_spaced"

# BERTweet vinai/bertweet-base supports effective input length 128.
MAX_LENGTH = 128

# Optional overlap between chunks.
# Helps to reduce hard boundary effects.
STRIDE = 32

# None = keep all generated chunks.
MAX_CHUNKS_PER_CELEBRITY = None

# Keep short/noisy chunks out.
MIN_TOKENS_PER_CHUNK = 16

# BERTweet normalization style
URL_TOKEN = "HTTPURL"
MENTION_TOKEN = "@USER"

NORMALIZE_URLS = True
NORMALIZE_MENTIONS = True

# Important: do not remove these for BERTweet.
KEEP_HASHTAGS = True
KEEP_EMOJIS = True
KEEP_PUNCTUATION = True
LOWERCASE = False