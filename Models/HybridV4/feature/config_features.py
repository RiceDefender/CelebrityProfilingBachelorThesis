# -------------------------------------------------------------------
# HybridV4 feature model config
# -------------------------------------------------------------------

RANDOM_SEED = 42

TARGETS = ["occupation", "gender", "birthyear"]

LABEL_ORDERS = {
    "occupation": ["sports", "performer", "creator", "politics"],
    "gender": ["male", "female"],
    "birthyear": ["1994", "1985", "1975", "1963", "1947"],
}

# TF-IDF settings
WORD_NGRAM_RANGE = (1, 3)
CHAR_NGRAM_RANGE = (3, 5)

WORD_MAX_FEATURES = 40_000
CHAR_MAX_FEATURES = 10_000

MIN_DF = 3
MAX_DF = 0.85

USE_WORD_TFIDF = True
USE_CHAR_TFIDF = True
USE_STYLE_FEATURES = True

# Clean word tokenizer
REMOVE_STOPWORDS_FOR_WORD_TFIDF = True
DROP_SOCIAL_TOKENS_FOR_WORD_TFIDF = True
NORMALIZE_HASHTAGS_FOR_WORD_TFIDF = True

SOCIAL_NGRAM_TOKENS = {
    "rt",
    "via",
    "amp",
    "gt",
    "lt",
    "httpurl",
    "url",
    "user",
}

# Repetition control
DEDUPLICATE_TWEETS_FOR_WORD_TFIDF = True
MAX_TWEETS_PER_CELEBRITY = 1000  # 0 = all

# Logistic Regression
MAX_ITER = 2000
C = 1.5
CLASS_WEIGHT = "balanced"

# -------------------------------------------------------------------
# SVD feature branch
# -------------------------------------------------------------------
SVD_N_COMPONENTS = 512
SVD_RANDOM_SEED = RANDOM_SEED
SVD_C = 1.5
SVD_MAX_ITER = 2000