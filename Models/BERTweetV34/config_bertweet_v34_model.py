# -------------------------------------------------------------------
# BERTweet V3.4 stopword-filtered model config
# -------------------------------------------------------------------

MODEL_NAME = "vinai/bertweet-base"
VERSION = "bertweet_v3_4_stopwords"

TARGET_LABEL = "occupation"

RANDOM_SEED = 42
VAL_RATIO = 0.1

# Training: keep identical to V3 for a fair tokenizer ablation.
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# Mixed precision
USE_FP16 = True

# Keep identical to V3 unless explicitly changed.
MAX_TRAIN_CHUNKS_PER_CELEB = 32
MAX_VAL_CHUNKS_PER_CELEB = 64
MAX_PREDICT_CHUNKS_PER_CELEB = 128
PREDICT_BATCH_SIZE = 32

# Aggregation after validation/test
VOTING_STRATEGY = "soft"  # "soft" = mean probabilities

# Labels
# birthyear = old V3/PAN-style 5-centroid task.
# birthyear_8range = Koloski-inspired train-quantile age-range task.
LABEL_ORDERS = {
    "occupation": ["sports", "performer", "creator", "politics"],
    "gender": ["male", "female"],
    "birthyear": ["1994", "1985", "1975", "1963", "1947"],
    "birthyear_8range": [
        "age_bin_0", "age_bin_1", "age_bin_2", "age_bin_3",
        "age_bin_4", "age_bin_5", "age_bin_6", "age_bin_7",
    ],
    "creator_binary": ["not_creator", "creator"],
    "occupation_3class": ["sports", "performer", "politics"],
}

# Class weights
CLASS_WEIGHT_BY_TARGET = {
    "occupation": None,
    "gender": "balanced",
    "birthyear": {
        "1994": 1.5,
        "1985": 0.9,
        "1975": 0.95,
        "1963": 1.0,
        "1947": 1.5,
    },
    # Quantile bins should already be more balanced, but balanced is still safe
    # if duplicate years at bin edges create uneven groups.
    "birthyear_8range": "balanced",
    "creator_binary": "balanced",
    "occupation_3class": None,
}
