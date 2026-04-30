# -------------------------------------------------------------------
# BERTweet V3 model config
# -------------------------------------------------------------------

MODEL_NAME = "vinai/bertweet-base"

TARGET_LABEL = "occupation"

RANDOM_SEED = 42
VAL_RATIO = 0.1

# Training
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# Mixed precision
USE_FP16 = True

MAX_TRAIN_CHUNKS_PER_CELEB = 32
MAX_VAL_CHUNKS_PER_CELEB = 64

# Aggregation after validation
VOTING_STRATEGY = "soft"  # "soft" = mean probabilities

# Labels
LABEL_ORDERS = {
    "occupation": ["sports", "performer", "creator", "politics"],
    "gender": ["male", "female"],
    "birthyear": ["1994", "1985", "1975", "1963", "1947"],
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
}