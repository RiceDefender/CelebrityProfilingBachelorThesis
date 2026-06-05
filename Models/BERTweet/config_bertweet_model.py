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

    # V3.1 auxiliary occupation models
    "creator_binary": ["not_creator", "creator"],
    "occupation_3class": ["sports", "performer", "politics"],
}

# -------------------------------------------------------------------
# BERTweet V3.1 occupation-gated settings
# -------------------------------------------------------------------

V31_TARGETS = ["creator_binary", "occupation_3class"]

V31_LOW_THRESHOLD = 0.35
V31_HIGH_THRESHOLD = 0.60

V31_THRESHOLD_GRID_LOW = [0.20, 0.25, 0.30, 0.35, 0.40]
V31_THRESHOLD_GRID_HIGH = [0.50, 0.55, 0.60, 0.65, 0.70]

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

    # V3.1
    "creator_binary": "balanced",
    "occupation_3class": None,
}