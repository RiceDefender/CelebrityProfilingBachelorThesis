# -------------------------------------------------------------------
# BERTweet V3.5 controlled-sampling model config
# -------------------------------------------------------------------
MODEL_NAME = "vinai/bertweet-base"
VERSION = "bertweet_v3_5_controlled_sampling"

TARGET_LABEL = "occupation"

RANDOM_SEED = 42
VAL_RATIO = 0.1

# Keep comparable to BERTweet V3/V3.4
NUM_EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

USE_FP16 = True

# Same chunk policy as V3.4 fusion experiments
MAX_TRAIN_CHUNKS_PER_CELEB = 32
MAX_VAL_CHUNKS_PER_CELEB = 64
MAX_PREDICT_CHUNKS_PER_CELEB = 128
PREDICT_BATCH_SIZE = 32

VOTING_STRATEGY = "soft"

LABEL_ORDERS = {
    "occupation": ["sports", "performer", "creator", "politics"],
    "gender": ["male", "female"],
    "birthyear": ["1994", "1985", "1975", "1963", "1947"],
    "creator_binary": ["not_creator", "creator"],
    "occupation_3class": ["sports", "performer", "politics"],
}

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
    "creator_binary": "balanced",
    "occupation_3class": None,
}