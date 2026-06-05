# -------------------------------------------------------------------
# BERTweet V3.6 Attention Pooling config
# -------------------------------------------------------------------

MODEL_NAME = "vinai/bertweet-base"
VERSION = "bertweet_v3_6_attention_pooling"

TARGET_LABEL = "occupation"

RANDOM_SEED = 42

# Training
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0

USE_FP16 = True

# Chunks per celebrity
MAX_TRAIN_CHUNKS_PER_CELEB = 32
MAX_VAL_CHUNKS_PER_CELEB = 64
MAX_PREDICT_CHUNKS_PER_CELEB = 128

# Attention regularization
ATTENTION_DROPOUT = 0.10
CLASSIFIER_DROPOUT = 0.20

# Chunk dropout during training.
# This approximates tweet/follower dropout on already-tokenized V3.5 chunks.
CHUNK_DROPOUT_MODE = "none"
# options:
# "none"
# "random"
# "class_specific"

RANDOM_CHUNK_DROPOUT_RATE = 0.15

CLASS_SPECIFIC_CHUNK_DROPOUT = {
    "sports": 0.05,
    "politics": 0.10,
    "performer": 0.20,
    "creator": 0.25,
}

MIN_CHUNKS_AFTER_DROPOUT = 8

PREDICT_BATCH_SIZE = 1
VOTING_STRATEGY = "attention_pooling"

LABEL_ORDERS = {
    "occupation": ["sports", "performer", "creator", "politics"],
    "gender": ["male", "female"],
    "birthyear": ["1994", "1985", "1975", "1963", "1947"],
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
}