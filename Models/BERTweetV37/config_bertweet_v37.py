# -------------------------------------------------------------------
# BERTweet V3.7 config
# Hierarchical occupation experiment:
#   Stage 1: sports / entertainment / politics
#   Stage 2: creator / performer
# -------------------------------------------------------------------

MODEL_NAME = "vinai/bertweet-base"

# Default target when no --target is passed.
TARGET_LABEL = "v37_occupation"

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

# BERTweet was tokenized with max sequence length 128 in the previous V3 pipeline.
# Keep the same chunk budget for a fair first comparison.
MAX_TRAIN_CHUNKS_PER_CELEB = 32
MAX_VAL_CHUNKS_PER_CELEB = 64

# Aggregation after validation
VOTING_STRATEGY = "soft"  # "soft" = mean probabilities across chunks

# Labels
LABEL_ORDERS = {
    # Direct baseline target, kept for compatibility / optional ablation.
    "occupation": ["sports", "performer", "creator", "politics"],

    # V3.7 Stage 1:
    # creator + performer are collapsed to entertainment.
    "occupation_group3": ["sports", "entertainment", "politics"],

    # V3.7 Stage 2:
    # trained only on rows where original occupation is creator or performer.
    "creator_performer": ["creator", "performer"],
}

# Target shortcut
V37_TARGETS = ["occupation_group3", "creator_performer"]

# Class weights
# occupation_group3 is imbalanced after mapping:
# sports=480, politics=480, entertainment=960 on the original PAN train split.
# Therefore balanced weighting is useful for the first proof-of-concept.
CLASS_WEIGHT_BY_TARGET = {
    "occupation": None,
    "occupation_group3": "balanced",
    "creator_performer": None,
}
