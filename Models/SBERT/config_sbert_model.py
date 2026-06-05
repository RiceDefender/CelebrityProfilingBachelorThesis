TARGET_LABEL = "occupation"

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS=100     # NUM_EPOCHS = 10000 in MVP
WEIGHT_DECAY = 0.01

RANDOM_SEED = 42
VAL_RATIO = 0.1

CLASSIFIER_TYPE = "mlp"   # "linear" oder "mlp"
HIDDEN_DIM = 512
DROPOUT = 0.3

POOLING_STRATEGY = "mean"
NORMALIZE_INPUTS = True

SAVE_CHECKPOINTS = True
SAVE_PREDICTIONS = True

EARLY_STOPPING_ENABLED = False
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 1e-4

# -------------------------------------------------------------------
# SBERT V2
# -------------------------------------------------------------------
V2_CLASSIFIER_TYPE = "logistic_regression"

V2_MAX_ITER = 5000
V2_C = 1.0
V2_SOLVER = "lbfgs"
V2_CLASS_WEIGHT_BY_TARGET = {
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

V2_VOTING_STRATEGY = "soft"  # "soft" oder "hard"
V2_NORMALIZE_INPUTS = True

# -------------------------------------------------------------------
# SBERT V2.1 Rebalancing
# -------------------------------------------------------------------
# V2_CLASS_WEIGHT_BY_TARGET = {
#     "occupation": {
#         "sports": 1.0,
#         "performer": 1.0,
#         "creator": 1.15,
#         "politics": 1.0,
#     },
#     "gender": "balanced",
#     "birthyear": "balanced",
# }

V2_USE_SAMPLE_WEIGHTS = True