TARGET_LABEL = "occupation"   # start simple: occupation first

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

RANDOM_SEED = 42
NUM_LABELS = 4   # sports, performer, creator, politics

CHUNK_AGGREGATION_METHOD = "mean_logits"

SAVE_CHECKPOINTS = True
SAVE_PREDICTIONS = True

VAL_RATIO = 0.1
SAVE_TOTAL_LIMIT = 2
MAX_GRAD_NORM = 1.0