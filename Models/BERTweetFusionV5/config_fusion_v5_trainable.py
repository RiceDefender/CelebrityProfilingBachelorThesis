# -------------------------------------------------------------------
# BERTweetFusionV5 trainable fusion config
# -------------------------------------------------------------------

import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

LABELS = ["sports", "performer", "creator", "politics"]
TARGET = "occupation"

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "bertweet_fusion_v5")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Train/selection split for the fusion layer.
# This should usually be fusion_val predictions, not the official test set.
FUSION_TRAIN_SPLIT = "fusion_val"
EVAL_SPLIT = "test"

# Candidate prediction paths by split and model.
# The script will use the first existing path for each model/split.
# Add/adjust paths here if your filenames differ.
PREDICTION_PATHS = {
    "fusion_val": {
        "v3": [
            os.path.join(PROJECT_ROOT, "outputs", "hybrid_v4", "bertweet_probs", "occupation_fusion_val_bertweet_v3_probs.json"),
            os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3", "predictions", "occupation_fusion_val_predictions.json"),
            os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3", "predictions", "occupation_val_predictions.json"),
        ],
        "v34": [
            os.path.join(PROJECT_ROOT, "outputs", "hybrid_v4", "bertweet_probs", "occupation_fusion_val_bertweet_v34_probs.json"),
            os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3_4_stopwords", "predictions", "occupation_fusion_val_predictions.json"),
            os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3_4_stopwords", "predictions", "occupation_val_predictions.json"),
        ],
        "v35": [
            os.path.join(PROJECT_ROOT, "outputs", "hybrid_v4", "bertweet_probs", "occupation_fusion_val_bertweet_v35_probs.json"),
            os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3_5_controlled_sampling", "predictions", "occupation_fusion_val_predictions.json"),
            os.path.join(PROJECT_ROOT, "outputs", "bertweet_v35", "predictions", "occupation_fusion_val_predictions.json"),
        ],
        "v36": [
            os.path.join(PROJECT_ROOT, "outputs", "hybrid_v4", "bertweet_probs", "occupation_fusion_val_bertweet_v36_probs.json"),
            os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3_6_attention_pooling", "predictions", "occupation_fusion_val_predictions.json"),
        ],
        # "v37": [
        #     os.path.join(PROJECT_ROOT, "outputs", "hybrid_v4", "bertweet_probs", "occupation_fusion_val_bertweet_v37_probs.json"),
        #     os.path.join(PROJECT_ROOT, "outputs", "bertweet_v37", "predictions", "occupation_fusion_val_predictions.json"),
        #     os.path.join(PROJECT_ROOT, "outputs", "bertweet_v37", "predictions", "occupation_val_predictions.json"),
        # ],
    },
    "test": {
        "v3": [
            os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3", "predictions", "occupation_test_predictions.json"),
        ],
        "v34": [
            os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3_4_stopwords", "predictions", "occupation_test_predictions.json"),
            os.path.join(PROJECT_ROOT, "outputs", "bertweet_v34", "predictions", "occupation_test_predictions.json"),
        ],
        "v35": [
            os.path.join(PROJECT_ROOT, "outputs", "hybrid_v4", "bertweet_probs", "occupation_test_bertweet_v35_probs.json"),
            os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3_5_controlled_sampling", "predictions", "occupation_test_predictions.json"),
            os.path.join(PROJECT_ROOT, "outputs", "bertweet_v35", "predictions", "occupation_test_predictions.json"),
        ],
        "v36": [
            os.path.join(PROJECT_ROOT, "outputs", "hybrid_v4", "bertweet_probs", "occupation_test_bertweet_v36_probs.json"),
            os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3_6_attention_pooling", "predictions", "occupation_test_predictions.json"),
        ],
        "v37": [
            os.path.join(PROJECT_ROOT, "outputs", "bertweet_v37", "predictions", "occupation_test_predictions.json"),
        ],
    },
}

# Grid-search step for weighted mean. Smaller = faster, larger = finer.
GRID_STEP = 0.05

# Meta model settings
LOGREG_MAX_ITER = 5000
LOGREG_C_VALUES = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
RANDOM_SEED = 42
