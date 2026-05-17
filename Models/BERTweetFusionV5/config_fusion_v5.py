# -------------------------------------------------------------------
# BERTweetFusionV5 config
# Controlled Confidence Gating for occupation
# -------------------------------------------------------------------

import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

TARGET = "occupation"
LABELS = ["sports", "performer", "creator", "politics"]

# Output directory for this fusion layer
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "bertweet_fusion_v5")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# -------------------------------------------------------------------
# Candidate prediction paths
# The script will try these in order. Adjust only if your filenames differ.
# -------------------------------------------------------------------

PREDICTION_PATHS = {
    # 4-class base models
    "v3_occupation": [
        os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3", "predictions", "occupation_test_predictions.json"),
    ],
    "v34_occupation": [
        os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3_4_stopwords", "predictions", "occupation_test_predictions.json"),
        os.path.join(PROJECT_ROOT, "outputs", "bertweet_v34", "predictions", "occupation_test_predictions.json"),
    ],
    "v35_occupation": [
        os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3_5_controlled_sampling", "predictions", "occupation_test_predictions.json"),
        os.path.join(PROJECT_ROOT, "outputs", "bertweet_v35", "predictions", "occupation_test_predictions.json"),
    ],
    "v36_occupation": [
        os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3_6_attention_pooling", "predictions", "occupation_test_predictions.json"),
        os.path.join(PROJECT_ROOT, "outputs", "bertweet_v36", "predictions", "occupation_test_predictions.json"),
    ],
    "v37_occupation": [
        os.path.join(PROJECT_ROOT, "outputs", "bertweet_v37", "predictions", "occupation_test_predictions.json"),
        os.path.join(PROJECT_ROOT, "outputs", "bertweet_v37", "predictions", "occupation_val_predictions.json"),
    ],

    # Auxiliary models
    "v3_creator_binary": [
        os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3", "predictions", "creator_binary_test_predictions.json"),
    ],
    "v3_occupation_3class": [
        os.path.join(PROJECT_ROOT, "outputs", "bertweet_v3", "predictions", "occupation_3class_test_predictions.json"),
    ],
    "v37_group3": [
        os.path.join(PROJECT_ROOT, "outputs", "bertweet_v37", "predictions", "occupation_group3_test_predictions.json"),
    ],
    "v37_creator_performer": [
        os.path.join(PROJECT_ROOT, "outputs", "bertweet_v37", "predictions", "creator_performer_test_predictions.json"),
    ],
}

# -------------------------------------------------------------------
# Base model weights
# Missing models are skipped and the remaining weights are normalized.
# Start conservative; later you can grid-search these on fusion_val.
# -------------------------------------------------------------------

BASE_MODEL_WEIGHTS = {
    "v3_occupation": 0.20,
    "v34_occupation": 0.15,
    "v35_occupation": 0.20,
    "v36_occupation": 0.20,
    "v37_occupation": 0.25,
}

# -------------------------------------------------------------------
# Gating D thresholds
# These are deliberately conservative because the goal is not to create
# many overrides, but to correct known high-value error patterns.
# -------------------------------------------------------------------

GATING_THRESHOLDS = {
    # Creator rescue from multiple independent signals
    "creator_binary_min": 0.50,
    "base_creator_min": 0.25,
    "any_4class_creator_min": 0.35,
    "v37_entertainment_min_for_creator_signal": 0.50,
    "v37_cp_creator_min_for_creator_signal": 0.55,
    "creator_rescue_votes_min": 3,

    # Sports/politics rescue from performer absorption
    "rescue_base_label": "performer",
    "sports_group3_min": 0.45,
    "politics_group3_min": 0.45,
    "sports_base_min": 0.30,
    "politics_base_min": 0.30,
    "rescue_votes_min": 3,

    # V3.7 entertainment confidence gate, creator-only override
    "entertainment_min": 0.55,
    "entertainment_margin_min": 0.10,
    "cp_creator_min": 0.55,
    "cp_margin_min": 0.10,
}

# If true, saves a detailed per-celebrity decision trace.
SAVE_DECISION_TRACE = True
