import os

# -------------------------------------------------------------------
# Root
# -------------------------------------------------------------------
root_dir = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------------
# Raw PAN data
# -------------------------------------------------------------------
data_dir = os.path.join(root_dir, "data")
label_name = "labels.ndjson"
follower_feeds_name = "follower-feeds.ndjson"

# training dataset
train_path = os.path.join(
    data_dir,
    "pan20-celebrity-profiling-training-dataset-2020-02-28",
    "pan20-celebrity-profiling-training-dataset-2020-02-28"
)
train_label_path = os.path.join(train_path, label_name)
train_feeds_path = os.path.join(train_path, follower_feeds_name)

# supplement dataset
supp_path = os.path.join(
    data_dir,
    "pan20-celebrity-profiling-supplement-dataset-2020-02-28",
    "pan20-celebrity-profiling-supplement-dataset-2020-02-28"
)
supp_label_path = os.path.join(supp_path, label_name)
supp_feeds_path = os.path.join(supp_path, follower_feeds_name)

# test dataset
test_path = os.path.join(
    data_dir,
    "pan20-celebrity-profiling-test-dataset-2020-02-28",
    "pan20-celebrity-profiling-test-dataset-2020-02-28"
)
test_label_path = os.path.join(test_path, label_name)
test_feeds_path = os.path.join(test_path, follower_feeds_name)

# -------------------------------------------------------------------
# Existing analysis / comparison
# -------------------------------------------------------------------
plots_dir = os.path.join(root_dir, "plots")

price_hodge_path = os.path.join(data_dir, "price_hodge")
price_hodge_label_path = os.path.join(price_hodge_path, label_name)
price_hodge_diff_path = os.path.join(price_hodge_path, "diff.txt")

comparison_dir = os.path.join(root_dir, "comparison")
comparison_plots_dir = os.path.join(comparison_dir, "plots")
comparison_tables_dir = os.path.join(comparison_dir, "tables")

# -------------------------------------------------------------------
# Preprocessing directories
# -------------------------------------------------------------------
preprocessing_dir = os.path.join(root_dir, "Preprocessing")
preprocessing_data_dir = os.path.join(preprocessing_dir, "data")

tokenizers_dir = os.path.join(preprocessing_dir, "tokenizers")
bert_tokenizer_dir = os.path.join(tokenizers_dir, "bert")

# tokenized BERT outputs stay inside Preprocessing/data
bert_processed_dir = os.path.join(preprocessing_data_dir, "bert_tokenized_chunked")
bert_train_processed_dir = bert_processed_dir
bert_test_processed_dir = bert_processed_dir
bert_supp_processed_dir = bert_processed_dir

bert_train_tokenized_path = os.path.join(bert_train_processed_dir, "train_tokenized.json")
bert_test_tokenized_path = os.path.join(bert_test_processed_dir, "test_tokenized.json")
bert_supp_tokenized_path = os.path.join(bert_supp_processed_dir, "supp_tokenized.json")

# optional future metadata / debug files
bert_train_meta_path = os.path.join(bert_train_processed_dir, "train_meta.json")
bert_test_meta_path = os.path.join(bert_test_processed_dir, "test_meta.json")
bert_supp_meta_path = os.path.join(bert_supp_processed_dir, "supp_meta.json")

# -------------------------------------------------------------------
# Model / evaluation directories
# -------------------------------------------------------------------
models_dir = os.path.join(root_dir, "Models")
bert_model_dir = os.path.join(models_dir, "BERT")

evaluation_dir = os.path.join(root_dir, "Evaluation")

# -------------------------------------------------------------------
# Training / inference outputs
# -------------------------------------------------------------------
outputs_dir = os.path.join(root_dir, "outputs")
bert_output_dir = os.path.join(outputs_dir, "bert_mvp")
bert_checkpoints_dir = os.path.join(bert_output_dir, "checkpoints")
bert_logs_dir = os.path.join(bert_output_dir, "logs")
bert_predictions_dir = os.path.join(bert_output_dir, "predictions")
bert_metrics_dir = os.path.join(bert_output_dir, "metrics")

# -------------------------------------------------------------------
# SBERT preprocessing directories
# -------------------------------------------------------------------
sbert_vectorizer_dir = os.path.join(preprocessing_dir, "vectorizers", "sbert")

sbert_processed_dir = os.path.join(preprocessing_data_dir, "sbert_vectors_chunked")
sbert_train_processed_dir = sbert_processed_dir
sbert_test_processed_dir = sbert_processed_dir
sbert_supp_processed_dir = sbert_processed_dir

sbert_train_vectors_path = os.path.join(sbert_train_processed_dir, "train_vectors.json")
sbert_test_vectors_path = os.path.join(sbert_test_processed_dir, "test_vectors.json")
sbert_supp_vectors_path = os.path.join(sbert_supp_processed_dir, "supp_vectors.json")

sbert_train_meta_path = os.path.join(sbert_train_processed_dir, "train_meta.json")
sbert_test_meta_path = os.path.join(sbert_test_processed_dir, "test_meta.json")
sbert_supp_meta_path = os.path.join(sbert_supp_processed_dir, "supp_meta.json")

# -------------------------------------------------------------------
# SBERT model / outputs
# -------------------------------------------------------------------
sbert_model_dir = os.path.join(models_dir, "SBERT")

sbert_output_dir = os.path.join(outputs_dir, "sbert_mvp")
sbert_checkpoints_dir = os.path.join(sbert_output_dir, "checkpoints")
sbert_logs_dir = os.path.join(sbert_output_dir, "logs")
sbert_predictions_dir = os.path.join(sbert_output_dir, "predictions")
sbert_metrics_dir = os.path.join(sbert_output_dir, "metrics")

# -------------------------------------------------------------------
# SBERT V2 model / outputs
# Logistic Regression + Chunk Voting
# -------------------------------------------------------------------
sbert_v2_output_dir = os.path.join(outputs_dir, "sbert_v2")
sbert_v2_checkpoints_dir = os.path.join(sbert_v2_output_dir, "checkpoints")
sbert_v2_predictions_dir = os.path.join(sbert_v2_output_dir, "predictions")
sbert_v2_metrics_dir = os.path.join(sbert_v2_output_dir, "metrics")
sbert_v2_test_metrics_dir = os.path.join(sbert_v2_output_dir, "test_metrics")