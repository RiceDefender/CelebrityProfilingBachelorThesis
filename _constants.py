import os

# root directory
root_dir = os.path.dirname(os.path.abspath(__file__))

# data directory
data_dir = os.path.join(root_dir, "data")

# training dataset
train_path = os.path.join(
    data_dir,
    "pan20-celebrity-profiling-training-dataset-2020-02-28",
    "pan20-celebrity-profiling-training-dataset-2020-02-28"
)
# label training
train_label_path = os.path.join(train_path, "labels.ndjson" )

# supplement dataset
supp_path = os.path.join(
    data_dir,
    "pan20-celebrity-profiling-supplement-dataset-2020-02-28",
    "pan20-celebrity-profiling-supplement-dataset-2020-02-28"
)
# label supplement
supp_label_path = os.path.join(supp_path, "labels.ndjson")

# test dataset
test_path = os.path.join(
    data_dir,
    "pan20-celebrity-profiling-test-dataset-2020-02-28",
    "pan20-celebrity-profiling-test-dataset-2020-02-28"
)
# label test
test_label_path = os.path.join(test_path, "labels.ndjson")

# plot directory
plots_dir = os.path.join(root_dir, "plots")