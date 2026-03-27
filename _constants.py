import os

# root directory
root_dir = os.path.dirname(os.path.abspath(__file__))

# data directory
data_dir = os.path.join(root_dir, "data")
label_name = "labels.ndjson"
follower_feeds_name = "follower-feeds.ndjson"

# training dataset
train_path = os.path.join(
    data_dir,
    "pan20-celebrity-profiling-training-dataset-2020-02-28",
    "pan20-celebrity-profiling-training-dataset-2020-02-28"
)
# label training
train_label_path = os.path.join(train_path, label_name)
# follower feeds training
train_feeds_path = os.path.join(train_path, follower_feeds_name)

# supplement dataset
supp_path = os.path.join(
    data_dir,
    "pan20-celebrity-profiling-supplement-dataset-2020-02-28",
    "pan20-celebrity-profiling-supplement-dataset-2020-02-28"
)
# label supplement
supp_label_path = os.path.join(supp_path, label_name)
# follower feeds supplement
supp_feeds_path = os.path.join(supp_path, follower_feeds_name)

# test dataset
test_path = os.path.join(
    data_dir,
    "pan20-celebrity-profiling-test-dataset-2020-02-28",
    "pan20-celebrity-profiling-test-dataset-2020-02-28"
)
# label test
test_label_path = os.path.join(test_path, label_name)
# follower feeds test
test_feeds_path = os.path.join(test_path, follower_feeds_name)

# plot directory
plots_dir = os.path.join(root_dir, "plots")
