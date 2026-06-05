import argparse
import joblib
import ndjson
import data_splitter
from numpy import load, delete
import os

# Monkey patch to fix old joblib files containing references to sklearn.linear_model.logistic
import sys
import sklearn.linear_model._logistic as _logistic

sys.modules['sklearn.linear_model.logistic'] = _logistic

# Models are in the root directory, so these paths are perfect when running from root!
occupation_model = joblib.load("trained_models/occupation_model.joblib")
gender_model = joblib.load("trained_models/gender_model.joblib")
birth_year_model = joblib.load("trained_models/birth_year_model.joblib")
scaler = joblib.load("trained_models/bert_scaler.joblib")


def predict_classes(initial_dir, input_dir, output_dir):
    # 1. Separate files
    data_splitter.split_data(initial_dir + '/follower-feeds.ndjson', input_dir)

    # 2. Load extracted features (Fixed path to match your screenshot!)
    test_features = load(input_dir + '/features.npy')

    scaled_test_ids = []
    occupation_dict = {0: 'sports', 1: 'creator', 2: 'politics', 3: 'performer'}
    gender_dict = {0: 'male', 1: 'female'}

    # Separate ids from features
    for val in test_features:
        scaled_test_ids.append(val[0])

    # Delete ID column and scale
    test_features = delete(test_features, 0, 1)
    scaled_test_features = scaler.transform(test_features)

    final_prediction_list = []

    occupation_predictions = occupation_model.predict(scaled_test_features)
    gender_predictions = gender_model.predict(scaled_test_features)
    by_predictions = birth_year_model.predict(scaled_test_features)

    # Construct output file
    for i in range(0, len(occupation_predictions)):
        occupation_label = occupation_dict[occupation_predictions[i]]
        gender_label = gender_dict[gender_predictions[i]]
        by_label = int(by_predictions[i])
        celeb_prediction = {"id": int(scaled_test_ids[i]), "occupation": occupation_label, "gender": gender_label, \
                            "birthyear": by_label}
        final_prediction_list.append(celeb_prediction)

    # Safely ensure output directory exists, then save
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir + '/labels.ndjson', 'w+') as file:
        writer = ndjson.writer(file, ensure_ascii=False)
        for prediction in final_prediction_list:
            writer.writerow(prediction)


def parse_command_line():
    description = 'predict classes from data'
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument('initial_dir', metavar='initial_dir', type=str,
                           help='Filepath of directory containing features to process')
    argparser.add_argument('output_dir', metavar='output_dir', type=str,
                           help='File name for output file containing predicted classes for each celebrity')
    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_command_line()
    # Fixed the hardcoded directory to point to the Testing folder
    predict_classes(args.initial_dir, "Testing/celeb_files", args.output_dir)