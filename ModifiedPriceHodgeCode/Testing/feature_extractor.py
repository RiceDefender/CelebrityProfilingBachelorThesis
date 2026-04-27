import spacy
import json
import numpy as np
import os
import argparse
from emoji import UNICODE_EMOJI

nlp = spacy.load("en_core_web_lg", disable=['parser'])

def process_file(file_name):
    json_file = open(file_name)
    data = json.load(json_file)

    follower_strings = []
    followers = data['text']
    celeb_id = data['id']
    print(celeb_id)
    vector_dict = {"PERSON":0, "NORP":0, "FAC":0, "ORG":0, "GPE":0, "LOC":0,"PRODUCT":0,
                   "EVENT":0,"WORK_OF_ART":0,"LAW":0,"LANGUAGE":0,"DATE":0,"TIME":0,
                   "PERCENT":0,"MONEY":0,"QUANTITY":0,"ORDINAL":0,"CARDINAL":0,
                   "ADJ": 0, "ADP": 0, "ADV": 0, "AUX": 0, "CONJ": 0, "CCONJ": 0,
                  "DET": 0, "INTJ": 0, "NOUN": 0, "NUM": 0, "PART": 0, "PRON": 0,
                  "PROPN": 0, "PUNCT": 0, "SCONJ": 0, "SYM": 0, "VERB": 0, "X": 0,
                   "SPACE": 0, "STOP_WORDS": 0, "AVG_TWEET_LEN": 0, "MENTIONS": 0, "HASHTAGS": 0,
                  "LINKS": 0, "EMOJI": 0}

    tweet_length = 0
    count_tweets = 0
    word_vecs = []
    num_tokens = 0

    for follower in followers:
        for tweet in follower:
            tweet_length += len(tweet)
            count_tweets += 1
        follower_text = " ".join(follower)
        tokens = follower_text.split()
        num_tokens += len(tokens)
        text = []
        for token in tokens:
            if token.startswith('#') and len(token) > 1:
                text.append(token[1:])
                vector_dict['HASHTAGS'] += 1
            elif token.startswith('@') and len(token) > 1:
                vector_dict['MENTIONS'] += 1
            elif token.startswith('http'):
                vector_dict['LINKS'] += 1
            elif len(token) == 1 and token in UNICODE_EMOJI:
                vector_dict['EMOJI'] += 1
            else:
                text.append(token)
        ftext = " ".join(text)
        doc = nlp(ftext)
        entities = []
        clean_text = []
        for entity in doc.ents:
            vector_dict[entity.label_] += 1
        for token in doc:
            vector_dict[token.pos_] += 1
            if token.is_stop:
                vector_dict['STOP_WORDS'] += 1
            if token.has_vector:
                word_vecs.append(token.vector)
    vector_dict = dict(map(lambda feature: (feature[0], feature[1] / num_tokens), vector_dict.items()))
    vector_dict['AVG_TWEET_LEN'] = tweet_length / count_tweets
    word_vec_array = np.array(word_vecs)
    wv = np.mean(word_vec_array, axis=0)
    word_vec_list = wv.tolist()
    total_list = [celeb_id] + word_vec_list + list(vector_dict.values())
    return total_list


def process_dir(dir_path, output_dir):
    feature_vecs = []
    for file in os.listdir('./' + dir_path):
        print(file)
        if file != '.DS_Store' and file != '.ipynb_checkpoints':
            vector_list = process_file(dir_path +'/'+ file)
            feature_vecs.append(vector_list)
    feature_vec_array = np.array(feature_vecs)
    np.save(dir_path+'_features.npy', feature_vec_array)

    # with open(dir_path + '/follower-feeds.ndjson') as follower_file:
    #     feature_vecs = []
    #     data = ndjson.reader(follower_file)
    #     count = 0
    #     for line in data:
    #         count += 1
    #         if count == 2:
    #             count = 0
    #             feature_vec_array = np.array(feature_vecs)
    #             predict_classes(feature_vec_array, output_dir)
    #             feature_vecs = []
    #         else:
    #             feature_vecs.append(vector_list)
    #         vector_list = process_file(line)
    #         feature_vec_array = np.array(vector_list)
            
            # feature_vecs.append(vector_list)
        # follower_file.close()

# def predict_classes(test_features, output_dir):
#     print(test_features[0])
#     scaled_test_id = test_features[0]

#     test_features = test_features[1:]
#     print(test_features[0])
#     print(len(test_features))
#     test_features = test_features.reshape(1, -1)
#     scaled_test_features = scale(test_features)

#     occupation_prediction = occupation_model.predict(scaled_test_features)
#     gender_prediction = gender_model.predict(scaled_test_features)
#     by_prediction = birth_year_model.predict(scaled_test_features)

#     occupation_label = occupation_dict[occupation_prediction[0]]
#     gender_label = gender_dict[gender_prediction[0]]
#     by_label = int(by_prediction[0])
#     celeb_prediction = {"id": int(scaled_test_id), "occupation": occupation_label, "gender": gender_label, \
#                         "birthyear": by_label}
                
#     with open(output_dir + '/labels.ndjson', 'a') as file:
#         writer = ndjson.writer(file, ensure_ascii=False)
#         writer.writerow(celeb_prediction)
#         file.close()

# # Function to scale input vectors
# def scale(input_vector):
#     scaler = StandardScaler()
#     scaler.fit(input_vector)
#     scaled_features = scaler.transform(input_vector)
#     return scaled_features


def parse_command_line():
    description = 'feature extractor'
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument('input_dir', metavar='input_dir', type=str,
                           help='Filepath of directory containing features to process')
    argparser.add_argument('output_dir', metavar='output_dir', type=str,
                           help='File name for output file containing predicted classes for each celebrity')
    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_command_line()
    process_dir(args.input_dir, args.output_dir)