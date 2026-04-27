import json
import torch
import numpy as np
import os
import re
from transformers import BertModel

# ==========================================
# 1. LOAD LABELS (From your original script)
# ==========================================
job_dict = {'sports': 0, 'creator': 1, 'politics': 2, 'performer': 3}
gender_dict = {'male': 0, 'female': 1}
label_dict = {}

print("Loading labels...")
with open('./traindata/labels.ndjson', encoding='utf-8') as file:
    labels = file.readlines()
    for line in labels:
        celeb = json.loads(line)
        label_dict[str(celeb['id'])] = [celeb['birthyear'], gender_dict[celeb['gender']], job_dict[celeb['occupation']]]

# ==========================================
# 2. GENERATE BERT EMBEDDINGS
# ==========================================
print("Loading tokenized JSON...")
with open('./traindata/train_tokenized.json', 'r', encoding='utf-8') as f:
    tokenized_data = json.load(f)

print("Loading BERT model...")
model = BertModel.from_pretrained('bert-base-uncased')

# --- NEW: Dynamic Model Resizing ---
# Find the absolute maximum token ID in your entire dataset
max_token_id = 0
for item in tokenized_data:
    current_max = max(item['input_ids']) if item['input_ids'] else 0
    if current_max > max_token_id:
        max_token_id = current_max

# If the max token exceeds the default vocab size (30521), resize the model
if max_token_id >= 30522:
    print(f"Detected custom tokens (Max ID: {max_token_id}). Resizing model embeddings...")
    model.resize_token_embeddings(max_token_id + 1)
# -----------------------------------

model.eval()

bert_data_dict = {}

print("Converting tokens to embeddings...")
with torch.no_grad():
    for item in tokenized_data:
        celeb_id = str(item['celebrity_id'])
        tokens = item['input_ids']

        input_ids = torch.tensor([tokens[:512]])
        outputs = model(input_ids)
        cls_embedding = outputs.last_hidden_state[0][0].numpy()

        bert_data_dict[str(celeb_id)] = cls_embedding

print(f"Successfully generated embeddings for {len(bert_data_dict)} celebrities.")

# ==========================================
# 3. FEATURE EXTRACTION FUNCTIONS
# ==========================================
def is_emoji(text):
    for char in text:
        if (0x1F600 <= ord(char) <= 0x1F64F or
                0x1F300 <= ord(char) <= 0x1F5FF or
                0x1F680 <= ord(char) <= 0x1F6FF or
                0x2600 <= ord(char) <= 0x26FF or
                0x2700 <= ord(char) <= 0x27BF or
                0x1F900 <= ord(char) <= 0x1F9FF or
                0x1FA70 <= ord(char) <= 0x1FAFF):
            return True
    return False


def process_celeb_data_with_bert(data, bert_embedding):
    followers = data['text']
    celeb_id = data['id']

    vector_dict = {"PERSON": 0, "NORP": 0, "FAC": 0, "ORG": 0, "GPE": 0, "LOC": 0, "PRODUCT": 0,
                   "EVENT": 0, "WORK_OF_ART": 0, "LAW": 0, "LANGUAGE": 0, "DATE": 0, "TIME": 0,
                   "PERCENT": 0, "MONEY": 0, "QUANTITY": 0, "ORDINAL": 0, "CARDINAL": 0,
                   "ADJ": 0, "ADP": 0, "ADV": 0, "AUX": 0, "CONJ": 0, "CCONJ": 0,
                   "DET": 0, "INTJ": 0, "NOUN": 0, "NUM": 0, "PART": 0, "PRON": 0,
                   "PROPN": 0, "PUNCT": 0, "SCONJ": 0, "SYM": 0, "VERB": 0, "X": 0,
                   "SPACE": 0, "STOP_WORDS": 0, "AVG_TWEET_LEN": 0, "MENTIONS": 0, "HASHTAGS": 0,
                   "LINKS": 0, "EMOJI": 0}

    tweet_length = 0
    count_tweets = 0

    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_pattern = re.compile(r'@\w+')
    hashtag_pattern = re.compile(r'#\w+')

    for follower in followers:
        for tweet in follower:
            tweet_length += len(tweet)
            count_tweets += 1

            vector_dict['LINKS'] += len(url_pattern.findall(tweet))
            vector_dict['MENTIONS'] += len(mention_pattern.findall(tweet))
            vector_dict['HASHTAGS'] += len(hashtag_pattern.findall(tweet))

            for char in tweet:
                if is_emoji(char):
                    vector_dict['EMOJI'] += 1

    if count_tweets > 0:
        vector_dict['AVG_TWEET_LEN'] = tweet_length / count_tweets
        vector_dict['LINKS'] /= count_tweets
        vector_dict['MENTIONS'] /= count_tweets
        vector_dict['HASHTAGS'] /= count_tweets
        vector_dict['EMOJI'] /= count_tweets

    bert_vec_list = bert_embedding.tolist() if isinstance(bert_embedding, np.ndarray) else list(bert_embedding)
    total_list = [celeb_id] + bert_vec_list + list(vector_dict.values())
    return total_list


def process_ndjson_with_bert(input_file, output_dir, bert_data_dict):
    feature_vecs = []
    label_vecs = []

    print(f"Reading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            celeb_id = str(data['id'])

            if celeb_id in label_dict:
                if celeb_id not in bert_data_dict:
                    print(f"Skipping {celeb_id}: No BERT embedding found.")
                    continue

                bert_embedding = bert_data_dict[celeb_id]
                label_vecs.append(label_dict[celeb_id])

                try:
                    print(f"Processing celeb ID: {celeb_id}")
                    vector_list = process_celeb_data_with_bert(data, bert_embedding)
                    feature_vecs.append(vector_list)
                except Exception as e:
                    print(f"Error processing {celeb_id}: {e}")
            else:
                print(f"No label found for {celeb_id}")

    if feature_vecs:
        feature_vec_array = np.array(feature_vecs)
        label_vec_array = np.array(label_vecs)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        np.save(os.path.join(output_dir, 'features.npy'), feature_vec_array)
        np.save(os.path.join(output_dir, 'labels.npy'), label_vec_array)
        print(f"Successfully saved features (Shape: {feature_vec_array.shape}) and labels to {output_dir}")
    else:
        print(f"No features processed")


# ==========================================
# 4. EXECUTE PIPELINE
# ==========================================
process_ndjson_with_bert('./traindata/follower-feeds.ndjson', './Testing/celeb_files/', bert_data_dict)