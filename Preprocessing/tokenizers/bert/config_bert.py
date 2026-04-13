MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 512

MAX_FOLLOWERS = 3
MAX_TWEETS_PER_FOLLOWER = 20
MAX_CHARS = 10000

INCLUDE_SUPP = False

OUTPUT_DIRNAME = "bert_tokenized_chunked" # evtl. OUTPUT_DIRNAME = f"bert_tokenized_chunked_{TWEETS_PER_CHUNK}" (Parameterize den Ordnernamen mit der Anzahl der Tweets pro Chunk)

URL_TOKEN = "[URL]"
MENTION_TOKEN = "[MENTION]"

SAVE_PRETTY_JSON = False

USE_CHUNKING = True
TWEETS_PER_CHUNK = 12