import re

URL_TOKEN = "[URL]"
MENTION_TOKEN = "[MENTION]"
TWEET_SEP = "[TWEET_SEP]"
FOLLOWER_SEP = "[FOLLOWER_SEP]"

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_tweet(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = URL_RE.sub(URL_TOKEN, text)
    text = MENTION_RE.sub(MENTION_TOKEN, text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text