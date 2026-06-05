import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.util import ngrams
from _constants import test_feeds_path, bertweet_v3_predictions_dir

# Pfad zur Vorhersagedatei
PRED_PATH = os.path.join(bertweet_v3_predictions_dir, "occupation_test_predictions.json")

# Erweiterte Stoppwortliste, um das "Rauschen" im Kontext zu minimieren
STOP_WORDS = {
    'this', 'that', 'with', 'your', 'have', 'just', 'what', 'from', 'they', 'about',
    'will', 'more', 'dont', 'when', 'people', 'know', 'good', 'been', 'there', 'were',
    'would', 'their', 'them', 'then', 'some', 'only', 'after', 'even', 'than', 'into'
}


def clean_and_tokenize(text):
    text = str(text).lower()
    # Entferne URLs, Mentions und Sonderzeichen
    text = re.sub(r'http\S+|@\w+|[^a-z\s]', '', text)
    tokens = [w for w in text.split() if len(w) > 2 and w not in STOP_WORDS]
    return tokens


def get_bigrams(texts):
    """Erzeugt Wortpaare aus den Texten eines Feeds."""
    phrases = []
    for t in texts:
        tokens = clean_and_tokenize(t)
        if len(tokens) >= 2:
            # Erstellt Paare wie ('new', 'video') -> 'new video'
            phrases.extend([" ".join(gram) for gram in ngrams(tokens, 2)])
    return phrases

def get_trigrams(texts):
    """Erzeugt Trigramme (3er Gruppen) für tieferen Kontext."""
    phrases = []
    for t in texts:
        tokens = clean_and_tokenize(t)
        if len(tokens) >= 3:
            # Erzeugt "check out my", "new video out", etc.
            phrases.extend([" ".join(gram) for gram in ngrams(tokens, 3)])
    return phrases

# Im Plot-Teil einfach die Funktion tauschen:
# trigram_pool.extend(get_trigrams(feeds_content[cid]))

def get_ngramms(texts, n=2):
    """Generische Funktion für n-Gramme."""
    phrases = []
    for t in texts:
        tokens = clean_and_tokenize(t)
        if len(tokens) >= n:
            phrases.extend([" ".join(gram) for gram in ngrams(tokens, n)])
    return phrases


def main():
    # 1. Predictions laden
    with open(PRED_PATH, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    # Metrik-Logik wie besprochen
    get_diff = lambda p: p['probabilities'][1] - p['probabilities'][2]

    categories = {
        "Correct Creator (High)": {
            "filter": lambda x: x['true_label'] == 'creator' and x['pred_label'] == 'creator',
            "sort_key": lambda x: x['probabilities'][2],
            "reverse": True
        },
        "Correct Performer (High)": {
            "filter": lambda x: x['true_label'] == 'performer' and x['pred_label'] == 'performer',
            "sort_key": lambda x: x['probabilities'][1],
            "reverse": True
        },
        "False Performer (High Conf)": {
            "filter": lambda x: x['true_label'] == 'creator' and x['pred_label'] == 'performer',
            "sort_key": get_diff,
            "reverse": True
        },
        "False Performer (Low Conf)": {
            "filter": lambda x: x['true_label'] == 'creator' and x['pred_label'] == 'performer',
            "sort_key": lambda x: abs(get_diff(x)),
            "reverse": False
        }
    }

    selected_ids_map = {}
    all_target_ids = set()

    for cat_name, config in categories.items():
        matches = [p for p in preds if config['filter'](p)]
        # Wir nehmen die Top 10 basierend auf deiner Metrik-Vorgabe
        top_matches = sorted(matches, key=config['sort_key'], reverse=config['reverse'])[:10]
        ids = [str(m['celebrity_id']) for m in top_matches]
        selected_ids_map[cat_name] = ids
        all_target_ids.update(ids)

    # 2. Feeds laden
    feeds_content = {}
    if os.path.exists(test_feeds_path):
        with open(test_feeds_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                cid = str(obj.get('id'))
                if cid in all_target_ids:
                    raw_texts = obj.get('text', [])
                    texts = [t.get('text', '') if isinstance(t, dict) else str(t) for t in raw_texts]
                    if len(texts) >= 20:
                        feeds_content[cid] = texts

    # 3. Bigramm-Plotting
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    axes = axes.flatten()

    for i, (cat_name, ids) in enumerate(selected_ids_map.items()):
        bigram_pool = []
        actual_count = 0

        for cid in ids:
            if cid in feeds_content:
                actual_count += 1
                bigram_pool.extend(get_ngramms(feeds_content[cid], n=4))

        common = Counter(bigram_pool).most_common(20)
        df_plot = pd.DataFrame(common, columns=['Phrase', 'Count'])

        sns.barplot(data=df_plot, x='Count', y='Phrase', ax=axes[i], palette='rocket')
        axes[i].set_title(f"{cat_name}\n(Bigramm-Kontext von {actual_count} IDs)", fontsize=14)

    plt.tight_layout()
    plt.savefig('context_ngram_analysis.png')
    plt.show()


if __name__ == "__main__":
    main()