import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from _constants import test_feeds_path, bertweet_v3_predictions_dir

# Pfad zur Vorhersagedatei
PRED_PATH = os.path.join(bertweet_v3_predictions_dir, "occupation_test_predictions.json")


def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|@\w+|[^a-z\s]', '', text)
    return [w for w in text.split() if len(w) > 3]


def main():
    # 1. Predictions laden
    with open(PRED_PATH, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    # Hilfsfunktion zur Berechnung der Differenz P(Performer) - P(Creator)
    # Index 1 = Performer, Index 2 = Creator (basierend auf analyse_occupation.py)
    def get_diff(p):
        return p['probabilities'][1] - p['probabilities'][2]

    # 2. Kategorien mit 10 Followern definieren
    # High Confidence False Performer: Maximiere die Differenz (Modell ist sich sehr sicher beim Fehler)
    # Low Confidence False Performer: Minimiere den Absolutwert der Differenz (Modell schwankt)

    categories = {
        "Correct Creator (High)": {
            "filter": lambda x: x['true_label'] == 'creator' and x['pred_label'] == 'creator',
            "sort_key": lambda x: x['probabilities'][2],  # Höchste Creator-Prob
            "reverse": True
        },
        "Correct Performer (High)": {
            "filter": lambda x: x['true_label'] == 'performer' and x['pred_label'] == 'performer',
            "sort_key": lambda x: x['probabilities'][1],  # Höchste Performer-Prob
            "reverse": True
        },
        "False Performer (High Conf)": {
            "filter": lambda x: x['true_label'] == 'creator' and x['pred_label'] == 'performer',
            "sort_key": get_diff,  # Maximiere P(Perf) - P(Creat)
            "reverse": True
        },
        "False Performer (Low Conf)": {
            "filter": lambda x: x['true_label'] == 'creator' and x['pred_label'] == 'performer',
            "sort_key": lambda x: abs(get_diff(x)),  # Minimiere Abstand
            "reverse": False
        }
    }

    selected_ids_map = {}
    all_target_ids = set()

    for cat_name, config in categories.items():
        matches = [p for p in preds if config['filter'](p)]
        top_matches = sorted(matches, key=config['sort_key'], reverse=config['reverse'])[:10]
        ids = [str(m['celebrity_id']) for m in top_matches]
        selected_ids_map[cat_name] = ids
        all_target_ids.update(ids)

    # 3. Feeds aus den PAN-Originaldaten laden (nur IDs mit >= 20 Texten)
    feeds_content = {}
    if os.path.exists(test_feeds_path):
        with open(test_feeds_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                cid = str(obj.get('id'))
                if cid in all_target_ids:
                    raw_texts = obj.get('text', [])
                    # Sicherstellen, dass es eine Liste von Texten ist
                    texts = [t.get('text', '') if isinstance(t, dict) else str(t) for t in raw_texts]

                    if len(texts) >= 20:  # Strenges Limit: 20 Texte
                        feeds_content[cid] = texts

    # 4. Linguistisches Plotting
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()

    for i, (cat_name, ids) in enumerate(selected_ids_map.items()):
        words_pool = []
        actual_count = 0

        for cid in ids:
            if cid in feeds_content:
                actual_count += 1
                for t in feeds_content[cid]:
                    words_pool.extend(clean_and_tokenize(t))

        common = Counter(words_pool).most_common(20)
        df_plot = pd.DataFrame(common, columns=['Word', 'Count'])

        sns.barplot(data=df_plot, x='Count', y='Word', ax=axes[i], palette='viridis')
        axes[i].set_title(f"{cat_name}\n({actual_count}/10 IDs mit >= 20 Tweets)")

    plt.tight_layout()
    plt.savefig('top10_follower_linguistics.png')
    plt.show()


if __name__ == "__main__":
    main()