# BERTweet V3.4 Stopword-Filtered Model Setup

## Ziel

V3.4 ist eine kontrollierte Ablation zu BERTweet V3:

- V3 nutzt rohe BERTweet-Tokenisierung.
- V3.4 nutzt stopword-gefilterte BERTweet-Tokenisierung.
- Training, Chunk-Limits und Soft-Voting bleiben möglichst identisch zu V3.

Dadurch wird getestet, ob Stopword-Filtering bei einem Twitter-vortrainierten Transformer hilft oder schadet.

## Dateien

Lege die Dateien so ab:

```text
Models/BERTweetV34/__init__.py
Models/BERTweetV34/config_bertweet_v34_model.py
Models/BERTweetV34/age_bins_v34.py
Models/BERTweetV34/train_bertweet_v34.py
Models/BERTweetV34/predict_bertweet_v34.py
Models/BERTweetV34/evaluate_bertweet_v34_test_predictions.py
```

Zusätzlich muss `_constants.py` um die V3.4-Pfade erweitert werden. Siehe `constants_v34_addition.py`.

## Reihenfolge

### 1. Stopword-tokenisierte Daten erzeugen

```bash
python -m Preprocessing.tokenizers.bertweet.tokenize_bertweet_v34_stopwords --split all
```

### 2. Modelle trainieren

```bash
python -m Models.BERTweetV34.train_bertweet_v34 --target occupation
python -m Models.BERTweetV34.train_bertweet_v34 --target gender
python -m Models.BERTweetV34.train_bertweet_v34 --target birthyear
```

Für die Koloski-inspirierte Age-Range-Variante:

```bash
python -m Models.BERTweetV34.train_bertweet_v34 --target birthyear_8range
```

### 3. Test-Predictions erzeugen

```bash
python -m Models.BERTweetV34.predict_bertweet_v34 --target occupation
python -m Models.BERTweetV34.predict_bertweet_v34 --target gender
python -m Models.BERTweetV34.predict_bertweet_v34 --target birthyear
python -m Models.BERTweetV34.predict_bertweet_v34 --target birthyear_8range
```

### 4. Test evaluieren

```bash
python -m Models.BERTweetV34.evaluate_bertweet_v34_test_predictions --target all
python -m Models.BERTweetV34.evaluate_bertweet_v34_test_predictions --target birthyear_8range
```

## Birthyear-Varianten

### `birthyear`

Das ist die bisherige 5-Klassen-Variante:

```text
1994 / 1985 / 1975 / 1963 / 1947
```

### `birthyear_8range`

Diese Variante bildet acht train-quantile Age-Ranges. Die Grenzen werden nur aus dem Trainingsset gelernt und in folgendem Pfad gespeichert:

```text
outputs/bertweet_v3_4_stopwords/metrics/birthyear_8range_bins.json
```

Dadurch wird kein Test-Verteilungswissen geleakt.

## Interpretation

V3.4 sollte nicht automatisch als neuer Hauptweg betrachtet werden. Es ist primär ein Test:

- Hilft Stopword-Filtering bei `occupation`?
- Schadet es `gender`, weil Funktionswörter und Stil verloren gehen?
- Verändert es `birthyear`, wo Community- und Stilmerkmale wichtig sein können?

Wenn V3.4 bei BERTweet nicht hilft, ist das trotzdem ein Ergebnis: Stopword-Filtering ist dann wahrscheinlich besser für TF-IDF/Hybridfeatures geeignet als für den Transformer-Eingang.
