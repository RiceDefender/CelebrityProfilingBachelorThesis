# BERTweet V3.4 Stopword Tokenizer

## Ziel

Diese Version ist ein kontrolliertes Tokenizer-Experiment für BERTweet. Die bestehende BERTweet-V3-Tokenisierung bleibt unverändert. V3.4 erzeugt einen separaten tokenisierten Output und entfernt vor der BERTweet-Tokenisierung eine konservative Stopword-Liste.

Die bestehende Struktur bleibt erhalten:

- `vinai/bertweet-base`
- `MAX_LENGTH = 128`
- `STRIDE = 32`
- `MAX_CHUNKS_PER_CELEBRITY = None`, also alle Chunks behalten
- `MIN_TOKENS_PER_CHUNK = 16`
- Tweet-Auswahl: `evenly_spaced`, maximal 5000 Tweets pro Celebrity

## Dateien

Lege die Dateien hier ab:

```text
Preprocessing/tokenizers/bertweet/config_bertweet_v34_stopwords.py
Preprocessing/tokenizers/bertweet/tokenize_bertweet_v34_stopwords.py
```

Die Outputs landen separat unter:

```text
Preprocessing/data/bertweet_v3_4_stopwords_tokenized_chunked/
```

Erzeugte Dateien:

```text
train_tokenized.ndjson
test_tokenized.ndjson
supp_tokenized.ndjson
train_meta.json
test_meta.json
supp_meta.json
```

Damit wird der vorhandene Raw-BERTweet-V3-Output nicht überschrieben.

## Warum konservativ?

Stopwords können für Author Profiling auch Stilinformationen enthalten. Deshalb entfernt V3.4 standardmäßig nur eine harte Stopword-Liste und behält Pronomen und Negationen:

```text
i, you, we, he, she, they, my, your, her, his, their
not, no, never, without, don't, can't, won't
```

Das ist sicherer als eine aggressive Stopword-Entfernung, weil Gender- und Age-Signale teilweise stilistisch sein können.

## Standardlauf

```bash
python -m Preprocessing.tokenizers.bertweet.tokenize_bertweet_v34_stopwords --split all
```

Schneller Debuglauf:

```bash
python -m Preprocessing.tokenizers.bertweet.tokenize_bertweet_v34_stopwords --split train --limit-celebrities 5
```

## Optionen

### Stopwords deaktivieren

```bash
python -m Preprocessing.tokenizers.bertweet.tokenize_bertweet_v34_stopwords --split train --no-stopwords
```

### Pronomen ebenfalls entfernen

```bash
python -m Preprocessing.tokenizers.bertweet.tokenize_bertweet_v34_stopwords --split train --drop-pronouns
```

Das ist riskanter, kann aber als Ablation interessant sein.

### Negationen ebenfalls entfernen

```bash
python -m Preprocessing.tokenizers.bertweet.tokenize_bertweet_v34_stopwords --split train --drop-negations
```

Das ist normalerweise nicht empfohlen, weil `not good` sonst semantisch zu `good` werden kann.

### `@USER` und `HTTPURL` entfernen

```bash
python -m Preprocessing.tokenizers.bertweet.tokenize_bertweet_v34_stopwords --split train --remove-social-tokens
```

Für BERTweet würde ich das erst als zweite Ablation testen. Mentions und URLs können Profiling-Signale sein, auch wenn sie N-Gramm-Plots stark stören.

### RT-Artefakte behalten

Standardmäßig entfernt V3.4 `rt`, `via`, `amp`, `gt`, `lt`. Zum Behalten:

```bash
python -m Preprocessing.tokenizers.bertweet.tokenize_bertweet_v34_stopwords --split train --keep-rt-artifacts
```

## Kritische Punkte

1. **Nicht als Ersatz für Raw-BERTweet behandeln.**  
   V3.4 ist ein Ablationsexperiment. Raw-BERTweet bleibt die zentrale Baseline.

2. **Pronomen und Negationen standardmäßig behalten.**  
   Diese Wörter können für Profiling relevant sein.

3. **Social Tokens nicht sofort entfernen.**  
   `@USER` und `HTTPURL` stören N-Gramm-Plots, können aber im Transformer nützliche Signale sein.

4. **Chunking bleibt gleich.**  
   Dadurch ist der Vergleich V3 vs. V3.4 fairer, weil sich primär die Textnormalisierung ändert.

## Was später verglichen werden sollte

Für jedes Target:

```text
BERTweet V3 raw
BERTweet V3.4 stopwords
BERTweet V3.4 stopwords + remove-social-tokens
TF-IDF clean
TF-IDF clean + style
Hybrid: BERTweet raw + TF-IDF clean + style + profile prior
```

Besonders interessant:

- Occupation: Hilft Stopword-Filterung bei Sports/Politics und Creator/Performer?
- Gender: Verschlechtert Stopword-Filterung die Stilinformation?
- Birthyear: Werden Age-Fehler durch Verlust von Community-/Fandom-Sprache stärker?

## Wissenschaftliche Formulierung

> Zur Untersuchung des Einflusses hochfrequenter Funktionswörter wurde eine zusätzliche BERTweet-Tokenisierung V3.4 implementiert. Im Unterschied zur Raw-BERTweet-V3-Tokenisierung entfernt diese Variante vor der BERTweet-Tokenisierung eine konservative Stopword-Liste, behält jedoch Pronomen und Negationen als potenziell profilrelevante Stilmerkmale bei. Die Chunking-Strategie bleibt unverändert, sodass Unterschiede in der Modellleistung primär auf die veränderte Textnormalisierung zurückgeführt werden können.
