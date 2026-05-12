# `plot_prediction_signals.py` – Erklärung

Das Skript visualisiert, welche sprachlichen Signale in den Follower-Feeds von Celebrities vorkommen, abhängig davon, wie BERTweet sie vorhergesagt hat.

Beispielaufruf:

```bash
python -m DataAnalyser.plot_prediction_signals --target gender --split test --modes frequency style --ngrams 2 3
```

Wichtig: Das Skript erklärt nicht direkt die interne Attention von BERTweet. Es zeigt stattdessen, welche Wörter, N-Gramme und Stilmerkmale in den Eingaben besonders häufig bei bestimmten Vorhersagegruppen auftreten. Es ist also eine korrelationsbasierte Fehler- und Signalanalyse.

---

## Grundidee

Das Skript verbindet drei Datenquellen:

1. die BERTweet-Prediction-Datei, zum Beispiel:

```text
gender_test_predictions.json
occupation_test_predictions.json
birthyear_val_predictions.json
```

2. die passenden Follower-Feeds:

```text
test_follower-feeds.ndjson
train_follower-feeds.ndjson
```

3. die Projektpfade aus `_constants.py`.

Dann gruppiert es die Celebrities nach Vorhersageverhalten, zum Beispiel:

```text
Correct male high
Correct female high
False male high
False female high
```

Für jede Gruppe werden anschließend N-Gramme oder Stilfeatures gesammelt und geplottet.

---

## Beispiel: Gender-Analyse

```bash
python -m DataAnalyser.plot_prediction_signals --target gender --split test --modes frequency style --ngrams 2 3
```

Dieser Befehl bedeutet:

```text
--target gender
```

Analysiere das Gender-Modell.

```text
--split test
```

Nutze die Test-Predictions und die Test-Feeds.

```text
--modes frequency style
```

Erzeuge Frequenzplots für N-Gramme und zusätzlich Stilfeature-Plots.

```text
--ngrams 2 3
```

Analysiere Bigramme und Trigramme.

---

## Modi

### `frequency`

Zeigt die häufigsten N-Gramme innerhalb einer Gruppe.

Beispielgruppen bei `gender`:

```text
Correct male high
Correct female high
False male high
False female high
```

Das beantwortet Fragen wie:

```text
Welche Bigramme kommen oft in Feeds vor, bei denen das Modell korrekt female vorhersagt?
Welche Trigramme kommen oft in Feeds vor, bei denen das Modell fälschlich male vorhersagt?
```

Beispiel:

```bash
python -m DataAnalyser.plot_prediction_signals --target occupation --split test --modes frequency --ngrams 2 3
```

---

### `logodds`

Zeigt nicht nur häufige, sondern überrepräsentierte N-Gramme.

Das ist oft stärker als reine Frequenz, weil generische Phrasen wie `thank you`, `for the`, `you are` überall häufig sein können.

Log-Odds fragt eher:

```text
Welche N-Gramme sind in dieser Gruppe stärker vertreten als im Rest der Daten?
```

Beispiel:

```bash
python -m DataAnalyser.plot_prediction_signals --target occupation --split test --modes logodds --ngrams 2 3
```

Das ist besonders sinnvoll für:

```text
sports
politics
creator vs performer
```

---

### `style`

Berechnet einfache Twitter-Stilmerkmale pro Celebrity und visualisiert sie als Boxplots.

Das sind zum Beispiel:

```text
avg_tokens_per_tweet
url_count
mention_count
hashtag_count
emoji_count
exclamation_count
question_count
uppercase_ratio
love_word_count
fan_word_count
politics_word_count
sports_word_count
```

Beispiel:

```bash
python -m DataAnalyser.plot_prediction_signals --target gender --split test --modes style
```

Damit sieht man zum Beispiel, ob Feeds, die als `female` vorhergesagt werden, mehr Emojis, Fan-Wörter oder Exclamation Marks enthalten.

---

## Targets

Das Skript unterstützt aktuell:

```text
occupation
occupation_3class
creator_binary
gender
birthyear
```

### `occupation`

4 Klassen:

```text
sports
performer
creator
politics
```

Beispiel:

```bash
python -m DataAnalyser.plot_prediction_signals --target occupation --split test --modes frequency logodds style --ngrams 2 3
```

### `occupation_3class`

3 Klassen:

```text
sports
performer
politics
```

Creator ist hier nicht enthalten.

Beispiel:

```bash
python -m DataAnalyser.plot_prediction_signals --target occupation_3class --split test --modes frequency logodds --ngrams 2 3
```

### `creator_binary`

Binäre Creator-Erkennung:

```text
creator
not_creator
```

Beispiel:

```bash
python -m DataAnalyser.plot_prediction_signals --target creator_binary --split test --modes frequency logodds --ngrams 2 3
```

### `gender`

```text
male
female
```

Beispiel:

```bash
python -m DataAnalyser.plot_prediction_signals --target gender --split test --modes frequency style --ngrams 2 3
```

### `birthyear`

```text
1994
1985
1975
1963
1947
```

Beispiel:

```bash
python -m DataAnalyser.plot_prediction_signals --target birthyear --split val --modes frequency logodds style --ngrams 2
```

Bei `birthyear` gibt es zusätzlich eine Analyse nach Fehlerrichtung:

```text
Predicted younger than true
Predicted older than true
Wrong/high confidence
```

---

## Splits

```text
--split test
```

nutzt Test-Predictions und Test-Feeds.

```text
--split val
```

nutzt Validation-Predictions und die Trainingsfeeds.

```text
--split val_all
```

nutzt Validation-All-Predictions und ebenfalls die Trainingsfeeds.

---

## Analysegruppen

Standardmäßig erzeugt das Skript diese Gruppen:

### Für `occupation`, `gender`, `creator_binary`, `occupation_3class`

```text
correct_by_class
false_by_pred
```

Das heißt:

```text
Correct sports high
Correct performer high
Correct creator high
Correct politics high

False sports high
False performer high
False creator high
False politics high
```

### Für `birthyear`

Standardmäßig:

```text
correct_by_class
age_direction
```

Also zusätzlich:

```text
Predicted younger than true
Predicted older than true
Wrong/high confidence
```

---

## `--analyses`

Mit `--analyses` kann man die Gruppen manuell setzen.

Beispiel:

```bash
python -m DataAnalyser.plot_prediction_signals --target occupation --split test --modes frequency logodds --ngrams 2 --analyses confusion_pairs
```

Dann werden konkrete Fehlerübergänge geplottet, zum Beispiel:

```text
creator → performer
politics → creator
sports → performer
```

Das ist sehr hilfreich für die Fehleranalyse.

---

## Stopwords entfernen

Standardmäßig bleiben Stopwords erhalten.

Das ist absichtlich so, weil Stopwords und generische Fan-Phrasen selbst ein interessantes Signal sein können.

Beispiel mit Stopwords:

```bash
python -m DataAnalyser.plot_prediction_signals --target occupation --split test --modes frequency --ngrams 2
```

Beispiel ohne Stopwords:

```bash
python -m DataAnalyser.plot_prediction_signals --target occupation --split test --modes frequency logodds --ngrams 2 3 --remove-stopwords
```

Mit `--remove-stopwords` werden die Plots thematischer. Dann sieht man eher Wörter wie:

```text
game
team
vote
president
album
youtube
```

statt:

```text
for the
you are
thank you
love you
```

---

## Alle Dokumente statt Top-Confidence

Standardmäßig nimmt das Skript pro Gruppe nur die Top-confident Celebrities:

```text
--top-docs 20
```

Das heißt: Für jede Gruppe werden die 20 sichersten Fälle genutzt.

Wenn man alle Fälle verwenden will:

```bash
python -m DataAnalyser.plot_prediction_signals --target gender --split test --modes frequency --ngrams 2 --top-docs 0
```

Das ist sinnvoll, wenn man robuste Gesamtsignale sehen möchte.

Top-Confidence ist dagegen sinnvoll, wenn man sehen will, welche Signale das Modell bei sehr sicheren Entscheidungen begleitet.

---

## Output

Die Ergebnisse landen standardmäßig unter:

```text
plots/prediction_signals/<target>/<split>/
```

Beispiel:

```text
plots/prediction_signals/gender/test/
```

Dort entstehen:

```text
.png
.csv
```

Die `.png`-Dateien sind die Visualisierungen.

Die `.csv`-Dateien enthalten die geplotteten Werte als Tabelle, zum Beispiel:

```text
group, rank, ngram, count
Correct female high, 1, thank you, 1523
Correct female high, 2, love you, 1410
```

oder bei Log-Odds:

```text
group, rank, ngram, log_odds
Correct sports high, 1, final score, 3.21
Correct sports high, 2, nba playoffs, 2.98
```

---

## Wichtige Optionen

### `--top-k`

Wie viele N-Gramme pro Gruppe geplottet werden.

```bash
--top-k 20
```

Standard ist 20.

---

### `--top-docs`

Wie viele Celebrities pro Gruppe genutzt werden.

```bash
--top-docs 20
```

Standard ist 20.

Alle nutzen:

```bash
--top-docs 0
```

---

### `--max-tweets-per-celebrity`

Begrenzt die Anzahl Tweets pro Celebrity.

```bash
--max-tweets-per-celebrity 500
```

Standard ist 0, also alle verfügbaren Tweets.

---

### `--min-token-len`

Minimale Tokenlänge.

```bash
--min-token-len 2
```

Standard ist 2.

---

### `--drop-hashtags`

Entfernt das `#`-Symbol und behandelt Hashtags wie normale Wörter.

```bash
--drop-hashtags
```

Ohne diese Option bleiben Hashtags als Hashtags erhalten, zum Beispiel:

```text
#worldcup
#newmusic
```

---

### `--pred-path`

Falls die Prediction-Datei nicht automatisch gefunden wird:

```bash
python -m DataAnalyser.plot_prediction_signals --target gender --split test --pred-path outputs/bertweet_v3/predictions/gender_test_predictions.json
```

---

### `--feed-path`

Falls die Feed-Datei manuell gesetzt werden soll:

```bash
python -m DataAnalyser.plot_prediction_signals --target gender --split test --feed-path data/.../follower-feeds.ndjson
```

---

### `--out-dir`

Falls die Outputs an einen anderen Ort geschrieben werden sollen:

```bash
python -m DataAnalyser.plot_prediction_signals --target gender --split test --out-dir plots/my_gender_analysis
```

---

## Sinnvolle Beispielaufrufe

### Gender: häufige Bigramme/Trigramme + Stil

```bash
python -m DataAnalyser.plot_prediction_signals --target gender --split test --modes frequency style --ngrams 2 3
```

---

### Gender: sauberere thematische Signale

```bash
python -m DataAnalyser.plot_prediction_signals --target gender --split test --modes frequency logodds --ngrams 2 3 --remove-stopwords
```

---

### Occupation: Sports, Performer, Creator, Politics vergleichen

```bash
python -m DataAnalyser.plot_prediction_signals --target occupation --split test --modes frequency logodds style --ngrams 2 3
```

---

### Occupation: konkrete Fehlklassifikationen

```bash
python -m DataAnalyser.plot_prediction_signals --target occupation --split test --modes frequency logodds --ngrams 2 3 --analyses confusion_pairs
```

---

### Sports/Politics-Signale sauber anschauen

```bash
python -m DataAnalyser.plot_prediction_signals --target occupation_3class --split test --modes logodds --ngrams 2 3 --remove-stopwords
```

---

### Creator-Erkennung

```bash
python -m DataAnalyser.plot_prediction_signals --target creator_binary --split test --modes frequency logodds style --ngrams 2 3
```

---

### Birthyear: Richtung der Fehler

```bash
python -m DataAnalyser.plot_prediction_signals --target birthyear --split val --modes frequency logodds style --ngrams 2 --analyses age_direction
```

---

## Interpretation

Die Plots sollten vorsichtig interpretiert werden.

Besser nicht schreiben:

```text
BERTweet achtet auf diese N-Gramme.
```

Besser:

```text
Die Analyse zeigt, welche N-Gramme in den Follower-Feeds bestimmter Vorhersagegruppen besonders häufig oder überrepräsentiert sind. Diese Signale liefern Hinweise darauf, welche sprachlichen Muster mit korrekten oder falschen Modellentscheidungen korrelieren.
```

Für die Arbeit ist besonders spannend:

```text
Sports und Politics sollten klarere thematische N-Gramme zeigen.
Creator und Performer könnten stärker von Fan-Sprache, Plattformphrasen und Named Entities geprägt sein.
Gender könnte eher über Stilmerkmale und Community-Sprache als über harte Themenwörter sichtbar werden.
Birthyear kann zeigen, ob das Modell systematisch zu jung oder zu alt schätzt.
```
