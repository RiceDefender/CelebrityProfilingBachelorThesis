# BERTweet V3 – Modellvarianten, Beobachtungen und finale Entscheidungen

## 1. Ausgangspunkt

Nach den SBERT-V2-Experimenten wurde eine BERTweet-basierte Modellversion als V3 aufgebaut. Ziel war es, stärker von einem Twitter-spezifischen Transformer-Modell zu profitieren und das Truncation-Problem durch eine chunk-basierte Verarbeitung zu reduzieren.

Die Pipeline basiert auf:

- BERTweet (`vinai/bertweet-base`)
- tokenisierten Celebrity-Follower-Feeds
- chunk-level Training
- celebrity-level Soft Voting
- separaten Modellen für:
  - `occupation`
  - `gender`
  - `birthyear`

Die finale Evaluation erfolgt auf Celebrity-Level.

---

## 2. Tokenisierung und Chunking

Ein wichtiger technischer Punkt war, dass BERTweet nur eine maximale Sequenzlänge von 128 Tokens unterstützt. Ein früherer Versuch mit `MAX_LENGTH=512` führte zu CUDA-Fehlern, weil die Positions-Embeddings von BERTweet nicht für so lange Sequenzen ausgelegt sind.

Daher wurde die Tokenisierung auf:

```python
MAX_LENGTH = 128
```

angepasst.

Dadurch entstehen mehr Chunks pro Celebrity, aber die Eingaben sind kompatibel mit BERTweet.

---

## 3. BERTweet V3 Baseline

Die erste stabile V3-Version trainierte pro Target ein normales BERTweet-Klassifikationsmodell.

Für Occupation, Gender und Birthyear wurden zunächst unterschiedliche Chunk-Mengen getestet. Eine zentrale Erkenntnis war:

``` text
Mehr Chunks helfen nicht allen Targets gleich stark.
```

### Beobachtung
Die Erhöhung auf ungefähr:

``` python
MAX_TRAIN_CHUNKS_PER_CELEB = 32
MAX_VAL_CHUNKS_PER_CELEB = 64
```
führte zu:

| Target     | Wirkung                                          |
| ---------- | ------------------------------------------------ |
| Occupation | deutliche Verbesserung                           |
| Gender     | leichte Verbesserung                             |
| Birthyear  | keine Verbesserung bzw. leichte Verschlechterung |

Die besten BERTweet-V3-Testwerte lagen ungefähr bei:

| Target     | Test Accuracy | Test Macro-F1 |
| ---------- | ------------: | ------------: |
| occupation |        0.6675 |        0.6493 |
| gender     |        0.6800 |        0.6799 |
| birthyear  |        0.4200 |        0.3233 |
---

## 4. Analyse der Occupation-Fehler
Bei Occupation zeigte sich ein klares Muster:

```text
sports, performer und politics werden relativ robust erkannt.
creator bleibt die schwächste Klasse.
```

Die Creator-Klasse wurde häufig verwechselt mit:
- performer
- politics

Die Baseline für Occupation war:

```text
BERTweet V3 Occupation:
Accuracy:  0.6675
Macro-F1:  0.6493
Creator F1: 0.3660
Creator Recall: 0.2800
```

Damit war klar: Das größte Problem ist nicht die Gesamtklassifikation, sondern die gezielte Erkennung von creator.

---

## 5. V3.1 – Creator-Gated Occupation

Aus der Fehleranalyse entstand die Idee, Occupation hierarchisch zu modellieren.

### Idee

Statt nur ein 4-Klassen-Modell zu verwenden:

````text
sports / performer / creator / politics
````

wurden zusätzliche Hilfsmodelle trainiert:

````text
creator_binary:
creator / not_creator

occupation_3class:
sports / performer / politics
````

Die finale Entscheidung in V3.1 erfolgt über Gating:

```python
if p_creator >= high_threshold:
    pred = "creator"

elif p_creator <= low_threshold:
    pred = pred_3class

else:
    pred = pred_4class
```

Die beste Threshold-Kombination auf Validation war:

```python
low_threshold = 0.25
high_threshold = 0.50
```
### Ergebnis V3.1

| Modell      | Test Accuracy | Test Macro-F1 | Creator F1 | Creator Recall |
| ----------- | ------------: | ------------: | ---------: | -------------: |
| V3 baseline |        0.6675 |        0.6493 |     0.3660 |         0.2800 |
| V3.1 gated  |        0.6650 |        0.6527 |     0.3951 |         0.3200 |

### Interpretation

V3.1 verbessert den Macro-F1 leicht und verbessert insbesondere die schwächste Klasse creator.

Der Accuracy-Wert sinkt minimal, aber das Modell ist aus analytischer Sicht interessanter, weil es gezielt das Hauptproblem adressiert.

Daher wird V3.1 für Occupation als stärkste BERTweet-V3-Variante betrachtet.

---
## 6. Test mit aggressiverem Low-Threshold

Zusätzlich wurde getestet, ob ein höherer low_threshold sinnvoll ist:

```python
low_threshold = 0.35
high_threshold = 0.50
```

Das bedeutet: Mehr Fälle werden direkt ins 3-Klassen-Modell geroutet.

| Variante                | Test Accuracy | Test Macro-F1 | Creator F1 |
| ----------------------- | ------------: | ------------: | ---------: |
| V3.1 low=0.25/high=0.50 |        0.6650 |        0.6527 |     0.3951 |
| V3.1 low=0.35/high=0.50 |        0.6525 |        0.6393 |     0.4000 |

### Interpretation
Der höhere Low-Threshold verbessert Creator-F1 minimal, verschlechtert aber den Gesamt-Macro-F1 deutlich.

Das aggressive Routing ins 3-Klassen-Modell ist also nicht stabil. Besonders sports und politics leiden darunter.

Daher wird die konservative Threshold-Kombination beibehalten:

```python
low_threshold = 0.25
high_threshold = 0.50
```

---
## 7. Wichtige Erkenntnis: Creator als Boundary-Klasse

Eine zentrale Beobachtung war:

```text
Obwohl das 3-Klassen-Modell ohne creator auf Validation stark wirkt,
verbessert ein aggressiver Einsatz dieses Modells die Testleistung nicht.
```

Daraus folgt:

```text
creator ist nicht nur eine schwierige Zielklasse,
sondern wirkt im 4-Klassen-Modell auch als strukturelle Abgrenzungsklasse.
```

Creator funktioniert also teilweise als Puffer- oder Boundary-Klasse. Wenn diese Klasse entfernt wird, müssen uneindeutige Fälle zwangsläufig einer der verbleibenden Klassen zugeordnet werden, was die Trennung von sports, performer und politics verschlechtern kann.

---

## 8. V3.2 – Soft Ensemble aus 4-Class und 3-Class

Als weiterer Versuch wurde V3.2 getestet.

### Idee

Statt hart zwischen 3-Class und 4-Class zu routen, werden die Wahrscheinlichkeiten kombiniert:

```python
final_creator_prob = p4_creator

final_non_creator_probs =
    alpha * p4_non_creator_probs
    + (1 - alpha) * p3_probs
```

Damit bleibt creator als Boundary-Signal erhalten, während das 3-Class-Modell die Verteilung zwischen sports, performer und politics beeinflussen kann.

### Beobachtung
V3.2 zeigte gute Validation-Ergebnisse, generalisierte aber schlechter auf Test.

Die Testleistung lag ungefähr bei:

```text
Test Accuracy: 0.6650
Test Macro-F1: 0.6395
Creator F1: 0.3521
```

Damit war V3.2 schlechter als:

V3 baseline
V3.1 gated

### Entscheidung

V3.2 wird nicht als finales Modell verwendet.

Es bleibt eine sinnvolle Ablation, zeigt aber, dass das Soft-Ensemble der BERTweet-Varianten keinen stabilen Gewinn bringt.

---

## 8.1 V3.3 – Creator-Binary Override ohne 3-Class-Modell

Nach V3.1 und V3.2 wurde zusätzlich geprüft, ob das 3-Klassen-Modell überhaupt notwendig ist. Dafür wurde eine vereinfachte Variante getestet, die nur das ursprüngliche 4-Klassen-Occupation-Modell und das binäre Creator-Modell verwendet.

### Idee

Das 4-Klassen-Modell bleibt das Hauptmodell:

```text
sports / performer / creator / politics
```
Zusätzlich wird nur dann überschrieben, wenn das binäre Creator-Modell eine genügend hohe Creator-Wahrscheinlichkeit liefert:

```python
if p_creator_binary >= threshold:
    pred = "creator"
else:
    pred = pred_4class
```

Damit wird kein separates 3-Klassen-Modell mehr verwendet. Die strukturelle Rolle von creator im ursprünglichen 4-Klassen-Modell bleibt also erhalten.

### Threshold Search

Die beste Threshold-Kombination auf Validation war:
```python
threshold = 0.50
min_p4_creator = 0.00
```

Auf Validation ergab diese Einstellung:

```text
Val Accuracy: 0.7240
Val Macro-F1: 0.7196
Creator F1: 0.5773
```

Ergebnis auf Test
```text
Test Accuracy: 0.6650
Test Macro-F1: 0.6527
Creator F1: 0.3951
Creator Recall: 0.3200
```

Die Decision-Distribution auf dem Testset war:
````text
use_4class: 371
creator_override: 29
````

Das bedeutet, dass V3.3 sehr konservativ eingreift. Die meisten Fälle bleiben beim robusteren 4-Klassen-Modell, während nur wenige Fälle durch das Creator-Binary-Modell überschrieben werden.

### Vergleich mit V3.1

| Modell                | Test Accuracy | Test Macro-F1 | Creator F1 | Creator Recall |
| --------------------- | ------------: | ------------: | ---------: | -------------: |
| V3 baseline           |        0.6675 |        0.6493 |     0.3660 |         0.2800 |
| V3.1 gated            |        0.6650 |        0.6527 |     0.3951 |         0.3200 |
| V3.3 creator override |        0.6650 |        0.6527 |     0.3951 |         0.3200 |

V3.3 erreicht damit dieselbe Testleistung wie V3.1, benötigt aber kein zusätzliches 3-Klassen-Occupation-Modell.

### Interpretation

Die Verbesserung gegenüber der normalen V3-Baseline entsteht nicht durch das Entfernen der Klasse creator aus dem Mehrklassenproblem. Stattdessen scheint das zusätzliche binäre Creator-Signal der relevante Faktor zu sein.

Damit bestätigt V3.3 die frühere Beobachtung, dass creator im 4-Klassen-Modell eine strukturelle Boundary-Rolle besitzt. Das Entfernen dieser Klasse in einem 3-Klassen-Modell ist nicht notwendig und kann bei aggressiver Nutzung sogar die Testleistung verschlechtern.

### Entscheidung

V3.3 wird als finales Occupation-Modell für BERTweet V3 gewählt, weil es:

- dieselbe Test-Macro-F1 wie V3.1 erreicht,
- dieselbe Verbesserung der Creator-Klasse erzielt,
- einfacher und besser interpretierbar ist,
- kein zusätzliches 3-Klassen-Modell benötigt,
- die robuste 4-Klassen-Struktur beibehält.

---

### 9. Birthyear

Birthyear bleibt das schwierigste Target.

Die Klassifikation in fünf Buckets:

```text
1994 / 1985 / 1975 / 1963 / 1947
```

zeigt eine starke Tendenz zur mittleren Klasse, insbesondere 1985.

Die Erhöhung der Chunk-Anzahl half hier nicht. Teilweise wurde die älteste Klasse 1947 gar nicht mehr erkannt.

### Entscheidung

Für Birthyear wird innerhalb von V3 das beste normale BERTweet-Modell verwendet.

Weitere Verbesserungen sollten eher über andere Ansätze geprüft werden, zum Beispiel:

- Regression statt Klassifikation
- ordinales Lernen
- Hybrid-Features
- TF-IDF/Surface Features

Birthyear wird nicht über V3.1 oder V3.2 erweitert, da diese Ansätze speziell auf das Occupation-Creator-Problem abzielen.

---
## 10. Gender

Gender ist das stabilste Target.

Die BERTweet-V3-Ergebnisse sind relativ ausgewogen zwischen male und female.

Da keine dominante Problemklasse wie bei Occupation sichtbar ist, wurden keine zusätzlichen Gating- oder Ensemble-Varianten für Gender eingeführt.

### Entscheidung

Für Gender wird das normale BERTweet-V3-Modell verwendet.

---

## 11. Finale V3-Entscheidung

| Target     | Finales V3-Modell                           | Begründung                                                              |
| ---------- | ------------------------------------------- | ----------------------------------------------------------------------- |
| occupation | BERTweet V3.3 Creator Override | gleiche Testleistung wie V3.1, aber einfacher; verbessert Creator ohne 3-Class-Modell |
| gender     | BERTweet V3 baseline                        | stabil, ausgewogen, keine zusätzliche Hierarchie nötig                  |
| birthyear  | BERTweet V3 baseline / bester Birthyear-Run | V3.1/V3.2 nicht relevant; Chunk-Erhöhung half nicht                     |

Für Occupation gilt:

```python
low_threshold = 0.25
high_threshold = 0.50
```
Die finale Occupation-Entscheidung basiert auf:

```python
if p_creator_binary >= 0.50:
    pred = "creator"
else:
    pred = pred_4class
```

V3.1 bleibt eine wichtige Ablation, wird aber nicht als finales Modell gewählt. Der Grund ist, dass V3.3 dieselbe Testleistung erreicht, aber weniger komplex ist und kein zusätzliches 3-Klassen-Modell benötigt.



---

## 12. Warum nicht weiter an V3 tunen?

Die Experimente zeigen:
```text
BERTweet alleine bringt bereits gute Verbesserungen,
aber weitere BERTweet-interne Entscheidungslogik bringt nur kleine Gewinne.
```

V3.1 und V3.3 verbessern Occupation leicht, aber der Effekt ist begrenzt. V3.3 ist dabei vorzuziehen, weil es dieselbe Testleistung wie V3.1 erreicht, aber mit einer einfacheren Entscheidungslogik auskommt.

V3.2 zeigt sogar, dass komplexere Kombinationen der BERTweet-Modelle nicht automatisch besser generalisieren.

Daher ist der nächste sinnvolle Schritt nicht weiteres BERTweet-Tuning, sondern ein Hybridmodell.

---

## 13. Übergang zu V4 Hybrid
Die bisherigen Ergebnisse motivieren V4:
```text
BERTweet liefert starke semantische Signale.
TF-IDF/SVM bzw. Logistic Regression kann sparse Oberflächenmerkmale erfassen.
```

Gerade für Author/Celebrity Profiling sind solche Merkmale wichtig:

- Wörter und N-Grams
- Hashtags
- Emojis
- Plattform- und Themenmarker
- typische Begriffe für Sport, Politik, Creator, Performer
- Stil- und Surface Features

### Geplante V4-Features

````text
1. TF-IDF word n-grams
2. TF-IDF character n-grams
3. BERTweet probabilities
4. optional:
   - creator_binary probability
   - 3class occupation probabilities
   - einfache linguistische Features
````

#### Ziel von V4

V4 soll nicht nur BERTweet ersetzen, sondern BERTweet ergänzen.

Die wichtigste Hypothese lautet:

```text
Ein Hybridmodell aus kontextuellen Transformer-Signalen und sparse TF-IDF-Features
kann die Schwächen reiner BERTweet-Modelle besser ausgleichen.
```

---

## 14. Zusammenfassung

Die BERTweet-V3-Experimente zeigen:

1. BERTweet ist klar stärker als einfache frühe Baselines.
2. Mehr Chunks helfen besonders bei Occupation und leicht bei Gender.
3. Birthyear bleibt schwierig und profitiert nicht zuverlässig von mehr Chunks.
4. Creator ist die zentrale Problemklasse bei Occupation.
5. Eine konservative hierarchische V3.1-Strategie verbessert Creator und Macro-F1 leicht.
6. Aggressives 3-Class-Routing verschlechtert die Testleistung.
7. Creator wirkt im 4-Class-Modell als strukturelle Boundary-Klasse.
8. V3.2 Soft-Ensembling generalisiert schlechter und wird nicht final verwendet.
9. Die finale V3-Version nutzt:
   1. V3.3 für Occupation
   2. V3 baseline für Gender
   3. V3 baseline / bester Run für Birthyear
10. Der nächste große Schritt ist V4 Hybrid mit TF-IDF und BERTweet-Probability-Features.