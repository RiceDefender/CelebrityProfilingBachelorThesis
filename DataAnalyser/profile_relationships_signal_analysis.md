# Profil-Zusammenhänge: Occupation × Gender × Birthyear

## Ziel der Analyse

Bisher wurden `occupation`, `gender` und `birthyear` als drei getrennte Vorhersageaufgaben betrachtet. Da alle drei Labels aber zur gleichen Celebrity gehören, können zwischen den Profil-Dimensionen statistische Zusammenhänge bestehen. Diese Zusammenhänge sind für die Modellierung relevant, weil sie zeigen können, ob BERTweet nur Textsignale nutzt oder zusätzlich implizite Datensatz-Priors reproduziert bzw. verstärkt.

Die zentrale Frage lautet:

> Können gemeinsame Profilstrukturen wie `occupation + gender + age` genutzt werden, um unsichere oder fehleranfällige Vorhersagen zu stabilisieren?

Gleichzeitig muss man vorsichtig sein: Solche Zusammenhänge können echte Muster in der Followerbasis widerspiegeln, aber auch Dataset-Bias verstärken.

---

## Datengrundlage

Analysiert wurden:

- Trainingslabels: 1'920 Celebrities
- Testlabels: 400 Celebrities
- Test-Predictions: 400 Celebrities

Die Occupation-Klassen sind im Trainings- und Testdatensatz balanciert:

| Split | sports | creator | performer | politics |
|---|---:|---:|---:|---:|
| Train Labels | 25.0 % | 25.0 % | 25.0 % | 25.0 % |
| Test Labels | 25.0 % | 25.0 % | 25.0 % | 25.0 % |
| Test Predictions | 20.2 % | 13.2 % | 38.2 % | 28.2 % |

Auffällig ist: Das Modell sagt deutlich zu selten `creator` und deutlich zu häufig `performer` voraus. Das passt zu den vorherigen Fehleranalysen, in denen `creator → performer` ein zentraler Fehler war.

---

## Gender-Verteilung nach Occupation

### Ground Truth: Train

| Occupation | female | male |
|---|---:|---:|
| creator | 50.0 % | 50.0 % |
| performer | 50.0 % | 50.0 % |
| politics | 26.7 % | 73.3 % |
| sports | 50.0 % | 50.0 % |

### Ground Truth: Test

| Occupation | female | male |
|---|---:|---:|
| creator | 50.0 % | 50.0 % |
| performer | 50.0 % | 50.0 % |
| politics | 50.0 % | 50.0 % |
| sports | 50.0 % | 50.0 % |

### Predictions: Test

| Predicted Occupation | predicted female | predicted male |
|---|---:|---:|
| creator | 88.7 % | 11.3 % |
| performer | 66.0 % | 34.0 % |
| politics | 18.6 % | 81.4 % |
| sports | 48.1 % | 51.9 % |

## Interpretation

Die Prediction-Struktur weicht klar von den Testlabels ab. Besonders auffällig:

- `creator` wird in den Predictions extrem stark mit `female` gekoppelt.
- `politics` wird extrem stark mit `male` gekoppelt.
- `performer` wird ebenfalls deutlich stärker als `female` vorhergesagt.
- `sports` bleibt relativ ausgeglichen.

Bei `politics` gibt es im Training tatsächlich einen männlichen Bias: 73.3 % der Politics-Celebrities im Training sind männlich. Im Testset ist Politics jedoch 50/50 verteilt. Das Modell scheint hier also teilweise einen Trainingsprior zu übernehmen und im Test zu verstärken.

Bei `creator` ist dieser Effekt problematischer: Im Training und Test ist `creator` 50/50 verteilt, aber in den Predictions ist `creator` zu 88.7 % weiblich. Das spricht nicht für einen einfachen Dataset-Prior, sondern eher für eine Modellverzerrung oder für Text-/Community-Signale, die vom Modell stark mit weiblicher Follower- bzw. Fan-Sprache verbunden werden.

---

## Birthyear-Verteilung nach Occupation

Da das Modell Birthyear als fünf Klassen vorhersagt (`1947`, `1963`, `1975`, `1985`, `1994`), wurden die echten Geburtsjahre für den Vergleich auf die nächstgelegene dieser Klassen abgebildet.

### Ground Truth: Train, P(age-bin | occupation)

| Occupation | 1947 | 1963 | 1975 | 1985 | 1994 |
|---|---:|---:|---:|---:|---:|
| creator | 10.2 % | 30.2 % | 34.4 % | 21.2 % | 4.0 % |
| performer | 4.4 % | 15.6 % | 30.0 % | 30.2 % | 19.8 % |
| politics | 28.5 % | 39.0 % | 25.6 % | 6.5 % | 0.4 % |
| sports | 1.5 % | 7.7 % | 22.9 % | 42.3 % | 25.6 % |

### Ground Truth: Test, P(age-bin | occupation)

| Occupation | 1947 | 1963 | 1975 | 1985 | 1994 |
|---|---:|---:|---:|---:|---:|
| creator | 20.0 % | 30.0 % | 26.0 % | 13.0 % | 11.0 % |
| performer | 4.0 % | 6.0 % | 30.0 % | 38.0 % | 22.0 % |
| politics | 28.0 % | 48.0 % | 18.0 % | 5.0 % | 1.0 % |
| sports | 0.0 % | 5.0 % | 15.0 % | 58.0 % | 22.0 % |

### Predictions: Test, P(predicted age-bin | predicted occupation)

| Predicted Occupation | 1947 | 1963 | 1975 | 1985 | 1994 |
|---|---:|---:|---:|---:|---:|
| creator | 0.0 % | 7.5 % | 81.1 % | 11.3 % | 0.0 % |
| performer | 0.0 % | 0.0 % | 0.7 % | 83.7 % | 15.7 % |
| politics | 2.7 % | 54.9 % | 38.9 % | 2.7 % | 0.9 % |
| sports | 0.0 % | 0.0 % | 3.7 % | 96.3 % | 0.0 % |

## Interpretation

Die Birthyear-Predictions sind stark komprimiert. Während Train und Test eine breite Altersverteilung haben, konzentrieren sich die Predictions auf wenige Kombinationen:

- `sports` wird fast immer als `1985` vorhergesagt.
- `performer` wird fast immer als `1985` oder `1994` vorhergesagt.
- `creator` wird fast immer als `1975` vorhergesagt.
- `politics` wird fast immer als `1963` oder `1975` vorhergesagt.

Das Modell erkennt also offenbar grobe Altersrichtungen nach Occupation, verliert aber viel Granularität. Besonders die älteste Klasse `1947` wird fast nie vorhergesagt, obwohl sie im Train- und Testset vorkommt. Das bestätigt die vorherige Birthyear-Analyse: Das Alter scheint schwerer direkt aus Follower-Texten zu lernen zu sein und wird vermutlich über indirekte Signale wie Occupation, Community-Themen und Fandom-Sprache geschätzt.

---

## Profilkombinationen: starke Konzentration in den Predictions

### Train Labels

- Anzahl beobachteter Profilkombinationen: 39
- Größte einzelne Kombination: 7.3 %
- Top-5-Kombinationen: 28.2 %
- Top-10-Kombinationen: 49.2 %

### Test Labels

- Anzahl beobachteter Profilkombinationen: 37
- Größte einzelne Kombination: 8.0 %
- Top-5-Kombinationen: 33.5 %
- Top-10-Kombinationen: 56.2 %

### Test Predictions

- Anzahl vorhergesagter Profilkombinationen: 22
- Größte einzelne Kombination: 19.5 %
- Top-5-Kombinationen: 65.0 %
- Top-10-Kombinationen: 93.5 %

## Interpretation

Die Predictions sind viel stärker konzentriert als die echten Labels. Das ist ein sehr wichtiger Befund.

Die echte Testverteilung ist relativ breit: Die Top-10-Profilkombinationen decken 56.2 % ab. Die Predictions dagegen sind stark auf wenige Standardprofile verdichtet: Die Top-10-Kombinationen decken 93.5 % der Celebrities ab.

Die häufigsten vorhergesagten Profile sind:

| Predicted Profile | Anteil |
|---|---:|
| performer + female + 1985 | 19.5 % |
| politics + male + 1963 | 12.75 % |
| performer + male + 1985 | 12.5 % |
| sports + male + 1985 | 10.25 % |
| creator + female + 1975 | 10.0 % |
| sports + female + 1985 | 9.25 % |
| politics + male + 1975 | 8.75 % |
| performer + female + 1994 | 5.5 % |

Diese Kombinationen sind in den echten Testlabels jeweils viel seltener. Beispiel:

| Profil | True % | Pred % | Differenz |
|---|---:|---:|---:|
| performer + female + 1985 | 1.25 % | 19.50 % | +18.25 |
| performer + male + 1985 | 0.50 % | 12.50 % | +12.00 |
| politics + male + 1963 | 1.00 % | 12.75 % | +11.75 |
| creator + female + 1975 | 1.00 % | 10.00 % | +9.00 |
| sports + male + 1985 | 1.50 % | 10.25 % | +8.75 |

Das spricht dafür, dass die drei getrennten Modelle zusammen ein stark vereinfachtes Profilbild erzeugen. Anstatt die reale Vielfalt der Profile abzubilden, entstehen wenige stereotype Kombinationen.

---

## Fehler nach Occupation

Die Accuracy je Target im Testset:

| Target | Accuracy |
|---|---:|
| occupation | 66.8 % |
| gender | 68.0 % |
| birthyear-bin | 42.0 % |

### Accuracy nach wahrer Occupation

| True Occupation | Occupation Accuracy | Gender Accuracy | Birthyear-bin Accuracy |
|---|---:|---:|---:|
| creator | 28.0 % | 60.0 % | 27.0 % |
| performer | 87.0 % | 60.0 % | 51.0 % |
| politics | 82.0 % | 67.0 % | 34.0 % |
| sports | 70.0 % | 85.0 % | 56.0 % |

## Interpretation

`creator` ist der klare Problemfall. Nicht nur Occupation ist für Creator schwach, auch Birthyear ist bei Creator deutlich schwächer. Das deutet darauf hin, dass Creator-Follower-Feeds sehr heterogen sind oder von Signalen dominiert werden, die eher Performer, Politics oder allgemeine Plattformaktivität repräsentieren.

Sports ist dagegen relativ stabil: Gender und Birthyear sind hier am stärksten. Das passt zu der Annahme, dass Sports eine thematisch und community-strukturell klarere Klasse ist.

Performer ist in Occupation stark, aber Gender nur mittelmäßig. Das kann daran liegen, dass Performer-Follower-Feeds stark von Fan-Sprache geprägt sind, die geschlechtsübergreifend vorkommt oder bestimmte Gender-Signale überbetont.

Politics ist in Occupation stark, aber Birthyear bleibt schwierig. Das Modell erkennt offenbar Politics-Themen gut, aber kann daraus Alter nur grob ableiten.

---

## Woran könnte BERTweet Probleme haben?

### 1. Getrennte Modelle erzeugen inkonsistente oder stereotype Profile

Da Occupation, Gender und Birthyear separat trainiert wurden, gibt es keine gemeinsame Korrektur der drei Outputs. Dadurch kann eine Kombination entstehen, die einzeln plausibel wirkt, aber insgesamt zu stark stereotypisiert ist.

Beispiel:

```text
performer + female + 1985
politics + male + 1963
creator + female + 1975
sports + male/female + 1985
```

Diese Standardprofile dominieren die Predictions deutlich stärker als die echten Daten.

### 2. Birthyear wird vermutlich indirekt gelernt

Birthyear ist die schwächste Aufgabe. Die Predictions zeigen, dass Alter stark an Occupation gekoppelt wird:

- Sports → 1985
- Performer → 1985/1994
- Politics → 1963/1975
- Creator → 1975

Das kann helfen, wenn die Occupation korrekt ist, aber es führt zu systematischen Fehlern, wenn die Person innerhalb der Klasse älter oder jünger ist als das stereotype Profil.

### 3. Creator ist zu heterogen

Creator wird zu selten vorhergesagt und wenn es vorhergesagt wird, dann fast immer mit `female + 1975`. Das wirkt wie eine starke Modellverengung. Möglicherweise fehlen für Creator klare Textanker, oder die vorhandenen N-Gramme sind zu stark von Plattformphrasen, Fan-Sprache oder einzelnen Themenclustern geprägt.

### 4. Politics-Prior wird übernommen bzw. verstärkt

Im Training ist Politics deutlich männlich geprägt. Im Test ist Politics 50/50, aber in den Predictions wird Politics zu 81.4 % als männlich vorhergesagt. Das ist ein Hinweis auf einen übertragenden Trainingsprior.

---

## Potenzial für bessere Modellierung

## 1. Joint-Decoding mit Profil-Prior

Die einfachste Erweiterung ist ein Post-Processing-Schritt, der die drei Modellwahrscheinlichkeiten gemeinsam betrachtet:

```text
score(o, g, a) = log p(o | text) + log p(g | text) + log p(a | text) + λ log P(o, g, a)
```

Dabei ist `P(o, g, a)` die Profilkombination aus dem Trainingsdatensatz.

Vorteil:

- keine neue BERTweet-Trainingsrunde nötig
- kann unsichere Fälle stabilisieren
- kann sehr unwahrscheinliche Kombinationen abschwächen

Risiko:

- kann Dataset-Bias verstärken
- muss mit kleinem λ getestet werden
- sollte vor allem bei niedriger Modellkonfidenz wirken

Empfehlung: Joint-Decoding nur als optionales Re-Ranking verwenden, nicht als harte Regel.

---

## 2. Stacking / Meta-Classifier

Ein stärkerer Ansatz wäre ein Meta-Modell, das die Outputs aller drei Modelle kombiniert.

Features:

```text
occupation probabilities
+ gender probabilities
+ birthyear probabilities
+ confidence scores
+ entropy scores
+ style features
+ TF-IDF scores
```

Beispiel für eine verbesserte Occupation-Vorhersage:

```text
Input: p_occ + p_gender + p_birthyear + style
Output: corrected occupation
```

Das wäre besonders interessant für `creator`, weil Creator stark mit Fehlern in anderen Dimensionen zusammenhängt.

---

## 3. Multi-Task Learning

Die sauberste Deep-Learning-Erweiterung wäre ein gemeinsamer BERTweet-Encoder mit drei Heads:

```text
BERTweet Encoder
├── occupation head
├── gender head
└── birthyear head
```

Vorteil:

- gemeinsame Repräsentation für alle Profil-Dimensionen
- kann geteilte Community-Signale lernen
- verhindert teilweise, dass jede Aufgabe isoliert stereotype Outputs erzeugt

Nachteil:

- aufwändiger zu trainieren
- benötigt gute Loss-Gewichtung
- Birthyear kann andere Tasks über Rauschen stören, wenn es zu stark gewichtet wird

Eine mögliche Loss-Funktion:

```text
L = L_occupation + α L_gender + β L_birthyear
```

Dabei sollten α und β experimentell bestimmt werden.

---

## 4. Hybridmodell mit TF-IDF + Style + BERTweet

Die bisherigen N-Gramm-Analysen zeigen, dass manche Klassen klare lexikalische Signale haben:

- Sports: thematische Sportsignale
- Politics: politische Begriffe und öffentliche Themen
- Performer: Fan- und Entertainment-Sprache
- Creator: Plattform- und Content-Signale, aber weniger stabil

Ein Hybridmodell könnte diese Signale besser nutzen:

```text
BERTweet probabilities
+ TF-IDF word n-grams
+ character n-grams
+ style features
+ profile-prior features
```

Für TF-IDF besonders relevant:

- separate TF-IDF-Modelle pro Task
- saubere Tokenisierung ohne `@user`, `httpurl`, `rt`
- Log-Odds/Chi² Feature Selection
- spezielle Features für Creator/Performer-Fehler

---

## 5. Kalibrierung und Entropie-basierte Korrektur

Da die Prediction-Kombinationen stark konzentriert sind, wäre es sinnvoll, die Modellunsicherheit explizit zu nutzen.

Mögliche Features:

```text
max_probability
entropy
margin between top-1 and top-2
agreement between occupation/gender/age priors
```

Bei hoher Unsicherheit könnte ein Hybrid- oder Joint-Modell stärker eingreifen. Bei hoher Sicherheit sollte die BERTweet-Vorhersage weniger verändert werden.

---

## Konkrete nächste Experimente

### Experiment A: Label-Prior nur analysieren

Ziel: prüfen, wie stark Train/Test/Predict auseinanderliegen.

Output:

- P(gender | occupation)
- P(age | occupation)
- Top-Profilkombinationen
- Prediction minus True Distribution

Das ist bereits durch dieses Skript gut abgedeckt.

---

### Experiment B: Joint-Decoding mit kleinem λ

Teste:

```text
λ = 0.0, 0.1, 0.25, 0.5, 1.0
```

Metriken:

- Occupation Accuracy
- Gender Accuracy
- Birthyear Accuracy
- Joint Profile Accuracy
- Creator Recall

Wichtig: Creator Recall separat betrachten, weil er aktuell stark leidet.

---

### Experiment C: Meta-Classifier

Trainiere auf Validation/Train-Split einen Logistic Regression oder Random Forest Meta-Classifier.

Input:

```text
p_occ_4class + p_gender + p_birthyear + style_features
```

Ziel:

```text
final occupation
final gender
final birthyear
```

Oder zuerst nur:

```text
creator vs not_creator
```

---

### Experiment D: TF-IDF-Korrektur für Creator/Performer

Da Creator der schwächste Punkt ist:

```text
Trainiere ein Zusatzmodell nur auf Fällen, die BERTweet als performer/creator einordnet.
```

Input:

```text
clean TF-IDF n-grams + style features + BERTweet probabilities
```

Output:

```text
creator vs performer
```

Das ist wahrscheinlich der pragmatischste Weg, um direkt Performance zu verbessern.

---

## Fazit

Die Profilanalyse zeigt ein klares Potenzial: Die drei Targets sind nicht unabhängig. Occupation, Gender und Birthyear bilden zusammen ein Profil, und das Modell erzeugt bereits implizit solche Profile. Allerdings sind die vorhergesagten Profile deutlich stärker konzentriert und stereotyper als die echten Labels.

Das ist gleichzeitig Problem und Chance:

- Problem: BERTweet verstärkt bestimmte Profilkombinationen wie `performer + female + 1985` oder `politics + male + 1963`.
- Chance: Diese Struktur kann kontrolliert genutzt werden, um unsichere Vorhersagen zu stabilisieren.

Für die Bachelorarbeit ist besonders interessant:

> Ein Hybridansatz aus BERTweet, TF-IDF, Style-Features und kontrolliertem Profil-Prior könnte die Schwächen der separaten Single-Task-Modelle reduzieren, insbesondere bei Creator/Performer und Birthyear.

Wichtig ist, den Profil-Prior nicht blind als Wahrheit zu verwenden. Er sollte als weiches Zusatzsignal modelliert und gegen Bias-Verstärkung abgesichert werden.
