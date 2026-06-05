# Analyse: BERTweet V3 Raw vs. BERTweet V3.4 Stopword-gefiltert

## 1. Ziel der Analyse

Mit BERTweet V3.4 wurde eine Stopword-gefilterte Tokenizer-Variante getestet. Die zentrale Frage ist:

> Hilft Stopword-Filtering direkt im BERTweet-Input, oder ist Stopword-Filtering eher als separater TF-IDF-/Analyse-Zweig geeignet?

Für einen fairen Vergleich wurde V3.4 als Ablation zu V3 betrachtet:

- **BERTweet V3:** raw Twitter-nahe Eingabe
- **BERTweet V3.4:** stopword-gefilterte Eingabe
- gleiche Grundarchitektur
- gleiche BERTweet-Basis (`vinai/bertweet-base`)
- gleiche Zielstruktur für `occupation`, `gender`, `birthyear`
- zusätzliche 8-Range-Variante für `birthyear_8range`

---

## 2. Gesamtvergleich

| Target | Modell | Accuracy | Macro-F1 | Veränderung Macro-F1 |
|---|---:|---:|---:|---:|
| occupation | V3 raw | 0.6675 | 0.6493 | — |
| occupation | V3.4 stopwords | 0.6375 | 0.6225 | -0.0268 |
| gender | V3 raw | 0.6800 | 0.6799 | — |
| gender | V3.4 stopwords | 0.7000 | 0.6997 | +0.0199 |
| birthyear 5-class | V3 raw | 0.4200 | 0.3233 | — |
| birthyear 5-class | V3.4 stopwords | 0.4550 | 0.3651 | +0.0418 |
| birthyear 8-range | V3.4 stopwords | 0.2825 | 0.1946 | zusätzliche Ablation |

## 3. Zentrales Ergebnis

Stopword-Filtering wirkt **nicht einheitlich** über alle Targets.

Es zeigt sich ein klares target-spezifisches Muster:

```text
occupation  → schlechter
gender      → besser
birthyear   → besser
8-range age → methodisch interessant, aber für BERTweet allein zu schwer
```

Das ist ein wichtiges Learning: Stopword-Filtering ist keine generell bessere Vorverarbeitung, sondern verändert die Art der Signale, die BERTweet nutzen kann.

---

# 4. Occupation

## 4.1 Ergebnis

| Klasse | V3 F1 | V3.4 F1 | Veränderung |
|---|---:|---:|---:|
| sports | 0.7735 | 0.7528 | -0.0207 |
| performer | 0.6877 | 0.6767 | -0.0111 |
| creator | 0.3660 | 0.3396 | -0.0264 |
| politics | 0.7700 | 0.7208 | -0.0491 |

V3.4 verschlechtert Occupation insgesamt:

```text
Accuracy:  0.6675 → 0.6375
Macro-F1:  0.6493 → 0.6225
```

## 4.2 Auffällige Confusion-Muster

### V3 raw

```text
True creator → performer: 40
True creator → politics: 22
True creator → creator: 28
```

### V3.4 stopwords

```text
True creator → performer: 46
True creator → politics: 17
True creator → creator: 27
```

Creator bleibt also die zentrale Problemklasse. Durch Stopword-Filtering wird Creator sogar noch etwas stärker in Richtung Performer gezogen.

Auch Politics leidet:

```text
V3 politics recall:   0.82
V3.4 politics recall: 0.71
```

Bei V3.4 werden deutlich mehr Politics-Fälle als Creator klassifiziert:

```text
True politics → creator:
V3:   14
V3.4: 25
```

## 4.3 Interpretation

Bei Occupation scheinen Stopwords und Funktionswörter für BERTweet nicht nur Rauschen zu sein. Sie können Teil der Kontextstruktur sein, die BERTweet beim Verständnis von Themen, Rollen und Community-Sprache nutzt.

Die Verschlechterung spricht dafür, dass ein Transformer wie BERTweet von natürlicher Satzstruktur profitiert. Stopword-Filtering macht den Input künstlicher:

```text
raw:     "thank you for watching my new video"
cleaned: "thank watching new video"
```

Für TF-IDF ist die zweite Variante oft gut. Für BERTweet kann sie aber unnatürlich sein.

## 4.4 Learning für Occupation

Für `occupation` sollte BERTweet wahrscheinlich **raw** bleiben.

Stopword-gefilterte Texte sind trotzdem wertvoll, aber eher als:

```text
clean TF-IDF-Zweig
N-Gramm-Analyse
Sparse Features
Hybrid-Feature
```

Nicht als alleinige BERTweet-Eingabe.

## 4.5 Potenzial

Für Occupation bietet sich ein Hybridmodell besonders an:

```text
BERTweet raw probabilities
+ TF-IDF word n-grams mit Stopword-Filtering
+ TF-IDF char n-grams
+ Style Features
+ optional creator_binary probability
```

Warum?

- BERTweet raw hält Kontext und natürliche Twitter-Struktur.
- TF-IDF clean kann Themenmarker besser isolieren.
- Style Features behalten Social-Media-Signale wie Mentions, Links, Hashtags, Emojis.
- Creator bleibt als Problemklasse gezielt adressierbar.

---

# 5. Gender

## 5.1 Ergebnis

| Klasse | V3 F1 | V3.4 F1 | Veränderung |
|---|---:|---:|---:|
| male | 0.6735 | 0.6907 | +0.0173 |
| female | 0.6863 | 0.7087 | +0.0225 |

V3.4 verbessert Gender:

```text
Accuracy:  0.6800 → 0.7000
Macro-F1:  0.6799 → 0.6997
```

Die Confusion-Matrix verbessert sich leicht auf beiden Seiten:

```text
Male korrekt:
V3:   132/200
V3.4: 134/200

Female korrekt:
V3:   140/200
V3.4: 146/200
```

## 5.2 Interpretation

Das ist interessant, weil man zunächst erwarten könnte, dass Stopwords für Gender wichtig sind. Author-Profiling nutzt oft Stilmerkmale wie Pronomen, Funktionswörter und Satzmuster.

Dass V3.4 trotzdem besser wird, deutet auf zwei mögliche Effekte hin:

1. Die entfernten Stopwords waren bei diesem Datensatz stärker Rauschen als Signal.
2. Die behaltenen Signale, etwa Hashtags, Namen, Themen, Fan-Sprache und Social-Media-Kontext, reichen für Gender-Profiling aus oder werden sogar klarer.

Wichtig ist: In V3.4 wurden nicht alle potenziell relevanten Funktionswörter aggressiv entfernt. Wenn Pronomen und Negationen erhalten bleiben, kann ein Teil der stilistischen Information weiterhin vorhanden sein.

## 5.3 Learning für Gender

Gender profitiert in diesem Experiment von Stopword-Filtering.

Das bedeutet aber nicht automatisch, dass Stopwords bei Gender generell unwichtig sind. Es bedeutet eher:

> In dieser Follower-Feed-Aufgabe scheinen viele generische Funktionswörter die Klassifikation eher zu verwässern, während bereinigtere Content- und Community-Signale hilfreicher werden.

## 5.4 Potenzial

Für Gender lohnt sich ein Vergleich dieser Varianten:

```text
BERTweet raw
BERTweet stopword-filtered
TF-IDF clean
TF-IDF clean + style
BERTweet raw + BERTweet stopword probabilities
```

Gerade weil V3 und V3.4 unterschiedliche Inputs sehen, könnten ihre Wahrscheinlichkeiten komplementär sein.

Ein kleines Meta-Modell könnte nutzen:

```text
p_male_raw
p_female_raw
p_male_stopword
p_female_stopword
emoji_count
hashtag_count
mention_count
url_count
avg_tweet_length
```

---

# 6. Birthyear 5-class

## 6.1 Ergebnis

| Klasse | V3 F1 | V3.4 F1 | Veränderung |
|---|---:|---:|---:|
| 1994 | 0.3704 | 0.5357 | +0.1653 |
| 1985 | 0.5917 | 0.6194 | +0.0276 |
| 1975 | 0.2809 | 0.2222 | -0.0587 |
| 1963 | 0.3733 | 0.4483 | +0.0749 |
| 1947 | 0.0000 | 0.0000 | ±0.0000 |

V3.4 verbessert Birthyear deutlich:

```text
Accuracy:  0.4200 → 0.4550
Macro-F1:  0.3233 → 0.3651
```

## 6.2 Wichtigste Verbesserung: 1994

Bei V3 wurde die jüngste Klasse sehr oft zur mittleren Klasse `1985` gezogen:

```text
V3: true 1994 → pred 1985 = 38
```

Bei V3.4 reduziert sich dieser Fehler:

```text
V3.4: true 1994 → pred 1985 = 23
```

Gleichzeitig steigt die korrekte Erkennung von 1994:

```text
V3:   15/56 korrekt
V3.4: 30/56 korrekt
```

Das ist ein starkes Signal: Stopword-Filtering hilft offenbar, jüngere Follower-/Fandom-/Community-Signale klarer sichtbar zu machen.

## 6.3 Problem bleibt: 1947

Die älteste Klasse bleibt ungelöst:

```text
V3 1947 F1:   0.0
V3.4 1947 F1: 0.0
```

V3.4 sagt praktisch nie korrekt `1947` vorher. Viele 1947-Fälle werden weiterhin in Richtung `1975` oder `1963` gezogen.

Das deutet darauf hin, dass das Modell ältere Celebrities über Follower-Feeds nicht sauber von mittelalten Celebrities unterscheiden kann. Mögliche Gründe:

- ältere Celebrities haben trotzdem jüngere oder gemischte Follower-Communities
- alte Celebrities werden über aktuelle Themen diskutiert
- Follower-Sprache spiegelt eher heutige Plattformkultur als tatsächliches Celebrity-Alter
- die Klasse 1947 hat weniger robuste Textsignale
- die Aufgabe ist ordinal, wird aber als normale Mehrklassenklassifikation behandelt

## 6.4 Learning für Birthyear

Stopword-Filtering ist bei Birthyear nützlich, aber nicht ausreichend.

Es reduziert teilweise die Tendenz zur dominanten Mittelklasse, verbessert die jüngste Klasse und hilft auch 1963. Die älteste Klasse bleibt aber ein strukturelles Problem.

Birthyear sollte wahrscheinlich nicht nur als flache Klassifikation modelliert werden.

## 6.5 Potenzial

Für Birthyear sind diese Ansätze besonders vielversprechend:

```text
1. ordinal classification
2. regression auf Birthyear
3. hierarchical age prediction
4. age direction / young vs old auxiliary task
5. Hybrid mit occupation/gender-priors
6. TF-IDF + style features
```

Besonders sinnvoll wäre ein zweistufiges Modell:

```text
Schritt 1: grob jung / mittel / alt
Schritt 2: feinere Klasse innerhalb der groben Gruppe
```

Für 1947 könnte ein zusätzlicher Old-vs-Rest-Classifier helfen.

---

# 7. Birthyear 8-range

## 7.1 Ergebnis

Die 8-Range-Variante erreicht:

```text
Accuracy:  0.2825
Macro-F1:  0.1946
Weighted-F1: 0.2202
```

Die verwendeten Ranges waren:

```text
age_bin_0: <= 1957
age_bin_1: 1958-1965
age_bin_2: 1966-1971
age_bin_3: 1972-1975
age_bin_4: 1976-1980
age_bin_5: 1981-1985
age_bin_6: 1986-1989
age_bin_7: >= 1990
```

## 7.2 Auffälligkeit

Das Modell erkennt vor allem Randbereiche:

```text
age_bin_0 F1: 0.5029
age_bin_7 F1: 0.3868
```

Mittlere Klassen fallen deutlich schwächer aus:

```text
age_bin_2 F1: 0.0
age_bin_4 F1: 0.0
```

## 7.3 Interpretation

Die 8-Range-Aufteilung ist methodisch interessant, weil sie näher an Koloski et al. liegt und die Altersverteilung feiner modelliert. Für BERTweet allein scheint sie aber zu granular zu sein.

Das Modell erkennt grobe Altersränder eher als feine Übergänge:

```text
sehr alt / sehr jung → teilweise erkennbar
mittlere Altersbereiche → stark überlappend
```

Das passt zur Natur der Aufgabe: Follower-Feeds enthalten wahrscheinlich keine exakten Altersmarker, sondern eher indirekte Community-Signale.

## 7.4 Learning

8 Ranges sind als Vergleich und Ablation sinnvoll, aber sollten nicht automatisch als finales Age-Modell gewählt werden.

Für die Arbeit kann man formulieren:

> Die feinere 8-Klassen-Aufteilung ist methodisch fairer gegenüber Koloski et al., zeigt aber, dass BERTweet allein die mittleren Altersbereiche nicht stabil trennen kann. Die Ergebnisse sprechen eher für ordinales oder hierarchisches Age-Modelling als für eine flache 8-Klassen-Klassifikation.

---

# 8. Übergreifende Insights

## 8.1 Stopword-Filtering ist target-abhängig

Der wichtigste Befund ist:

```text
Stopword-Filtering hilft Gender und Birthyear,
schadet aber Occupation.
```

Das spricht gegen eine globale Preprocessing-Regel.

Stattdessen sollte das System mehrere Textrepräsentationen nutzen:

```text
raw_text für BERTweet
clean_text für TF-IDF
style/social features separat
```

## 8.2 BERTweet braucht natürliche Sprache für Occupation

Occupation ist semantisch und kontextuell. Gerade die Abgrenzung von Creator, Performer und Politics profitiert vermutlich von Satzstruktur und Kontext.

Stopword-Filtering kann hier wichtige Verbindungen entfernen.

## 8.3 Für Profiling können bereinigte Community-Signale helfen

Gender und Birthyear werden nicht nur durch Semantik bestimmt, sondern wahrscheinlich durch:

```text
Fan-Sprache
Follower-Community
Plattformnutzung
Namen und Entitäten
Hashtags
Emojis
Themencluster
```

Stopword-Filtering kann diese Signale sichtbarer machen.

## 8.4 Die älteste Age-Klasse bleibt ein strukturelles Problem

Unabhängig von V3 oder V3.4 wird `1947` nicht erkannt. Das ist kein einfaches Stopword-Problem, sondern vermutlich ein Problem der Labelstruktur und der indirekten Beobachtbarkeit von Alter über Follower-Feeds.

## 8.5 8-Range-Age ist fair, aber schwer

Die 8-Range-Variante ist wertvoll für den Literaturvergleich, aber als reine BERTweet-Klassifikation nicht stark genug. Sie zeigt, dass feinere Age-Klassen stark überlappen.

---

# 9. Konsequenzen für die Modellstrategie

## 9.1 Kein Ersatz von V3 durch V3.4 für alle Targets

V3.4 sollte nicht pauschal als neues Gesamtmodell verwendet werden.

Empfehlung:

| Target | Besseres BERTweet-Modell | Begründung |
|---|---|---|
| occupation | V3 raw | bessere Accuracy und Macro-F1 |
| gender | V3.4 stopwords | bessere Accuracy und Macro-F1 |
| birthyear 5-class | V3.4 stopwords | bessere Accuracy und Macro-F1 |
| birthyear 8-range | V3.4 als Ablation | methodisch interessant, aber schwächer |

## 9.2 Hybridmodell als nächster logischer Schritt

Die Ergebnisse sprechen stark für ein Hybridmodell:

```text
BERTweet raw
+ BERTweet stopword-filtered probabilities
+ TF-IDF clean word n-grams
+ TF-IDF char n-grams
+ style/social features
+ optional profile-priors
```

Warum?

- Raw BERTweet ist stark für semantisch-kontextuelle Occupation.
- Stopword-BERTweet bringt Zusatzsignale für Gender und Age.
- TF-IDF kann diskrete Topic- und Fandom-Marker erfassen.
- Style Features retten Social-Media-Signale, ohne `@USER` und `HTTPURL` als N-Gramme dominieren zu lassen.
- Profil-Priors können Zusammenhänge zwischen Occupation, Gender und Birthyear modellieren.

---

# 10. Konkrete nächste Experimente

## Experiment A: Target-spezifische BERTweet-Auswahl

```text
occupation: V3 raw
gender: V3.4 stopwords
birthyear: V3.4 stopwords
```

Das ist eine pragmatische Best-of-BERTweet-Variante.

## Experiment B: Raw + Stopword Ensemble

Für jedes Target:

```text
input features:
- V3 class probabilities
- V3.4 class probabilities

model:
- Logistic Regression
- small MLP
- calibrated weighted average
```

Hypothese:

> V3 und V3.4 machen unterschiedliche Fehler. Ein Meta-Modell könnte diese komplementären Signale nutzen.

## Experiment C: TF-IDF clean

Trainiere ein klassisches Modell auf bereinigtem Text:

```text
word n-grams: 1–3
char n-grams: 3–5
classifier: Logistic Regression / Linear SVM
```

Besonders interessant für:

```text
occupation: sports/politics/topic markers
creator/performer/fandom markers
birthyear: age/community/fandom markers
```

## Experiment D: Style Features

Ergänze numerische Features:

```text
mention_count
url_count
hashtag_count
emoji_count
retweet_ratio
avg_tweet_length
uppercase_ratio
exclamation_count
question_count
```

Diese Features sollten nicht als N-Gramme dominieren, aber als eigene Profiling-Signale erhalten bleiben.

## Experiment E: Age ordinal / hierarchical

Für Birthyear:

```text
1. old vs not-old classifier
2. young vs middle vs old classifier
3. fine-grained classification innerhalb der Grobgruppe
```

Oder:

```text
ordinal regression / CORAL-style ordinal classification
```

---

# 11. Mögliche Formulierung für die Bachelorarbeit

> Die Stopword-gefilterte BERTweet-Variante V3.4 zeigt, dass Preprocessing-Entscheidungen stark vom Zielattribut abhängen. Während die Entfernung von Stopwords die Klassifikation von Gender und Birthyear verbessert, verschlechtert sie die Occupation-Erkennung. Dies legt nahe, dass Occupation stärker von natürlicher Satzstruktur und kontextuellen semantischen Hinweisen profitiert, während Gender und Birthyear in den Follower-Feeds stärker über bereinigte Community- und Themenmarker erschlossen werden können. Die Ergebnisse sprechen daher gegen ein einheitliches Preprocessing für alle Targets und motivieren ein Hybridmodell, das rohe BERTweet-Repräsentationen, bereinigte TF-IDF-Features und Social-/Style-Merkmale kombiniert.

---

# 12. Kurzfazit

V3.4 ist ein sehr wertvolles Experiment.

Es zeigt:

```text
Stopword-Filtering ist nicht einfach gut oder schlecht.
Es verschiebt die Modellperspektive.
```

Für deine Arbeit ist das ein starkes Learning:

- BERTweet raw bleibt wichtig für kontextuelle Semantik.
- Stopword-gefiltertes BERTweet kann Gender und Age verbessern.
- Clean Tokenization ist besonders wertvoll als zusätzlicher Feature-Zweig.
- Age braucht wahrscheinlich andere Modellierungsformen als flache Klassifikation.
- Der nächste überzeugende Schritt ist ein Hybridmodell.
