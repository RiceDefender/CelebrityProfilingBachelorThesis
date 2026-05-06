# Gender Signal Analysis – BERTweet Prediction Groups

## 1. Ziel der Analyse

Diese Analyse untersucht, welche sprachlichen Muster in den Follower-Feeds mit den Gender-Vorhersagen des BERTweet-Modells zusammenfallen. Dabei werden Test- und Validation-Split gemeinsam betrachtet.

Wichtig: Die Analyse ist **keine direkte Attention- oder Gradienten-Erklärung** des Modells. Die Tabellen zeigen, welche N-Gramme und Stilmerkmale in bestimmten Vorhersagegruppen häufig oder überrepräsentiert sind. Sie liefern daher Hinweise auf mögliche Signale, Störfaktoren und Fehlerquellen auf Eingabeebene.

---

## 2. Erste Auffälligkeit: Raw-N-Gramme werden von Twitter-Artefakten dominiert

In den Raw-N-Gramm-Tabellen dominieren fast überall Muster wie:

```text
rt @user
@user @user
@user @user @user
rt @user @user
of the
```

Diese Muster sind für Twitter-Daten erwartbar, aber sie verdecken die eigentlich interessanten thematischen und stilistischen Signale. Für die Interpretation von N-Grammen sind deshalb die `nostop`-Tabellen sinnvoller. Mentions und URLs sollten jedoch nicht vollständig verworfen werden, weil sie als **Style-Features** weiterhin informativ sein können.

**Empfehlung:** Für N-Gramm-Visualisierungen sollten `@user`, `HTTPURL`, `rt`, `via`, `amp` und ähnliche Tokens ausgeblendet werden. Für Style-Features sollten Counts wie `mention_count`, `url_count` und `hashtag_count` erhalten bleiben.

---

## 3. Correct-by-Class: Was sieht man bei korrekten Gender-Vorhersagen?

### 3.1 Test-Split

Bei korrekt als `female` vorhergesagten Fällen erscheinen im Test-Split häufig Fan-, Voting- und Popkultur-Signale:

| Gruppe | Auffällige N-Gramme |
|---|---|
| Correct female high | `happy birthday`, `love gina`, `#mtvstars seconds`, `can't wait`, `well done`, `i'm going`, `good luck` |
| Correct female high, Trigramme | `#mtvstars seconds summer`, `tweet gets rts`, `i'm going tweet`, `votes #mtvstars seconds` |

Bei korrekt als `male` vorhergesagten Fällen erscheinen im Test-Split stärker technische, politische, Crypto- oder Plattform-/Posting-Signale:

| Gruppe | Auffällige N-Gramme |
|---|---|
| Correct male high | `access control`, `check muhammad`, `video #tiktok`, `muhammad nasar`, `imran khan`, `prime minister`, `posted new`, `new photo` |
| Correct male high, Log-Odds | `#trx trx`, `book network`, `#tron #trx`, `asad umar`, `daily transactions` |

**Interpretation:** Das Modell scheint bei Gender nicht nur geschlechtsspezifische Sprache zu erfassen, sondern stark auf **Community- und Themencluster** zu reagieren. `female` korreliert im Test-Split stärker mit Fan-/Voting-Sprache, während `male` stärker mit Politik-, Tech-, Crypto- und Nachrichtenclustern korreliert.

---

### 3.2 Validation-Split

Im Validation-Split ist dieses Muster ähnlich, aber nicht identisch.

Bei korrekt als `female` vorhergesagten Fällen dominieren wieder Fan-, Voting- und Popkultur-Signale:

| Gruppe | Auffällige N-Gramme |
|---|---|
| Correct female high | `retweet vote`, `#peopleschoice retweet`, `demi lovato`, `voted demi`, `lovato #celebrityjudge`, `#popartist #peopleschoice` |
| Correct female high, Trigramme | `#peopleschoice retweet vote`, `voted demi lovato`, `demi lovato #celebrityjudge` |

Bei korrekt als `male` vorhergesagten Fällen dominieren im Validation-Split sehr spezifische Themencluster:

| Gruppe | Auffällige N-Gramme |
|---|---|
| Correct male high | `solid waste`, `dumping ground`, `kibao vodka`, `dam without`, `river technology`, `new job`, `job alert` |
| Correct male high, Trigramme | `dam without river`, `without river technology`, `new job alert`, `solid waste soil` |

**Interpretation:** Das `female`-Signal wirkt zwischen Test und Validation relativ konsistent: Fan-/Voting-/Popkultur-Sprache. Das `male`-Signal ist dagegen instabiler und wirkt stärker abhängig von einzelnen dominanten Themenclustern. Das spricht dafür, dass N-Gramme allein bei Gender leicht **Dataset- und Account-Artefakte** lernen können.

---

## 4. False-by-Prediction und Confusion-Pairs

### 4.1 Female → Male

Diese Gruppe enthält eigentlich weibliche Celebrities, die als `male` vorhergesagt wurden.

Im Test-Split treten unter anderem auf:

```text
video #tiktok
check muhammad
muhammad nasar
nasar video
story pls
pls revert
posted new photo
new photo facebook
```

Im Validation-Split treten sehr starke politische/geopolitische Cluster auf:

```text
commy regime
korea must
korean leader
leader moon
free speech
must banned
korean dictator
korean leader moon
korea must sanctioned
must sanctioned strictly
```

**Interpretation:** Female-Profile werden besonders dann als `male` klassifiziert, wenn die Follower-Feeds stark nach Politik, Aktivismus, Nachrichten, Technik oder stark repetitiven Plattform-/Posting-Mustern aussehen. Das ist kein echtes Gender-Signal, sondern vermutlich ein **Themen- oder Community-Bias**.

---

### 4.2 Male → Female

Diese Gruppe enthält eigentlich männliche Celebrities, die als `female` vorhergesagt wurden.

Im Test-Split treten auf:

```text
weekend shopping
happy birthday
follow back
can't wait
make sure
feel like
dm follow
shopping weekend
```

Die Trigramme zeigen stark repetitive Fan-/Giveaway-/Voting-Sprache:

```text
weekend shopping weekend
shopping dm follow
dm follow make
follow make sure
make sure retweet
sure retweet win
#bestfanarmy #limelights #iheartawards
```

Im Validation-Split treten ähnliche weichere Fan-/Pop-/Alltagssignale auf:

```text
happy birthday
can't wait
last night
looks like
looking forward
follow back
get follow back
follow back please
can't wait see
```

**Interpretation:** Male-Profile werden eher als `female` klassifiziert, wenn die Follower-Feeds viel Fan-Sprache, soziale Interaktion, Follow-back-/Voting-Muster oder emotionale Alltagssprache enthalten. Auch das ist wahrscheinlich kein direktes Gender-Signal, sondern ein **Community-Stil-Signal**.

---

## 5. Style-Features

Die Style-Features zeigen ein gemischtes Bild. Die Unterschiede sind nicht riesig, aber es gibt wiederkehrende Tendenzen.

### 5.1 Test-Split

| Merkmal | Pred female | Pred male | Auffälligkeit |
|---|---:|---:|---|
| avg_chars_per_tweet | 93.30 | 102.88 | `male`-Predictions haben längere Tweets |
| avg_tokens_per_tweet | 13.83 | 14.90 | `male`-Predictions haben mehr Tokens |
| mention_count | 932.38 | 1148.11 | `male`-Predictions haben mehr Mentions |
| hashtag_count | 407.79 | 454.06 | `male`-Predictions haben etwas mehr Hashtags |
| emoji_count | 230.89 | 188.54 | `female`-Predictions haben mehr Emojis |
| exclamation_count | 314.50 | 214.91 | `female`-Predictions haben deutlich mehr Ausrufezeichen |
| love_word_count | 48.57 | 33.57 | `female`-Predictions haben mehr Love-/Emotion-Wörter |
| politics_word_count | 15.61 | 26.53 | `male`-Predictions haben mehr Politik-Wörter |
| sports_word_count | 37.98 | 42.43 | `male`-Predictions leicht höher |

### 5.2 Validation-Split

| Merkmal | Pred female | Pred male | Auffälligkeit |
|---|---:|---:|---|
| avg_chars_per_tweet | 96.05 | 102.98 | `male`-Predictions wieder länger |
| avg_tokens_per_tweet | 14.26 | 15.16 | `male`-Predictions wieder mehr Tokens |
| mention_count | 939.06 | 1066.02 | `male`-Predictions mehr Mentions |
| hashtag_count | 429.39 | 390.54 | Unterschied hier nicht stabil |
| emoji_count | 151.73 | 168.52 | Unterschied kippt im Validation-Split |
| exclamation_count | 321.06 | 279.40 | `female`-Predictions weiterhin mehr Ausrufezeichen |
| love_word_count | 41.83 | 33.48 | `female`-Predictions weiterhin mehr Love-Wörter |
| fan_word_count | 15.82 | 26.19 | `male`-Predictions höher im Validation-Split |
| politics_word_count | 26.10 | 31.49 | `male`-Predictions höher |
| sports_word_count | 48.71 | 56.36 | `male`-Predictions höher |

### 5.3 Was ist stabil?

Stabil über Test und Validation sind vor allem:

```text
Pred male  → längere Tweets, mehr Tokens, mehr Mentions, mehr Politik-/Sport-Wörter
Pred female → mehr Ausrufezeichen, mehr Love-/Emotion-Wörter
```

Nicht stabil sind:

```text
Emoji-Count
Hashtag-Count
Fan-Word-Count
URL-Count
```

**Interpretation:** Gender wird offenbar nicht nur über einzelne Wörter, sondern auch über Kommunikationsstil und Community-Typ sichtbar. Die stabileren Style-Signale könnten für ein Hybridmodell nützlich sein. Gleichzeitig sind einige Features split-abhängig und sollten vorsichtig verwendet werden.

---

## 6. Woran könnte BERTweet Probleme haben?

### 6.1 Gender-Signal ist indirekt

Die Tabellen zeigen kaum direkte Gender-Signale. Stattdessen erscheinen:

```text
Fan-Voting-Sprache
Popkultur-Fandoms
Politik-/News-Cluster
Crypto-/Tech-Cluster
Follow-back-/Giveaway-Muster
repetitive Kampagnen-Hashtags
```

Das Modell lernt daher vermutlich nicht „Gender“ im engeren Sinn, sondern Eigenschaften der Follower-Community. Da die Aufgabe follower-basiert ist, ist das grundsätzlich erwartbar, aber es erhöht die Gefahr, dass einzelne Community-Artefakte zu stark gewichtet werden.

---

### 6.2 Einzelne Cluster dominieren die N-Gramme

Auffällige Beispiele:

```text
#mtvstars seconds summer
#peopleschoice retweet vote
demi lovato #celebrityjudge
check muhammad nasar
korean leader moon
commy regime
solid waste soil
#tron #trx trx
```

Solche Phrasen sind sehr spezifisch. Sie können im Modell als starke Shortcuts wirken, generalisieren aber wahrscheinlich schlecht.

---

### 6.3 False male und False female folgen unterschiedlichen Fehlerlogiken

`female → male` scheint vor allem durch Politik-/Tech-/News-/Posting-Cluster ausgelöst zu werden.

`male → female` scheint eher durch Fan-, Follow-back-, Voting- und emotionale Alltagssprache ausgelöst zu werden.

Das deutet auf eine asymmetrische Fehlerstruktur hin: Das Modell verwechselt nicht zufällig, sondern folgt bestimmten Community-Signalen.

---

### 6.4 BERTweet kann Twitter-Artefakte aufnehmen, aber schwer erklären

BERTweet ist auf Twitter-Sprache vortrainiert und kann Mentions, URLs, Hashtags, Emojis und Retweets grundsätzlich gut verarbeiten. Für die Erklärung sind diese Elemente aber problematisch: In Raw-N-Grammen verdrängen sie fast alles andere.

Daher sollte man unterscheiden zwischen:

```text
Modellinput: Twitter-Artefakte behalten
Analyse/TF-IDF: Artefakte kontrolliert filtern oder separat zählen
```

---

## 7. Potenziale für bessere Tokenisierung

Ein sauberer Analyse-Tokenizer könnte helfen, ohne die BERTweet-Inputs aggressiv zu verändern.

### 7.1 Für N-Gramme entfernen oder separat behandeln

```text
@user
HTTPURL
rt
via
amp
gt
lt
```

Diese Tokens sollten für N-Gramm-Visualisierungen ausgeblendet werden. Als Style-Features können ihre Counts erhalten bleiben.

### 7.2 Hashtags differenzieren

Hashtags sind teilweise sehr informativ:

```text
#mtvstars
#peopleschoice
#celebrityjudge
#trx
#tron
#tiktok
```

Sie sollten nicht pauschal entfernt werden. Sinnvoll wären zwei Varianten:

```text
Variante A: Hashtags als Token behalten
Variante B: Hashtag-Text normalisieren, z. B. #PeoplesChoice → peopleschoice
```

Optional könnte man Hashtag-Cluster separat zählen.

### 7.3 Repetitive Kampagnen-Phrasen begrenzen

Viele Top-N-Gramme stammen aus stark repetitiven Fan-Kampagnen. Für TF-IDF kann das nützlich sein, aber auch überdominant. Eine Begrenzung pro Celebrity könnte helfen:

```text
maximaler N-Gramm-Count pro Celebrity
sublinear_tf=True
min_df / max_df
```

---

## 8. Potenzial für TF-IDF

Ein TF-IDF-Zweig ist für Gender interessant, aber sollte vorsichtig konstruiert werden.

### 8.1 Gute Kandidaten für TF-IDF

```text
word n-grams 1–3
char n-grams 3–5
hashtags
normalisierte Content-Wörter ohne @user/HTTPURL
sublinear TF-IDF
max_df zur Entfernung sehr generischer Twitter-Phrasen
```

### 8.2 Warum TF-IDF helfen kann

BERTweet verarbeitet die Feeds semantisch, aber seine Fehleranalyse ist schwer interpretierbar. TF-IDF kann ergänzen:

```text
Welche Community-Cluster sprechen für male/female?
Welche Phrasen treiben False male / False female?
Welche Features sind stabil über Validation und Test?
```

Besonders `logodds` und TF-IDF könnten helfen, Gender-Signale als explainable side-channel sichtbar zu machen.

### 8.3 Risiko bei TF-IDF

Die Tabellen zeigen viele sehr spezifische Phrasen. Ein reiner TF-IDF-Klassifikator könnte stark auf Namen, Kampagnen und einzelne Accounts overfitten.

Daher sollte TF-IDF nicht blind mit riesigem Vocabulary genutzt werden. Sinnvoller:

```text
min_df erhöhen
max_df setzen
sublinear_tf=True
max_features begrenzen
L2-regularisierte Logistic Regression oder Linear SVM
Cross-Validation nach Celebrity, nicht nach Tweet
```

---

## 9. Potenzial für Hybridmodell

Ein Hybridmodell wirkt besonders sinnvoll, weil Gender-Signale sowohl aus BERTweet-Semantik als auch aus Stil-/Community-Mustern bestehen.

Mögliche Feature-Fusion:

```text
BERTweet logits oder embeddings
+ TF-IDF word/char n-grams
+ Style-Features
+ Hashtag-Features
+ Retweet/Mention/URL counts
```

### 9.1 Besonders nützliche Style-Features

Auf Basis der Tabellen wirken diese Features am ehesten stabil:

```text
avg_chars_per_tweet
avg_tokens_per_tweet
mention_count
exclamation_count
love_word_count
politics_word_count
sports_word_count
```

Weniger stabil, aber trotzdem prüfenswert:

```text
emoji_count
hashtag_count
url_count
fan_word_count
```

### 9.2 Warum Hybrid helfen könnte

BERTweet kann semantische Kontexte verarbeiten, aber ein Style-/TF-IDF-Zweig kann explizit kontrollieren, ob eine Prediction stark von Community-Artefakten geprägt ist. Außerdem kann man bei Fehlern prüfen, ob BERTweet und TF-IDF unterschiedliche Signale sehen.

Beispielhafte Nutzung:

```text
Wenn BERTweet female vorhersagt, aber TF-IDF starke Politics-/Tech-/Sports-Signale zeigt, ist die Entscheidung potenziell unsicher.
Wenn BERTweet und Style-Features übereinstimmen, steigt die Confidence.
```

---

## 10. Konkrete nächste Experimente

### Experiment 1: Analyse-Tokenizer verbessern

N-Gramme erneut erzeugen mit:

```text
--remove-stopwords
--drop-social-ngram-tokens
--drop-rt-artifacts
```

Ziel: Prüfen, ob nach Entfernen von `@user`, `HTTPURL` und `rt` stabilere Gender-Signale sichtbar werden.

---

### Experiment 2: TF-IDF Gender-Baseline

Trainiere eine einfache Baseline:

```text
TF-IDF word n-grams 1–3
char n-grams 3–5
Logistic Regression oder Linear SVM
sublinear_tf=True
min_df=3 oder 5
max_df=0.8 oder 0.9
```

Vergleiche mit BERTweet auf Validation.

---

### Experiment 3: TF-IDF ohne Named-Entity-/Campaign-Dominanz

Teste Varianten:

```text
A: Hashtags behalten
B: Hashtags entfernen
C: Hashtags separat als Feature zählen
D: pro Celebrity N-Gramm-Counts cappen
```

Ziel: Prüfen, ob das Modell zu stark auf einzelne Kampagnen wie `#peopleschoice`, `#mtvstars` oder `#trx` reagiert.

---

### Experiment 4: Hybrid Gender-Modell

Kombiniere:

```text
BERTweet softmax probabilities: p_male, p_female
+ margin/confidence
+ Style-Features
+ TF-IDF SVD-Komponenten oder Top-N TF-IDF-Features
```

Ein einfacher Meta-Klassifikator könnte sein:

```text
Logistic Regression
LightGBM / XGBoost
Random Forest als Interpretationshilfe
```

---

### Experiment 5: Fehlergruppen gezielt auswerten

Besonders analysieren:

```text
female → male mit Politics-/News-Clustern
male → female mit Fan-/Voting-/Follow-back-Clustern
```

Ziel: Herausfinden, ob diese Fehler durch wenige dominante Celebrities/Follower-Feeds entstehen oder ein allgemeines Muster sind.

---

## 11. Zusammenfassung

Die Gender-Analyse zeigt, dass BERTweet offenbar stark von indirekten Community- und Stil-Signalen beeinflusst wird. Korrekte `female`-Vorhersagen korrelieren häufig mit Fan-, Voting-, Popkultur- und Emotionssprache. Korrekte `male`-Vorhersagen sowie `female → male`-Fehler enthalten häufiger Politik-, Tech-, News-, Crypto- oder stark sachlich wirkende Themencluster. `male → female`-Fehler entstehen dagegen häufiger bei Fan-, Follow-back-, Giveaway- und emotionalen Alltagssignalen.

Das Potenzial liegt daher weniger in aggressivem Entfernen von Textbestandteilen für BERTweet, sondern in einer sauberen Trennung:

```text
BERTweet bekommt weiterhin natürlichen Twitter-Text.
N-Gramm-/TF-IDF-Analyse filtert Artefakte kontrolliert.
Style-Features erfassen Mentions, URLs, Hashtags und Emojis separat.
Ein Hybridmodell kombiniert semantische BERTweet-Signale mit erklärbaren TF-IDF- und Style-Signalen.
```

Damit könnte das Modell robuster werden und zugleich besser erklärbar sein.
