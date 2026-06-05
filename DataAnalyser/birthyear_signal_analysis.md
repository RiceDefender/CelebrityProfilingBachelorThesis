# Birthyear Signal Analysis: BERTweet-Vorhersagen auf Follower-Feeds

## 1. Ziel der Analyse

Diese Analyse untersucht, welche sprachlichen Muster in den Follower-Feeds mit den Birthyear-Vorhersagen des BERTweet-Modells korrelieren. Die Tabellen stammen aus den N-Gramm- und Style-Feature-Exports für `birthyear` auf Test- und Validation-Split.

Wichtig: Die Analyse ist **keine direkte Attention-Erklärung** von BERTweet. Sie zeigt, welche Input-Muster in Gruppen wie `Correct 1985 high`, `Predicted younger than true` oder `Wrong/high confidence` besonders häufig oder überrepräsentiert sind. Damit eignet sie sich als Fehlerdiagnose und als Grundlage für mögliche zusätzliche Features.

---

## 2. Gesamtbild: Birthyear ist deutlich schwieriger als Occupation/Gender

Das Modell zeigt eine starke Tendenz zu mittleren Altersklassen, besonders `1985`.

| Split | Accuracy aus Style-Datei | auffälligste Tendenz |
|---|---:|---|
| Test | 0.420 | starke Übervorhersage von `1985`; `1947` wird praktisch nicht getroffen |
| Validation | 0.375 | ebenfalls starke Übervorhersage von `1985`; `1994` wird gar nicht korrekt getroffen |

### Test: Verteilung der echten und vorhergesagten Klassen

| Klasse | True Count | Pred Count | Recall |
|---|---:|---:|---:|
| 1947 | 50 | 3 | 0.000 |
| 1963 | 84 | 66 | 0.333 |
| 1975 | 87 | 91 | 0.287 |
| 1985 | 123 | 215 | 0.813 |
| 1994 | 56 | 25 | 0.268 |

### Validation: Verteilung der echten und vorhergesagten Klassen

| Klasse | True Count | Pred Count | Recall |
|---|---:|---:|---:|
| 1947 | 19 | 4 | 0.053 |
| 1963 | 42 | 42 | 0.357 |
| 1975 | 53 | 44 | 0.302 |
| 1985 | 54 | 99 | 0.741 |
| 1994 | 24 | 3 | 0.000 |


**Interpretation:** BERTweet scheint weniger ein robustes feingranulares Alterssignal zu lernen, sondern eher eine grobe Tendenz aus Community-, Themen- und Plattformmustern. Die Randklassen (`1947`, `1994`) sind besonders instabil. Das spricht dafür, Birthyear nicht nur als harte 5-Klassen-Klassifikation zu betrachten, sondern zusätzlich ordinal oder über Fehler-Richtungen (`zu jung`, `zu alt`) zu analysieren.

---

## 3. Korrekte Klassen: Welche Signale tauchen auf?

### 3.1 Correct `1985`

In beiden Splits tauchen bei korrekt vorhergesagtem `1985` viele community-/sport-/plattformnahe Muster auf.

Test, Bigram-Frequenz:

`happy birthday` (229.00), `upcoming artist` (193.00), `please help` (157.00), `music career` (83.00), `oluwadee upcoming` (79.00), `want housefull4` (71.00), `housefull4 poster` (71.00), `thank much` (69.00)

Validation, Bigram-Frequenz:

`youth league` (195.00), `follow everyone` (190.00), `posted new` (167.00), `new photo` (157.00), `photo facebook` (157.00), `central coast` (121.00), `happy birthday` (114.00), `league women` (105.00)

Validation, Log-Odds:

`youth league` (12.15), `league women` (11.54), `championship men` (11.50), `coast crusaders` (11.30), `league men` (10.99), `#fundourvets today` (10.83), `#harryreidsshutdown please` (10.83), `make vets` (10.83)

Auffällig ist, dass `1985` nicht durch klare Altersmarker erklärt wird. Stattdessen dominieren thematische Cluster wie Sport-/League-Sprache, Fan-/Follow-Muster und Plattformphrasen. Das kann bedeuten, dass `1985` als Default-Klasse von sehr unterschiedlichen, aber häufigen Feed-Typen profitiert.

### 3.2 Correct `1975`

Validation zeigt bei `1975` sehr technische, Lauf-/Fitness- oder Podcast-Cluster:

`km #endomondo` (158.00), `#endomondo #endorphins` (140.00), `running km` (126.00), `posted new` (100.00), `new photo` (98.00), `new blog` (93.00), `photo facebook` (93.00), `blog post` (85.00)

Log-Odds:

`craig wolfley podcast` (10.50), `stop federal purchasing` (10.42), `provide birth control` (10.40), `federal purchasing businesses` (10.37), `businesses provide birth` (10.37), `purchasing businesses provide` (10.37), `km #endomondo see` (10.35), `new craig wolfley` (10.26)

Auch hier sind die Signale nicht wirklich alterstypisch, sondern thematisch. Das kann für die Klassifikation helfen, ist aber riskant, wenn einzelne Follower-Communities oder Bot-/Posting-Cluster dominieren.

### 3.3 Correct `1963`

Validation zeigt bei `1963` stark arbeits-/recruitingnahe N-Gramme:

`new job` (141.00), `job alert` (138.00), `posted new` (73.00), `new photo` (70.00), `photo facebook` (69.00), `type news` (57.00), `happy birthday` (51.00), `news use` (50.00)

Log-Odds:

`new job alert` (11.86), `vote dc represent` (10.15), `dc represent views` (10.15), `change lives vulnerable` (10.11), `lives vulnerable youth` (10.07), `programming change lives` (10.02), `build deliver programming` (9.93), `job alert indian` (9.93)

Das ist potenziell ein plausibleres Signal: Job-, Politik-, Organisations- oder News-Sprache kann mit älteren Follower-Communities korrelieren. Trotzdem bleibt es ein indirektes Signal, da die Tweets von Followern stammen, nicht von der Celebrity selbst.

### 3.4 Correct `1947` und `1994`: instabile Randklassen

Die Randklassen sind nicht stabil über beide Splits sichtbar:

- Im Test gibt es keine korrekten `1947`-Vorhersagen.
- In Validation gibt es nur sehr wenige korrekte `1947`-Fälle.
- Im Test gibt es korrekte `1994`-Fälle, in Validation jedoch keine.

Validation `1947` zeigt unter anderem:

`turtleback books` (12.46), `turtleback editions` (11.51), `classroom organizer` (11.33), `school library` (11.33), `genuine turtleback` (10.97), `library vendor` (10.63), `favorite school` (10.63), `demand genuine` (10.63)

Test `1994` zeigt unter anderem:

`#beforeanyoneelse #blackarrowexpress` (12.19), `keywords text` (12.12), `#blackarrowexpress #bae` (12.09), `bucks balls` (11.96), `plus trip` (11.74), `kiss cash` (11.74), `listen kiss` (11.69), `two chances` (11.69)

Die Randklassen wirken stark von wenigen spezifischen Themenclustern abhängig. Das Modell findet also nicht zuverlässig eine allgemeine `sehr alt`- oder `sehr jung`-Sprache.

---

## 4. Fehlerrichtung: Wann schätzt das Modell zu jung oder zu alt?

### 4.1 `Predicted younger than true`

Hier werden ältere Celebrities einer jüngeren Birthyear-Klasse zugeordnet. In Test und Validation tauchen häufig Fan-, Voting-, Hashtag- und Entertainment-Cluster auf.

Test, Log-Odds Bigramme:

`meetandgreet tophernatics` (11.63), `thalapathy fans` (11.01), `#gloriousthalapathyvijayera #bigil` (10.98), `#bigil #hbdeminentvijay` (10.65), `song teaser` (10.50), `topherwoman sofiaandres` (10.50), `#freepalestine #istandwithpalestine` (10.50), `follow id` (10.44)

Validation, Log-Odds Bigramme:

`king decrumer` (12.40), `#liveme tbgc` (11.94), `tbgc king` (11.93), `special children` (11.85), `challenged children` (11.59), `yasss it's` (11.17), `handsome faces` (11.07), `vote #barunsobti` (11.05)

Interpretation: Wenn ältere Celebrities Follower-Feeds mit Fan-Kampagnen, Voting-Aufrufen, Entertainment-Hashtags oder starkem Social-Media-Rauschen haben, kann BERTweet diese Feeds als jünger interpretieren. Das ist plausibel, aber problematisch: Es ist eher ein Signal für die **Follower-Community** als für das tatsächliche Geburtsjahr der Celebrity.

### 4.2 `Predicted older than true`

Hier werden jüngere Celebrities als älter geschätzt. In den Tabellen finden sich eher Blog-/Commerce-/Marketing-/Service- oder Organisationscluster.

Test, Log-Odds Bigramme:

`vote #catforwg` (11.94), `dezemberphoto blog` (11.38), `blog new` (11.07), `say received` (10.71), `follow backk` (10.41), `#tennis #london` (10.41), `mmc finance` (10.34), `cyclo cross` (10.28)

Validation, Log-Odds Bigramme:

`today stats` (11.75), `stats new` (11.35), `daily stats` (11.23), `new unfollowers` (11.09), `follower unfollowers` (10.95), `stats one` (10.74), `prince nice` (10.62), `past day` (10.60)

Interpretation: Business-, Blog-, Service- und Produkt-Cluster können das Modell in Richtung älterer Birthyear-Klassen ziehen. Auch das ist kein direktes Alterssignal, sondern ein Community-/Themensignal.

### 4.3 `Wrong/high confidence`

Besonders kritisch sind hochkonfidente Fehler. Im Test werden sie von extrem repetitiven Phrasen dominiert:

`yasss it's time` (996.00), `it's time great` (996.00), `time great show` (996.00), `fumishunbase roadto2m fumishunbase` (69.00), `fumiya for100handsomefaces fumiya` (64.00), `roadto2m fumishunbase roadto2m` (59.00), `for100handsomefaces fumiya for100handsomefaces` (51.00), `ass ass ass` (38.00)

Validation, Log-Odds Trigramme:

`stats new followers` (11.17), `today stats followers` (11.07), `daily stats new` (10.95), `today stats one` (10.86), `stats one follower` (10.86), `#mtvstars one direction` (10.74), `followers new unfollowers` (10.63), `new followers new` (10.63)

Das ist ein wichtiges Warnsignal: Hochkonfidente Fehler entstehen häufig durch sehr dominante, wiederholte Themen- oder Kampagnenmuster. Solche Muster können BERTweet überstimmen, obwohl sie wenig mit dem eigentlichen Geburtsjahr zu tun haben.

---

## 5. Confusion Pairs: konkrete Verwechslungen

Die Confusion-Pair-Tabellen zeigen, dass viele Fehlklassifikationen nicht zufällig sind, sondern durch dominante Einzelcluster geprägt werden.

### Beispiele aus dem Test-Split

- `1947 → 1985`: `posted new photo`, `new photo facebook`, `photo facebook`  
  → Plattform-/Posting-Sprache lässt sehr alte Celebrities deutlich jünger wirken.

- `1963 → 1994`: `fumishunbase roadto2m`, `fumiya for100handsomefaces`, `roadto2m fumishunbase`  
  → Fan-Kampagnen und Voting-Sprache ziehen ältere Celebrities stark in Richtung jüngster Klasse.

- `1975 → 1994`: `yasss it's time`, `it's time great`, `time great show`  
  → extrem repetitive Entertainment-/Show-Sprache verursacht hochkonfidente Jung-Schätzung.

- `1994 → 1985`: `happy birthday`, `new blog post`, `posted new photo`, `new photo facebook`  
  → jüngste Celebrities werden häufig in die zentrale Klasse `1985` gezogen.

### Beispiele aus dem Validation-Split

- `1994 → 1985`: `stats new followers`, `today stats followers`, `daily stats new`  
  → automatische Twitter-Statistik-Posts oder Account-Management-Muster ziehen `1994` zu `1985`.

- `1985 → 1994`: `call viber`, `get dreamhair`, `dreamhair call`  
  → einzelne Commerce-/Spam-Cluster können zu jung wirken.

- `1975 → 1963`: `access control`, `biometric access`, `control technology`  
  → technische/Business-Sprache wirkt älter.

**Kernproblem:** Viele Confusion-Signale sind nicht alterssemantisch, sondern stammen aus einzelnen wiederholten Kampagnen, Plattformen, Bots, Shops, Jobs oder Fan-Communities.

---

## 6. Style-Features

### 6.1 Durchschnitt nach vorhergesagter Klasse: Test

|   pred_label |   confidence |   margin |   avg_tokens_per_tweet |   url_count |   mention_count |   hashtag_count |   emoji_count |   exclamation_count |   question_count |   uppercase_ratio |   love_word_count |   fan_word_count |   politics_word_count |   sports_word_count |
|-------------:|-------------:|---------:|-----------------------:|------------:|----------------:|----------------:|--------------:|--------------------:|-----------------:|------------------:|------------------:|-----------------:|----------------------:|--------------------:|
|         1947 |         0.3  |     0    |                  15.62 |      526    |          920.33 |          239    |         69.67 |              198    |            95.33 |              0.11 |             20.33 |             9.67 |                 78.67 |               45.67 |
|         1963 |         0.29 |     0.03 |                  16.65 |      413.91 |         1252.91 |          489.7  |        107.33 |              214.58 |           119.39 |              0.11 |             26.45 |            14.18 |                 52.47 |               28.94 |
|         1975 |         0.28 |     0.03 |                  14.99 |      414.33 |         1002.1  |          426.82 |        141.41 |              249.41 |           126.38 |              0.11 |             36.12 |            16.56 |                 23.41 |               28.62 |
|         1985 |         0.33 |     0.07 |                  13.47 |      411.13 |          984.77 |          438.09 |        249.2  |              291.88 |           110.2  |              0.12 |             46.1  |            20.4  |                 10.24 |               48.86 |
|         1994 |         0.37 |     0.06 |                  13.25 |      447.08 |         1040.12 |          237.32 |        419.44 |              258.84 |           110.12 |              0.14 |             61.68 |            19.24 |                 12.36 |               35.6  |

### 6.2 Durchschnitt nach vorhergesagter Klasse: Validation

|   pred_label |   confidence |   margin |   avg_tokens_per_tweet |   url_count |   mention_count |   hashtag_count |   emoji_count |   exclamation_count |   question_count |   uppercase_ratio |   love_word_count |   fan_word_count |   politics_word_count |   sports_word_count |
|-------------:|-------------:|---------:|-----------------------:|------------:|----------------:|----------------:|--------------:|--------------------:|-----------------:|------------------:|------------------:|-----------------:|----------------------:|--------------------:|
|         1947 |         0.33 |     0.04 |                  17.41 |      587.25 |         1033    |          110    |        203.75 |              320    |           149.75 |              0.1  |             18.5  |             4    |                 71.25 |               14.25 |
|         1963 |         0.3  |     0.03 |                  15.65 |      437.74 |         1092.79 |          392.52 |         88.26 |              266.79 |           125.86 |              0.11 |             28.74 |            14.24 |                 45.05 |               30.17 |
|         1975 |         0.3  |     0.03 |                  15.37 |      415.43 |          903.55 |          438.89 |        127.68 |              296.25 |           132.32 |              0.1  |             45.09 |            15.59 |                 17.89 |               40.07 |
|         1985 |         0.34 |     0.08 |                  13.02 |      434    |          972.25 |          355.64 |        172.42 |              342.4  |           112.17 |              0.13 |             38.86 |            25.94 |                 12.72 |               75.45 |
|         1994 |         0.35 |     0.07 |                  12.5  |      501.67 |          764    |          378.33 |        226.67 |              420    |           126.33 |              0.13 |             53    |            25.33 |                  3.33 |               15.67 |

### 6.3 Durchschnitt nach Fehlerrichtung: Test

| age_error_direction   |   confidence |   margin |   avg_tokens_per_tweet |   url_count |   mention_count |   hashtag_count |   emoji_count |   exclamation_count |   question_count |   uppercase_ratio |   love_word_count |   fan_word_count |   politics_word_count |   sports_word_count |
|:----------------------|-------------:|---------:|-----------------------:|------------:|----------------:|----------------:|--------------:|--------------------:|-----------------:|------------------:|------------------:|-----------------:|----------------------:|--------------------:|
| correct               |         0.32 |     0.06 |                  14.14 |      415.73 |         1040.14 |          400.5  |        212.39 |              252.06 |           106.14 |              0.12 |             43.08 |            17.47 |                 19.63 |               39.86 |
| predicted_older       |         0.32 |     0.05 |                  14.72 |      394.97 |         1054.39 |          411.16 |        205.84 |              264.15 |           124.27 |              0.11 |             39.6  |            19.92 |                 27.6  |               47.67 |
| predicted_younger     |         0.3  |     0.04 |                  14.38 |      424.87 |         1022.61 |          470.57 |        210.87 |              283.57 |           120.78 |              0.12 |             40.38 |            18.54 |                 18.92 |               36.78 |

### 6.4 Durchschnitt nach Fehlerrichtung: Validation

| age_error_direction   |   confidence |   margin |   avg_tokens_per_tweet |   url_count |   mention_count |   hashtag_count |   emoji_count |   exclamation_count |   question_count |   uppercase_ratio |   love_word_count |   fan_word_count |   politics_word_count |   sports_word_count |
|:----------------------|-------------:|---------:|-----------------------:|------------:|----------------:|----------------:|--------------:|--------------------:|-----------------:|------------------:|------------------:|-----------------:|----------------------:|--------------------:|
| correct               |         0.32 |     0.06 |                  14.24 |      398.88 |          983.57 |          364.9  |        122.4  |              331.67 |           132.61 |              0.11 |             38.29 |            24.04 |                 19.38 |               70.06 |
| predicted_older       |         0.32 |     0.06 |                  14.7  |      508.94 |          956.18 |          405.67 |        152.78 |              278.22 |           105.88 |              0.12 |             34.73 |            20.69 |                 25.55 |               43.8  |
| predicted_younger     |         0.32 |     0.05 |                  13.83 |      417.52 |          996.35 |          371.28 |        163.55 |              327.67 |           119.46 |              0.12 |             39.75 |            16.78 |                 22.25 |               48.2  |

### Style-Interpretation

Einige Tendenzen sind sichtbar, aber nicht vollständig stabil:

- Vorhersagen für `1994` haben im Test die höchsten Emoji- und Love-Word-Werte. Das passt zur Hypothese, dass emotionalere Fan-Sprache jüngere Klassen begünstigt.
- Vorhersagen für `1947` und `1963` haben tendenziell längere Tweets bzw. mehr tokens pro Tweet. Das kann auf News-, Politik-, Organisations- oder Business-Sprache hindeuten.
- `1985` wird sehr häufig vorhergesagt und wirkt wie eine Sammelklasse für viele mittlere und unklare Fälle.
- In Validation sind korrekte Fälle auffällig stark mit `sports_word_count` verbunden. Das zeigt, dass einzelne thematische Communities sehr stark wirken können.

Style-Features allein lösen Birthyear wahrscheinlich nicht, aber sie können als Zusatzsignale für ein Hybridmodell nützlich sein, besonders für Fehlerrichtung und Randklassen.

---

## 7. Woran könnte BERTweet Probleme haben?

### 7.1 Kein direktes Alterssignal

Die stärksten N-Gramme sind meistens keine direkten Altersmarker. Stattdessen erscheinen:

- Fan-Kampagnen und Voting-Aufrufe
- Plattformphrasen wie `posted new photo`
- Blog-/Commerce-/Service-Muster
- automatische Statistik-Posts
- Sport-/League-/Job-/Politics-Cluster
- sehr spezifische Named Entities oder Hashtags

Das Modell lernt damit vermutlich eher: **Welche Community folgt dieser Celebrity?** Nicht: **Wie alt ist die Celebrity?**

### 7.2 Zentralisierung auf `1985`

In beiden Splits wird `1985` massiv übervorhergesagt:

- Test: `1985` wird {int(test.pred_label.astype(str).value_counts().get('1985',0))} Mal vorhergesagt bei {int(test.true_label.astype(str).value_counts().get('1985',0))} echten `1985`-Fällen.
- Validation: `1985` wird {int(val.pred_label.astype(str).value_counts().get('1985',0))} Mal vorhergesagt bei {int(val.true_label.astype(str).value_counts().get('1985',0))} echten `1985`-Fällen.

Das deutet auf eine Default-/Mitte-Tendenz hin. Besonders `1994` und `1947` leiden darunter.

### 7.3 Repetitive Accounts dominieren

Mehrere hochkonfidente Fehler werden von extrem repetitiven Phrasen dominiert, z. B. Kampagnen- oder Bot-artige Wiederholungen. Dadurch kann eine kleine Anzahl lauter Follower einen gesamten Celebrity-Feed stark prägen.

### 7.4 Split-Instabilität

Einige Signale sind im Test sichtbar, aber nicht in Validation, oder umgekehrt. Das spricht gegen robuste, allgemeine Alterssignale und für Split-spezifische Themencluster.

---

## 8. Potenziale für bessere Modelle

### 8.1 Sauberere N-Gramm-Tokenisierung

Für die Analyse und für klassische Features sollten Social-Artefakte separat behandelt werden:

- `@USER`, `HTTPURL`, `rt`, `via`, `amp` aus N-Grammen entfernen oder gesondert zählen
- Hashtags nicht blind löschen, sondern normalisieren und optional segmentieren
- wiederholte identische N-Gramme pro Celebrity begrenzen
- sehr häufige Plattformphrasen als eigene Binär-/Count-Features behandeln

Wichtig: Diese Bereinigung sollte zunächst **nicht zwingend vor BERTweet** passieren. Für BERTweet kann der natürliche Twitter-Kontext wichtig sein. Für TF-IDF und Analyse ist eine sauberere Tokenisierung aber sehr sinnvoll.

### 8.2 TF-IDF-Zweig

Ein TF-IDF-Modell kann besonders nützlich sein, weil es:

- stark repetitive Phrasen abschwächen kann,
- seltene, aber diskriminative Hashtags kontrollierter nutzt,
- Feature-Gewichte interpretierbar macht,
- als Gegenpol zu BERTweet dient.

Empfehlung:

```text
TF-IDF word n-grams: 1–3
TF-IDF char n-grams: 3–5
min_df: 2 oder 3
max_df: 0.7 bis 0.9
sublinear_tf: True
class_weight: balanced
```

Zusätzlich sollte man Varianten testen:

```text
raw Twitter-normalisiert
ohne @USER/HTTPURL in N-Grammen
ohne Stopwords
mit Hashtag-Normalisierung
mit per-celebrity term cap
```

### 8.3 Style-Feature-Zweig

Die Style-Features liefern keine perfekte Trennung, aber sie können Randklassen und Fehlerrichtung unterstützen:

- Tweet-Länge
- URL-/Mention-/Hashtag-Anteil
- Emoji-/Exclamation-/Question-Counts
- Uppercase-Ratio
- Fan-/Love-/Sports-/Politics-Wörter
- Anzahl eindeutiger Hashtags
- Wiederholungsrate identischer Tweets oder N-Gramme

Gerade die Wiederholungsrate wäre für Birthyear wichtig, weil viele Fehler durch Kampagnen-/Bot-Muster entstehen.

### 8.4 Hybridmodell

Ein sinnvoller nächster Schritt wäre:

```text
BERTweet logits oder embedding
+ TF-IDF logits
+ Style features
→ Logistic Regression / LightGBM / small MLP
```

Dabei kann BERTweet die semantische Repräsentation liefern, während TF-IDF und Style-Features kontrollierte, interpretierbare Signale ergänzen.

### 8.5 Ordinales Age-Modell

Birthyear ist ordinal. `1947 → 1963` ist weniger falsch als `1947 → 1994`. Deshalb wäre ein ordinales Setup sinnvoll:

- Regression auf Birthyear-Bins
- ordinal classification
- zwei zusätzliche Hilfsziele: `older / middle / younger` oder `predicted direction`
- Loss mit Distanzgewichtung zwischen Klassen

Das könnte besonders helfen, weil das Modell aktuell häufig benachbarte oder zentrale Klassen bevorzugt.

---

## 9. Konkrete nächste Experimente

### Experiment A: TF-IDF Birthyear Baseline

```text
Input: bereinigte Follower-Feeds
Features: word n-grams 1–3 + char n-grams 3–5
Classifier: Logistic Regression oder Linear SVM
Evaluation: Macro-F1 + confusion matrix + distance-aware error
```

Ziel: Prüfen, ob klassische Features die Randklassen besser erkennen als BERTweet.

### Experiment B: BERTweet + TF-IDF Late Fusion

```text
Features: BERTweet class probabilities + TF-IDF class probabilities
Meta-classifier: Logistic Regression
```

Ziel: Nutzen, wenn BERTweet semantisch stark ist, TF-IDF aber spezifische N-Gramme besser kontrolliert.

### Experiment C: Repetition-aware Features

```text
repeat_ngram_ratio
unique_tweet_ratio
top_ngram_share
url_ratio
mention_ratio
hashtag_ratio
```

Ziel: Erkennen, wann ein Feed von Bot-/Kampagnenmustern dominiert wird und daher für Birthyear riskant ist.

### Experiment D: Ordinaler Fehler

Zusätzlich zur Accuracy:

```text
mean absolute bin error
predicted older vs predicted younger accuracy
macro-F1 auf older/middle/younger
```

Ziel: Besser beurteilen, ob das Modell wenigstens die Richtung des Alters versteht.

---

## 10. Fazit

Die Birthyear-Analyse zeigt deutlich, dass BERTweet bei Alter stärker an seine Grenzen stößt als bei Occupation oder Gender. Die stärksten Signale sind selten echte Altersmarker. Stattdessen dominieren Community-, Plattform-, Fan-, Bot-, Business- und Themencluster.

Das größte Problem ist die starke Zentralisierung auf `1985` und die schwache Erkennung der Randklassen `1947` und `1994`. Gleichzeitig zeigen die N-Gramme und Style-Features Potenzial: Mit einem bereinigten TF-IDF-Zweig, Repetition-Features und einem ordinalen Hybridansatz könnte man die Fehler besser kontrollieren und interpretierbarer machen.

Für die Bachelorarbeit ist Birthyear deshalb besonders interessant als Beispiel dafür, dass follower-basiertes Celebrity Profiling nicht nur Demographie lernt, sondern stark durch die Struktur und Sprache der Follower-Community beeinflusst wird.
