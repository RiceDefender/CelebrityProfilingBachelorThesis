# Occupation Signal Analysis – BERTweet Prediction Groups

Diese Analyse basiert auf den erzeugten Tabellen für `occupation` auf **Test** und **Validation**. Sie beschreibt keine direkte interne Transformer-Attention, sondern eine korrelationsbasierte Analyse der Eingabetexte: Welche N-Gramme und Stilmerkmale treten in Gruppen auf, die BERTweet korrekt oder falsch einer Occupation-Klasse zuordnet?

## 1. Kurzfazit

Die Occupation-Ergebnisse zeigen ein recht klares Muster:

- **Sports** und **Politics** sind tendenziell besser erkennbar, weil sie stärker thematische Signale besitzen.
- **Performer** wird häufig über Fan-, Musik-, Event- und Emotionssprache erkannt.
- **Creator** bleibt die schwierigste Klasse, weil sie sehr heterogen ist und häufig mit Performer, Politics oder allgemeinen Plattformmustern überlappt.
- Sehr viele Roh-N-Gramme bestehen aus Twitter-Artefakten wie `rt @user`, `@user @user`, `@user httpurl` und `httpurl via @user`. Diese sind als Meta-Features potenziell nützlich, aber für semantische N-Gramm-Diagramme störend.
- Log-Odds zeigt viele sehr spezifische Phrasen. Das ist hilfreich für Fehlerdiagnose, kann aber auch auf einzelne dominante Accounts oder Kampagnen hinweisen. Deshalb sollte man Log-Odds immer mit Dokumenthäufigkeit oder Mindestanzahl Celebrities absichern.

Die zentrale Schlussfolgerung ist: **Ein sauberer Hybrid aus BERTweet + TF-IDF/Style-Features kann sinnvoll sein**, aber das klassische Feature-Set muss stärker gegen Twitter-Artefakte, Wiederholungen und Einzelaccount-Dominanz kontrolliert werden.

---

## 2. Gesamtperformance und Confusion-Muster

### Test

| True \ Pred | creator | performer | politics | sports |
|---|---:|---:|---:|---:|
| creator | 28 | 40 | 22 | 10 |
| performer | 10 | 87 | 2 | 1 |
| politics | 14 | 4 | 82 | 0 |
| sports | 1 | 22 | 7 | 70 |

Test Accuracy aus den Style-Tabellen: **0.6675**.

Auffällig:

- **Creator wird oft als Performer vorhergesagt**: 40 von 100 Creator-Fällen.
- **Creator wird auch relativ oft als Politics vorhergesagt**: 22 von 100.
- **Sports wird häufig als Performer verwechselt**: 22 von 100.
- **Performer ist auf Test sehr stabil**: 87 von 100 korrekt.
- **Politics ist ebenfalls recht stabil**: 82 von 100 korrekt.

### Validation

| True \ Pred | creator | performer | politics | sports |
|---|---:|---:|---:|---:|
| creator | 24 | 7 | 13 | 4 |
| performer | 9 | 30 | 4 | 5 |
| politics | 4 | 2 | 40 | 2 |
| sports | 2 | 6 | 4 | 36 |

Validation Accuracy aus den Style-Tabellen: **0.6771**.

Auffällig:

- Die Grundtendenz bestätigt sich: Politics und Sports sind relativ stabil.
- Creator bleibt unsauber, aber auf Validation ist die Creator→Performer-Verwechslung weniger stark als auf Test.
- Performer wird auf Validation öfter mit Creator oder Sports verwechselt als auf Test.

**Interpretation:** Die Fehler sind nicht rein zufällig. Die stärkste Problemzone ist die Grenze zwischen **Creator** und **Performer**. Zusätzlich gibt es einzelne Themencluster, die Creator/Performer in Richtung Politics oder Sports ziehen.

---

## 3. N-Gramm-Signale in korrekt klassifizierten Fällen

### 3.1 Raw-N-Gramme: Twitter-Artefakte dominieren

In den rohen Frequenzlisten stehen bei fast allen Klassen ganz oben:

```text
rt @user
@user @user
@user httpurl
httpurl httpurl
httpurl via @user
```

Das zeigt: Ein großer Teil der Follower-Feeds besteht aus Retweets, Mentions und Links. Diese Signale sind nicht wertlos, aber für semantische Erklärdiagramme überdecken sie die eigentlichen Themen.

Empfehlung:

- Für N-Gramm-Visualisierung und TF-IDF-Semantik: `@user`, `httpurl`, `rt`, `via` aus den N-Grammen herausfiltern.
- Für Style-/Meta-Features: Mentions, URLs und Retweet-Anteil separat als numerische Features behalten.

---

### 3.2 Correct Sports

**Validation-Frequenzsignale:**

```text
stanford university
happy birthday
good luck
years ago today
#raw #rawbrooklyn #wwe
km #endomondo #endorphins
```

**Validation-Log-Odds:**

```text
#raw #rawphilly
#raw #rawpittsburgh
#rawbrooklyn #wwe
stanford university school
```

**Test-Log-Odds:**

```text
terry shields toyota
craig wolfley podcast
t's c's apply
```

Interpretation:

- Sports besitzt durchaus thematische Signale, aber nicht alle sind klassische Sports-Wörter wie `game`, `team`, `win`.
- In Validation treten WWE-/RAW- und Aktivitäts-/Lauf-Features auf.
- In Test tauchen spezifische Namen/Organisationen auf, die eventuell nur einzelne Feeds dominieren.

Potenzial:

- TF-IDF kann Sports helfen, wenn es robuste Sportbegriffe, Teams, Ligen, Hashtags und Aktivitätswörter aufnimmt.
- Gleichzeitig braucht man Dokumentfrequenz-Filter, damit Einzelbegriffe wie `terry shields toyota` nicht überbewertet werden.

---

### 3.3 Correct Performer

**Test-Frequenzsignale:**

```text
@user love
rt @user happy
@user happy birthday
```

**Test-Log-Odds:**

```text
dan phil
starship protect wonho
wonho starship protect
rome italy #r5rockstheworld
```

**Validation-Frequenzsignale:**

```text
happy birthday
posted photo
albummm follow
#android #androidgames #gameinsight
liammm #littlethings perfect
```

**Validation-Log-Odds:**

```text
handsome faces
vote #barunsobti
livin' moneymazi prod
pinkey beatz #soundcloud
```

Interpretation:

- Performer wird stark durch Fan- und Popkultur-Sprache getragen.
- Typische Muster: `love`, `happy birthday`, Fan-Aktionen, Musik-/Album-/SoundCloud-Signale, Namen von Bands/Artists/Idols.
- Das ist plausibel, aber auch gefährlich: Creator mit ähnlicher Fan-Community oder Event-Kommunikation können als Performer enden.

Potenzial:

- TF-IDF kann Performer gut ergänzen, wenn Musik-/Fan-/Event-Signale sichtbar bleiben.
- BERTweet scheint hier bereits relativ gut zu sein, aber Creator→Performer zeigt, dass diese Signale zu breit wirken.

---

### 3.4 Correct Creator

**Test-Frequenzsignale:**

```text
posted new photo
new photo facebook
photo facebook httpurl
```

**Test-Log-Odds:**

```text
#anegis #erp #msdynax
#erp #msdynax #mspartner
#cancer love horoscope
click read #cancer
```

**Validation-Frequenzsignale:**

```text
join us
going live
i'm listening
happy birthday
```

**Validation-Log-Odds:**

```text
hearts rockets
house plan
favorite streamers
going live come
come laugh little
```

Interpretation:

- Creator ist kein einheitliches Thema. Es gibt Business-/ERP-Cluster, Horoskop-/Content-Cluster, Streaming-/Live-Signale und generische Social-Media-Posting-Muster.
- Genau diese Heterogenität macht Creator schwierig.
- Creator hat weniger klare, stabile semantische Marker als Sports oder Politics.

Potenzial:

- Ein reiner BERTweet-Classifier kann Schwierigkeiten haben, weil Creator aus vielen Subtypen besteht.
- Ein Hybrid könnte helfen: TF-IDF erkennt einzelne Creator-Subdomänen, während BERTweet breitere Semantik abdeckt.
- Sinnvoll wären explizite Features wie `live`, `stream`, `youtube`, `blog`, `podcast`, `article`, `photo`, `facebook`, `twitch`, `design`, `book`, `art`, `writing`, aber mit Vorsicht, weil manche davon auch Performer oder Politics triggern können.

---

### 3.5 Correct Politics

**Test-Frequenzsignale:**

```text
rt @user dear
#thequranman #mohammadshaikh
```

**Test-Log-Odds:**

```text
global needs academy
follow much appreciated
#tron #trx trx
```

**Validation-Frequenzsignale:**

```text
must banned
commy regime
korean dictator
free speech
never communism
new job alert
korean leader moon
humanitarin aid korean cooperation
```

**Validation-Log-Odds:**

```text
commy regime
korean dictator
never communism
dictator kim
communist federal
new job alert
korean leader moon
rights liberty korean
```

Interpretation:

- Politics zeigt auf Validation sehr klare politische Signale: Kommunismus, Diktator, Korea, Free Speech, Regierung/Leader-Kontexte.
- Test wirkt dagegen stärker von einzelnen Themenclustern und Accounts beeinflusst.
- Trotzdem ist Politics insgesamt stabiler als Creator.

Potenzial:

- Politics profitiert sehr wahrscheinlich von TF-IDF, weil politische Vokabeln oft explizit sind.
- Aber es braucht Schutz gegen Keyword-Overfitting auf einzelne Kampagnen oder Named Entities.

---

## 4. Fehleranalyse: False-by-Prediction

### 4.1 False Performer

**Test false Performer:**

```text
@user love
@user happy
@user happy birthday
win plus trip
two chances win
great show
kiss cash
```

**Validation false Performer:**

```text
please cam
sound nightclub
#callmecam
music music music
want win giveaway
```

Interpretation:

- Das Modell scheint stark auf Fan-, Event-, Giveaway- und Entertainment-Sprache zu reagieren.
- Diese Sprache ist plausibel für Performer, kommt aber auch bei Creator, Sports oder Politics-Followern vor.
- Besonders Creator→Performer entsteht, wenn Creator-Feeds Event-, Musik-, Fan- oder Live-Sprache enthalten.

Potenzial:

- Für TF-IDF sollte Performer nicht nur mit Fan-Sprache modelliert werden. Man braucht Gegenfeatures für Creator: `stream`, `video`, `blog`, `posted`, `tutorial`, `design`, `photo`, `facebook`, `youtube`.
- Außerdem sollte man Giveaway-/Spam-Phrasen eventuell als Rauschcluster markieren.

---

### 4.2 False Creator

**Test false Creator:**

```text
fashion cause
perfectly posh
angry elephant
occupied india
bangladesh vs
delta state government
gm twitter family
```

**Validation false Creator:**

```text
#lovefood #lovefood
trash trash
completed km
new workout
#foodpic #foodie
others checked
```

Interpretation:

- False Creator entsteht oft durch sehr heterogene Inhalte: Food, Fitness, Fashion, regional politics, personal/community content.
- Das zeigt: Creator ist semantisch breit und wird teilweise als Sammelbecken für „nicht klar Sports/Politics/Performer“ genutzt.
- BERTweet könnte hier unsicher sein und Creator als Default für Social-/Lifestyle-/Content-Phrasen verwenden.

Potenzial:

- Creator braucht wahrscheinlich einen eigenen spezialisierten Zweig oder ein binäres Creator-Gating.
- Ein TF-IDF-Zweig mit Subdomänen kann helfen, aber nur, wenn die Features nicht zu stark auf einzelne Accounts overfitten.

---

### 4.3 False Politics

**Test false Politics:**

```text
enjoy content
content we'd love
love subscribe
ray ban sunglasses
video enjoy content
```

**Validation false Politics:**

```text
#leadership #seo
#seo #branding
#branding #smm
#marketing #startups
#iot #ai #ceo
call viber text
posted new photo
new photo facebook
```

Interpretation:

- Politics wird teilweise durch öffentliche/organisatorische Business-Sprache, Leadership-, SEO-, Startup- und AI-Hashtags getriggert.
- Das ist eine wichtige Fehlerquelle: `leadership`, `branding`, `ceo`, `startups`, `ai` klingen gesellschaftlich/öffentlich, sind aber nicht zwingend Politics.
- Test enthält zudem Spam-/Commerce-Cluster wie `ray ban sunglasses sale`.

Potenzial:

- TF-IDF sollte Politics nicht nur über `leadership`, `ceo`, `public`, `government` lernen, sondern stärker echte politische Begriffe gewichten: `vote`, `election`, `minister`, `government`, `senate`, `policy`, `democracy`, `rights`, `law`.
- Business-/Marketing-Hashtags könnten als eigenes Rausch-/Business-Cluster markiert werden.

---

### 4.4 False Sports

**Test false Sports:**

```text
#atd #godawgs
aint done yet
#blacktwitter aint done
re black like
```

**Validation false Sports:**

```text
posted new photo
new photo facebook
happy birthday
julian assange
lol nd
candy cart
ppr chark jamaal
```

Interpretation:

- False Sports ist nicht nur Sportvokabular. Es treten Hashtags, Social-Media-Posting-Muster und einzelne Namen/Themen auf.
- In Validation taucht mit `ppr chark jamaal` ein klares Fantasy-Football-Signal auf, das Politics→Sports auslösen kann.
- Andere False-Sports-Signale sind aber eher Artefakte oder Einzelaccount-Cluster.

Potenzial:

- Sports lässt sich mit robusten Sportlexika wahrscheinlich verbessern.
- Gleichzeitig müssen zufällige Hashtag-Ketten und Einzelcluster gedämpft werden.

---

## 5. Confusion-Pairs: Was passiert konkret?

### 5.1 Creator → Performer

**Test:**

```text
yasss it's time
it's time great
time great show
want housefull4 poster
```

**Validation:**

```text
can't wait
sound nightclub
new event alert
excited announce i'll
free rsvp
```

Interpretation:

- Creator wird als Performer klassifiziert, wenn Feeds Event-, Show-, Nightclub-, Announcement- oder Fan-Sprache enthalten.
- Das ist plausibel: Viele Creator arbeiten ebenfalls mit Events, Shows, Releases und Publikumssprache.

Ansatz:

- Ergänzende Features sollten Creator-spezifische Plattform-/Produktionssignale abgrenzen: `going live`, `stream`, `video`, `blog`, `tutorial`, `podcast`, `posted photo`, `facebook`, `youtube`.
- Performer-spezifische Signale sollten stärker musikalisch/artistisch sein: `album`, `song`, `concert`, `tour`, `stage`, `fan`, `idol`, `actor`, `movie`.

---

### 5.2 Creator → Politics

**Test:**

```text
check muhammad nasar
muhammad nasar video
nasar video #tiktok
```

**Validation:**

```text
popular link among
among people follow
ur email id
```

Interpretation:

- Creator→Politics scheint stark durch einzelne Content-/Video-/Community-Ketten geprägt zu sein.
- Hier könnte das Modell nicht echte Politics erkennen, sondern öffentliche/virale Kommunikationsmuster.

Ansatz:

- Entfernen oder Dämpfen hochrepetitiver Kampagnen-/Spam-N-Gramme.
- Mindestanzahl verschiedener Celebrities pro N-Gramm einführen.

---

### 5.3 Politics → Creator

**Test:**

```text
delta state government
governor ifeanyi okowa
happy new year
```

**Validation:**

```text
trash trash trash
entering palestine israel
netanyahu plans block
stand presidential bullying
```

Interpretation:

- Politics→Creator zeigt, dass politische Inhalte nicht immer als Politics erkannt werden, wenn sie stark personalisiert, aktivistisch, ironisch, repetitiv oder kampagnenartig sind.
- Einige Phrasen sind politisch, aber eventuell nicht im typischen Politics-Cluster des Modells.

Ansatz:

- Politics-Features sollten breiter werden: Aktivismus, Protest, Menschenrechte, internationale Konflikte, Regierung, Parteien, Politiker, Wahlen.
- Zusätzlich kann ein Named-Entity-/Gazetteer-Zweig helfen.

---

### 5.4 Sports → Performer

**Test:**

```text
happy birthday
tracking number
#beforeanyoneelse #blackarrowexpress
kindly send us
```

**Validation:**

```text
please cam #facetimemecam
#callmecam please cam
want win giveaway
music music music
```

Interpretation:

- Sports wird als Performer klassifiziert, wenn die Follower-Feeds fan-, giveaway-, promotion- oder customer-support-artige Sprache enthalten.
- Das ist wahrscheinlich kein echtes Sports-Signal, sondern Community-/Follower-Rauschen.

Ansatz:

- Per-Follower-Aggregation statt nur globaler Aggregation: Einzelne laute Fan-/Spam-Accounts dürfen nicht den ganzen Celebrity-Feed dominieren.
- Pro Celebrity oder pro Follower N-Gramm-Capping einführen.

---

### 5.5 Sports → Politics

**Test und Validation:**

```text
check muhammad nasar
muhammad nasar video
nasar video #tiktok
prime minister imran
```

Interpretation:

- Sports→Politics scheint vor allem durch politische/virale Video-Cluster ausgelöst zu werden.
- Das ist ein Hinweis, dass einzelne Follower-Feeds thematisch stark vom Celebrity-Label abweichen können.

Ansatz:

- Robuste Aggregation über Follower: Median/Trimmed Mean von Features statt nur Summe.
- Dokumentfrequenz auf Celebrity-Ebene statt Tweet-Ebene.

---

## 6. Style-Features

### 6.1 Klassenprofile auf Test

| True Label | avg tokens/tweet | URL | Mentions | Hashtags | Emojis | Love-Words | Politics-Words | Sports-Words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| creator | 14.30 | 443.66 | 971.54 | 440.27 | 216.67 | 48.47 | 19.24 | 34.43 |
| performer | 13.95 | 428.00 | 1016.00 | 464.57 | 296.71 | 52.48 | 12.88 | 28.34 |
| politics | 15.65 | 409.01 | 1184.19 | 501.83 | 116.92 | 27.71 | 40.36 | 29.20 |
| sports | 13.47 | 381.04 | 971.99 | 313.34 | 211.96 | 36.82 | 10.91 | 68.49 |

### 6.2 Klassenprofile auf Validation

| True Label | avg tokens/tweet | URL | Mentions | Hashtags | Emojis | Love-Words | Politics-Words | Sports-Words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| creator | 14.38 | 430.06 | 918.67 | 331.77 | 132.75 | 46.62 | 28.04 | 27.62 |
| performer | 12.91 | 427.58 | 904.60 | 413.69 | 325.35 | 64.85 | 15.96 | 30.25 |
| politics | 16.76 | 447.85 | 1187.48 | 406.08 | 137.06 | 32.42 | 50.23 | 34.38 |
| sports | 13.73 | 433.40 | 926.10 | 401.94 | 227.79 | 30.77 | 11.92 | 88.08 |

### 6.3 Interpretation der Style-Features

Stabile Muster:

- **Politics** hat im Mittel die längsten Tweets und die meisten Politics-Wörter.
- **Sports** hat mit Abstand die meisten Sports-Wörter.
- **Performer** hat deutlich mehr Emojis und Love-Wörter, besonders auf Validation.
- **Creator** liegt oft zwischen den Klassen und wirkt stilistisch weniger eindeutig.

Problematisch:

- Correct vs. Wrong unterscheidet sich im Test nur schwach. Auf Test sind richtige Vorhersagen etwas konfidenter, aber die Style-Features allein trennen die Fehler nicht sauber.
- Auf Validation sind richtige Vorhersagen klarer: höhere Confidence, höherer Margin und mehr Sports-/Politics-Wörter in den korrekten Fällen.

Potenzial:

- Style-Features sollten nicht allein klassifizieren, aber als Zusatzfeatures können sie helfen.
- Besonders nützlich wirken: `emoji_count`, `love_word_count`, `politics_word_count`, `sports_word_count`, `avg_tokens_per_tweet`, `mention_count`, `hashtag_count`.
- Mentions/URLs sollten nicht als N-Gramme dominieren, aber als numerische Features erhalten bleiben.

---

## 7. Woran BERTweet wahrscheinlich Probleme hat

### 7.1 Heterogene Creator-Klasse

Creator ist thematisch breit: Live-Streams, Blogs, Fotos, Business, Horoskope, Food, Fitness, Design, Tech, Social-Media-Posting. Dadurch gibt es weniger stabile Kernsignale.

Folge:

- Creator wird oft mit Performer verwechselt, wenn Fan-/Event-Sprache vorkommt.
- Creator wird mit Politics verwechselt, wenn öffentliche/virale Content- oder Kampagnenmuster vorkommen.

### 7.2 Dominante Follower- oder Kampagnencluster

Viele Log-Odds-Topphrasen sind extrem spezifisch:

```text
terry shields toyota
starship protect wonho
#android #androidgames #gameinsight
muhammad nasar video
#seo #branding #smm
ray ban sunglasses sale
```

Solche Phrasen können aus einzelnen Followern oder wenigen stark repetitiven Quellen stammen.

Folge:

- BERTweet und TF-IDF können auf zufällige Cluster lernen.
- Frequenzanalyse wirkt dann erklärbar, ist aber nicht unbedingt generalisierbar.

### 7.3 Fan-Sprache überlappt stark

`happy birthday`, `love`, `can't wait`, `great show`, `giveaway`, `please cam`, `sound nightclub` sind nicht exklusiv Performer.

Folge:

- Performer wird als attraktive Zielklasse für viele expressive Community-Feeds.
- Sports und Creator können dadurch zu Performer kippen.

### 7.4 Politics vs. Business/Leadership/SEO

Politics wird durch Begriffe wie `leadership`, `ceo`, `branding`, `startups`, `ai` manchmal falsch getriggert.

Folge:

- Business-/Marketing-Sprache wirkt öffentlich/gesellschaftlich, ist aber nicht notwendigerweise Politik.

---

## 8. Potenziale für saubereren Tokenizer

Für die Visualisierung und klassische Features sollte eine zweite, analyseorientierte Tokenisierung genutzt werden:

### 8.1 Social-Artefakte aus N-Grammen entfernen

Für N-Gramme:

```text
@user
httpurl
rt
via
amp
gt
lt
```

entfernen oder aus N-Grammen ausschließen.

Aber separat zählen als Features:

```text
mention_count
url_count
retweet_count
hashtag_count
```

### 8.2 Repetition-Capping

Ein N-Gramm sollte pro Celebrity oder pro Follower nur begrenzt zählen, zum Beispiel:

```text
max_count_per_ngram_per_celebrity = 5 oder 10
```

Das verhindert, dass ein einzelner Account mit 1000 ähnlichen Tweets den Plot dominiert.

### 8.3 Celebrity-Level Document Frequency

Für TF-IDF und Log-Odds sollte nicht nur Tweet-Frequenz zählen, sondern in wie vielen Celebrities ein N-Gramm vorkommt.

Empfohlene Filter:

```text
min_celeb_df >= 3
max_celeb_df <= 0.7
min_total_count >= 5
```

Damit verschwinden Einzelcluster wie sehr spezifische Namen oder Spam-Ketten teilweise.

### 8.4 Hashtags differenziert behandeln

Hashtags sind wichtig, aber oft spammy. Vorschlag:

- Hashtags als eigene Features behalten.
- Für semantische N-Gramme entweder normalisieren (`#worldcup` → `worldcup`) oder separat auswerten.
- Hashtag-Ketten wie `#seo #branding #smm` als potenzielles Business-/Marketing-Cluster erkennen.

---

## 9. Potenzial für TF-IDF

Ein TF-IDF-Zweig ist besonders sinnvoll, weil die N-Gramm-Analyse deutliche klassenspezifische Muster zeigt. Aber er sollte sorgfältig gebaut werden.

### 9.1 Empfohlenes TF-IDF-Setup

```python
word_tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=3,              # besser auf Celebrity-Level, falls möglich
    max_df=0.7,
    sublinear_tf=True,
    max_features=50000,
    token_pattern=custom_tokenizer
)

char_tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    max_features=30000,
    sublinear_tf=True
)
```

Zusätzlich numerische Features:

```text
url_count
mention_count
hashtag_count
emoji_count
exclamation_count
question_count
avg_tokens_per_tweet
love_word_count
fan_word_count
politics_word_count
sports_word_count
```

### 9.2 Warum TF-IDF helfen kann

- Politics: explizite politische Vokabeln sind gut erfassbar.
- Sports: Teams, Ligen, Sportbegriffe und Hashtags sind gut erfassbar.
- Performer: Fan-/Musik-/Eventsignale sind sichtbar.
- Creator: Subdomänen wie Streaming, Blog, Foto, Food, Fitness, Business, Design können über N-Gramme besser erkannt werden.

### 9.3 Risiko bei TF-IDF

Ohne Kontrolle lernt TF-IDF sehr schnell Datenartefakte:

```text
ray ban sunglasses sale
#seo #branding #smm
muhammad nasar video
starship protect wonho
terry shields toyota
```

Deshalb sind wichtig:

- Celebrity-Level `min_df`.
- Per-Celebrity Capping.
- Separater Artefaktfilter.
- Evaluation auf Validation und Test getrennt.

---

## 10. Potenzial für Hybrid-Modell

Ein Hybrid ist wahrscheinlich sinnvoller als BERTweet oder TF-IDF allein.

### 10.1 Variante A: Late Fusion

BERTweet liefert Wahrscheinlichkeiten:

```text
p_sports, p_performer, p_creator, p_politics
```

TF-IDF/Style-Modell liefert ebenfalls Wahrscheinlichkeiten.

Dann kombiniert man:

```text
final_logits = alpha * bertweet_logits + (1 - alpha) * tfidf_logits
```

Vorteil:

- Einfach zu testen.
- Interpretierbar.
- Alpha kann auf Validation optimiert werden.

### 10.2 Variante B: Stacking

Features:

```text
BERTweet probabilities
BERTweet confidence
BERTweet margin
TF-IDF model probabilities
Style features
```

Darauf ein kleiner Logistic-Regression-Classifier.

Vorteil:

- Kann lernen, wann BERTweet unsicher ist.
- Besonders nützlich für Creator/Performer-Grenzen.

### 10.3 Variante C: Class-specific Gates

Mögliche Gates:

```text
Politics-vs-Rest
Sports-vs-Rest
Creator-vs-Rest
Performer-vs-Creator
```

Gerade Creator könnte von einem Binary-Gate profitieren, weil die Hauptfehler bei Creator entstehen.

---

## 11. Konkrete nächste Experimente

### Experiment 1: Saubere N-Gramme erneut plotten

```bash
python -m DataAnalyser.plot_prediction_signals_fast \
  --target occupation \
  --split test \
  --modes frequency logodds \
  --ngrams 2 3 \
  --remove-stopwords \
  --drop-social-ngram-tokens \
  --drop-rt-artifacts \
  --top-docs 20 \
  --max-tweets-per-celebrity 1000
```

Dasselbe für Validation.

Ziel: Prüfen, welche Signale nach Entfernung von `@user` und `httpurl` stabil bleiben.

### Experiment 2: TF-IDF Baseline mit Artefaktfilter

Trainiere Logistic Regression oder Linear SVM auf:

```text
word n-grams 1–3
char n-grams 3–5
style features
```

Vergleiche:

```text
BERTweet allein
TF-IDF allein
BERTweet + TF-IDF Late Fusion
BERTweet + TF-IDF + Style Stacking
```

### Experiment 3: Per-Celebrity Capping

Vor dem Zählen:

```text
jedes N-Gramm pro Celebrity maximal 5 oder 10 mal zählen
```

Ziel: Einzelaccount-Dominanz reduzieren.

### Experiment 4: Creator/Performer-Spezialanalyse

Nur Fälle:

```text
creator → creator
creator → performer
performer → performer
performer → creator
```

Dann prüfen:

- Welche Creator-Subdomänen landen bei Performer?
- Welche Performer-Fälle landen bei Creator?
- Welche Features trennen die beiden wirklich?

### Experiment 5: Fehlercluster manuell labeln

Für die häufigsten False-Prediction-Cluster manuell Kategorien vergeben:

```text
Spam/Commerce
Fan/Performer
Politics/Activism
Sports/Fantasy
Creator/Streaming
Business/Marketing
Platform Posting
```

Das kann in der Bachelorarbeit als qualitative Fehleranalyse verwendet werden.

---

## 12. Formulierung für die Bachelorarbeit

Mögliche Formulierung:

> Die Analyse der Occupation-Vorhersagen zeigt, dass korrekt klassifizierte Sports- und Politics-Fälle stärker durch thematische N-Gramme und domänenspezifische Wörter geprägt sind, während Performer-Fälle häufig Fan-, Musik- und Eventsprache enthalten. Die Creator-Klasse zeigt dagegen eine hohe semantische Heterogenität und wird insbesondere mit Performer und Politics verwechselt. Rohe N-Gramm-Frequenzen werden stark durch Twitter-Artefakte wie Mentions, Retweets und URLs dominiert. Diese Artefakte sollten daher nicht als semantische N-Gramme interpretiert, aber als separate Stil- und Metafeatures erhalten werden. Die Ergebnisse sprechen für einen Hybridansatz, der BERTweet-Repräsentationen mit bereinigten TF-IDF-N-Grammen und einfachen Twitter-Stilfeatures kombiniert.

---

## 13. Wichtigste Takeaways

1. **Creator ist das Hauptproblem.** Die Klasse ist semantisch breit und überlappt mit Performer, Politics und Plattform-Sprache.
2. **Performer wird stark über Fan-/Event-/Emotionssprache erkannt.** Das hilft, verursacht aber auch False Performer.
3. **Politics und Sports haben klarere thematische Signale.** Sie eignen sich gut für ergänzende TF-IDF-Features.
4. **Twitter-Artefakte dominieren Raw-N-Gramme.** Für Visualisierung und TF-IDF sollten sie gefiltert, aber als Style-Features erhalten bleiben.
5. **Log-Odds ist nützlich, aber anfällig für Einzelcluster.** N-Gramme brauchen Celebrity-Level-Dokumentfrequenz und Capping.
6. **Ein Hybridmodell ist vielversprechend.** Besonders Late Fusion oder Stacking aus BERTweet-Probabilities, TF-IDF und Style-Features.
