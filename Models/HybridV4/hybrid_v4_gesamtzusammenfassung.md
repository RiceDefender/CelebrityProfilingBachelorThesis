# HybridV4 – Gesamtzusammenfassung, Beobachtungen und Interpretation

## 1. Ausgangslage

Im aktuellen Stand wurde für das Celebrity-Profiling-Projekt ein HybridV4-Ansatz untersucht. Ziel war es, die Klassifikation der drei Targets `occupation`, `gender` und später `birthyear` durch eine Kombination verschiedener Modellquellen zu verbessern.

Die Fusion kombiniert drei Informationsquellen:

1. **BERTweet v3**  
   Ein textbasiertes Modell auf Basis der Follower-Feeds.

2. **BERTweet v3.4**  
   Eine weiterentwickelte BERTweet-Variante, ebenfalls auf Tweet-Texten.

3. **Sparse Feature Model**  
   Ein klassisches Feature-basiertes Modell mit extrahierten linguistischen und statistischen Merkmalen.

Die ursprüngliche Erwartung war, dass sich die Stärken dieser drei Modelle ergänzen und dadurch eine stabilere, bessere Gesamtleistung entsteht als bei einzelnen Modellen.

---

## 2. Fair Weighted Fusion

Für die Fusion wurde eine gewichtete Mittelung der Wahrscheinlichkeiten verwendet:

```text
final_probs = w1 * bertweet_v3_probs
            + w2 * bertweet_v34_probs
            + w3 * sparse_feature_probs
```

Die Gewichte wurden auf dem `fusion_val`-Split per Grid Search bestimmt und danach auf dem offiziellen Testsplit evaluiert. Dadurch wurde vermieden, die Testdaten zur Modellselektion zu verwenden.

Für `occupation` wurde zusätzlich ein optionaler `creator_boost_alpha` getestet, um die stark unterrepräsentierte beziehungsweise schwer erkennbare Klasse `creator` gezielt zu unterstützen.

---

## 3. Ergebnisse: Occupation

### 3.1 Beste Fusion ohne Creator Boost

Für `occupation` ergab die Fair Weighted Fusion auf `fusion_val` folgende beste Konfiguration:

```json
{
  "weights": [
    0.6428571428571429,
    0.2142857142857143,
    0.14285714285714288
  ],
  "creator_boost_alpha": 0.0,
  "fusion_val_accuracy": 0.7630208333333334,
  "fusion_val_macro_f1": 0.7600956457112553
}
```

Auf dem Testsplit ergab sich:

```text
test_acc      = 0.6475
test_macro_f1 = 0.6322
```

### 3.2 Occupation Classification Report

```text
              precision    recall  f1-score   support

      sports     0.7604    0.7300    0.7449       100
   performer     0.5370    0.8700    0.6641       100
     creator     0.5472    0.2900    0.3791       100
    politics     0.7865    0.7000    0.7407       100

    accuracy                         0.6475       400
   macro avg     0.6578    0.6475    0.6322       400
weighted avg     0.6578    0.6475    0.6322       400
```

### 3.3 Occupation Confusion Matrix

Labels:

```text
['sports', 'performer', 'creator', 'politics']
```

Confusion Matrix:

```text
[[73 23  0  4]
 [ 3 87  9  1]
 [13 44 29 14]
 [ 7  8 15 70]]
```

### 3.4 Wichtigste Fehler bei Occupation

```text
creator    -> performer : 44
sports     -> performer : 23
politics   -> creator   : 15
creator    -> politics  : 14
creator    -> sports    : 13
performer  -> creator   : 9
politics   -> performer : 8
politics   -> sports    : 7
sports     -> politics  : 4
performer  -> sports    : 3
performer  -> politics  : 1
```

### 3.5 Interpretation Occupation

Die Fusion hat für `occupation` nicht den erwarteten klaren Leistungsgewinn gebracht. Das Hauptproblem bleibt die Klasse `creator`:

- Nur 29 von 100 Creator-Beispielen wurden korrekt erkannt.
- 44 Creator wurden als `performer` klassifiziert.
- Weitere Creator wurden als `politics` oder `sports` klassifiziert.
- `performer` hat einen sehr hohen Recall von 0.87, aber eine relativ schwache Precision von 0.5370.

Das bedeutet: Das Modell tendiert stark dazu, unsichere oder ambivalente Profile als `performer` einzuordnen. Die Klasse `performer` wirkt wie eine dominante Auffangklasse.

Der Creator Boost konnte dieses Problem nicht zuverlässig lösen. In einigen Varianten wurde `creator` zwar leicht häufiger vorhergesagt, aber gleichzeitig stiegen False Positives. Dadurch verbesserte sich die Gesamtleistung nicht stabil.

---

## 4. Ergebnisse: Gender

Für `gender` wurde die Fair Weighted Fusion ebenfalls erfolgreich ausgeführt.

### 4.1 Beste Gender-Fusion

```json
{
  "weights": [
    0.2777777777777778,
    0.5555555555555556,
    0.16666666666666666
  ],
  "creator_boost_alpha": 0.0,
  "fusion_val_accuracy": 0.7630208333333334,
  "fusion_val_macro_f1": 0.7594167177291098
}
```

Auf dem Testsplit ergab sich:

```text
test_acc      = 0.6725
test_macro_f1 = 0.6671
```

### 4.2 Interpretation Gender

Auch bei `gender` war die Fusion nicht der klare Durchbruch. Es gibt Hinweise darauf, dass die Modelle einige starke semantische Signale in den Tweets nicht ausreichend ausnutzen.

Besonders interessant war die qualitative Beobachtung bei `female -> male`-Fehlern. In mehreren Fällen enthielten die Tweets Themen wie:

- Ehe und Partnerschaft
- `wife`, `husband`, `woman`, `daughter`, `mother`
- Gewalt gegen Frauen
- familiäre Konflikte
- Beziehungsthemen
- moralisch-emotionale Reaktionen auf gesellschaftliche Ereignisse

Diese Inhalte scheinen potenziell gender-informativ zu sein, werden aber vom aktuellen Modell teilweise überdeckt. Vermutlich dominieren andere Signale wie Politik, Sport, Religion, Nachrichten oder regionale Hashtags.

Wichtig für die Bachelorarbeit: Diese Beobachtung sollte vorsichtig und datenanalytisch formuliert werden. Nicht als Aussage wie „Frauen schreiben emotionaler“, sondern als:

> Several misclassified female profiles contain gender-associated topical cues related to family roles, marriage, relationships, violence against women, and emotionally charged social issues. These cues may not be sufficiently captured by the current fusion model.

---

## 5. PAN-basierte Error- und Token-Contrast-Analyse

Zur genaueren Fehleranalyse wurden die PAN-Follower-Feeds aus folgender Datei verwendet:

```text
data/pan20-celebrity-profiling-test-dataset-2020-02-28/pan20-celebrity-profiling-test-dataset-2020-02-28/follower-feeds.ndjson
```

Die Idee war, nicht nur Modellmetriken zu betrachten, sondern konkrete Tweet-Inhalte und Token-Verteilungen zwischen korrekt und falsch klassifizierten Gruppen zu vergleichen.

### 5.1 Occupation Token Contrast

Für `occupation` wurden unter anderem folgende Gruppen analysiert:

- `correct_performer`
- `correct_sports`
- `correct_politics`
- `correct_creator`
- `creator_to_performer`
- `sports_to_performer`
- `politics_to_creator`
- `creator_to_politics`
- `creator_to_sports`

Die Analyse zeigte, dass viele Fehlklassifikationen durch thematische Überlappung erklärbar sind:

- `creator_to_performer` enthält viele Entertainment-, Fan-, Musik-, Film- und Social-Media-Marketing-Signale.
- `sports_to_performer` enthält teils Fan- und Entertainment-Tokens, wodurch Sportprofile wie Performer wirken können.
- `performer_to_creator` enthält deutliche Creator-/Vintage-/Design-/Shop-Signale, z. B. `#vintage`, `#vintageshop`, `#vintagedress`, `#interiordesign`.

Das spricht dafür, dass die Klassen semantisch nicht scharf getrennt sind. Besonders `creator` und `performer` überlappen stark.

### 5.2 Gender Token Contrast

Für `gender` wurden unter anderem folgende Gruppen analysiert:

- `correct_male`
- `correct_female`
- `female_to_male`
- `male_to_female`

Die Token-Contrast-Analyse zeigte, dass viele der auffälligen Tokens sehr stark von einzelnen Autoren oder Communities beeinflusst sind. Beispiele sind spezifische Hashtags, Fan-Communities, regionale politische Themen oder Kurzlinks.

Damit ergibt sich eine wichtige Erkenntnis:

> Die Modelle lernen nicht nur stabile demografische Signale, sondern auch Community-, Topic-, Sprachraum- und Fandom-Signale. Diese können mit den Zielklassen korrelieren, sind aber nicht immer generalisierbar.

---

## 6. Zentrale Erkenntnisse

### 6.1 Die Fusion allein reicht nicht aus

Die gewichtete Fusion der Modellwahrscheinlichkeiten verbessert die Leistung nicht automatisch. Wenn die Basismodelle ähnliche Fehler machen, kann eine lineare Fusion diese Fehler kaum korrigieren.

Besonders deutlich wurde das bei High-Confidence-Fehlern, bei denen alle drei Modelle denselben falschen Labelvorschlag hatten.

### 6.2 Creator ist das Hauptproblem bei Occupation

Die Klasse `creator` hat den schwächsten Recall und wird häufig mit `performer` verwechselt. Das ist vermutlich kein reines Modellproblem, sondern auch ein Datenproblem:

- Creator und Performer teilen viele Fan-, Medien-, Plattform- und Entertainment-Signale.
- Creator können über Inhalte, Kampagnen, Bücher, Musik, Design, Shops oder Social Media auftreten.
- Performer haben ebenfalls starke öffentliche und fanbezogene Diskurse.

Die Grenze zwischen beiden Klassen ist in Follower-Feeds offenbar schwer zu erkennen.

### 6.3 Sparse Features helfen, aber lösen das Problem nicht allein

Das Feature-Modell liefert zusätzliche Informationen, aber in der gewichteten Fusion bleibt sein Anteil meist relativ klein. Das deutet darauf hin, dass die Sparse Features hilfreich, aber nicht stark genug sind, um die dominanten BERTweet-Fehlentscheidungen systematisch zu korrigieren.

### 6.4 Semantische Signale sind vielversprechend

Die qualitative PAN-Analyse zeigt, dass bestimmte semantische Muster für Fehlergruppen relevant sein könnten:

Für `gender`:

- Familie
- Ehe
- Beziehung
- Gewalt gegen Frauen
- weibliche Rollenbegriffe
- emotional diskutierte soziale Konflikte

Für `occupation`:

- Creator-Signale wie Writing, Design, Vintage, Shop, Startup, Branding, Content, Social Media
- Performer-Signale wie Fan-Army, Musik, Film, Schauspieler, Shows, Celebrities
- Politics-Signale wie Parteien, Wahlen, politische Konflikte, Institutionen
- Sports-Signale wie Teams, Spiele, Athleten, Ligen, Wettbewerbe

Diese Signale könnten als neue Feature-Gruppen extrahiert und in einem Meta-Modell genutzt werden.

### 6.5 Vorsicht vor Overfitting auf einzelne Tokens

Die Token-Contrast-Analyse zeigt viele sehr spezifische Hashtags und IDs. Diese sind für die Interpretation interessant, aber für ein robustes Modell gefährlich. Ein neues Feature-Modell sollte daher nicht einfach einzelne Top-Tokens übernehmen, sondern semantische Gruppen bilden.

Beispiel:

Nicht nur:

```text
#vintageshop
#vintagedress
#vintagefashion
```

sondern:

```text
creator_design_shop_signal
```

Nicht nur:

```text
wife
husband
daughter
mother
```

sondern:

```text
family_relationship_signal
```

---

## 7. Was bisher programmatisch entstanden ist

Im bisherigen Verlauf wurden beziehungsweise sollten folgende Komponenten aufgebaut werden:

### 7.1 Fair Weighted Fusion

```text
Models/HybridV4/fusion/fair_weighted_fusion.py
```

Funktion:

- Lädt BERTweet-v3-, BERTweet-v3.4- und Sparse-Feature-Wahrscheinlichkeiten.
- Sucht optimale Gewichte auf `fusion_val`.
- Evaluiert auf `test`.
- Speichert Predictions, Metrics, Grid Search Summary und Final Report.

### 7.2 Fusion Analyse

```text
Models/HybridV4/fusion/analyze_fair_fusion.py
```

Funktion:

- Classification Report
- Confusion Matrix
- Fehlergruppen `true -> pred`
- Modellübereinstimmung
- Creator-Boost-Diagnostik
- High-Confidence-Wrong-Predictions

### 7.3 Confusion Matrix Plot Script

Es wurde entschieden, für die Confusion-Matrix-Darstellung eine separate Python-Datei zu verwenden. Ziel:

- Aus gespeicherten Fusion-Prediction-Dateien Confusion-Matrix-Bilder erzeugen.
- Vergleich der Targets und Modelle visuell ermöglichen.

### 7.4 PAN Error Analysis

Es wurde begonnen, die Modellfehler mit den originalen PAN-Follower-Feeds zu verbinden. Ziel:

- Fehlergruppen direkt mit Tweet-Texten analysieren.
- Häufige Tokens und Themen pro Fehlergruppe extrahieren.
- Qualitative Hinweise für neue Features finden.

---

## 8. Gesamtinterpretation für die Bachelorarbeit

Die Experimente zeigen, dass eine einfache gewichtete Fusion aus neuronalen und Feature-basierten Modellen nur begrenzt zusätzliche Leistung bringt. Zwar ist der Ansatz methodisch sauber und liefert interpretierbare Vergleichswerte, aber die größten Fehler entstehen nicht nur durch falsche Gewichtung, sondern durch tieferliegende semantische Überlappungen zwischen Klassen.

Insbesondere bei `occupation` ist die Unterscheidung zwischen `creator` und `performer` schwierig. Die Follower-Feeds enthalten oft Fan-, Entertainment-, Social-Media- und Community-Signale, die für beide Klassen relevant sein können. Dadurch wird `performer` zur dominanten Klasse, während `creator` häufig nicht erkannt wird.

Bei `gender` zeigen sich ebenfalls Hinweise darauf, dass thematische und semantische Signale vorhanden sind, aber von den aktuellen Modellen nicht optimal genutzt werden. Die qualitative Analyse legt nahe, dass zusätzliche semantische Feature-Gruppen helfen könnten, bestimmte Fehlerfälle besser zu erkennen.

Für die Arbeit ist das ein wertvoller Befund: Auch wenn die Fusion nicht den erwarteten Performance-Gewinn liefert, zeigt sie klar, wo die Grenzen des aktuellen Ansatzes liegen und welche Richtung für Verbesserungen sinnvoll ist.

---

## 9. Empfohlene nächste Richtung

Der nächste große Hebel liegt wahrscheinlich nicht in weiterer manueller Gewichtsanpassung, sondern in einer besseren Fehlerkorrektur über semantische Features und Meta-Learning.

Empfohlen wird:

1. Semantische Signalfeatures aus PAN-Tweets bauen.
2. Diese Features auf `fusion_val` evaluieren.
3. Ein kleines Meta-Modell trainieren, das nicht nur Modellwahrscheinlichkeiten, sondern auch semantische Signale nutzt.
4. Gezielt prüfen, ob die häufigsten Fehlergruppen reduziert werden:
   - `creator -> performer`
   - `female -> male`
   - `sports -> performer`
   - `politics -> creator`

