# HybridV4 Fair Weighted Fusion – Zwischenergebnisse

Stand: nach Abschluss der Fair-Weighted-Fusion für `occupation` und `gender`.

## 1. Ziel des Experiments

Für HybridV4 werden drei Modellquellen kombiniert:

1. `BERTweet V3`
2. `BERTweet V3.4`
3. `Sparse Feature Model`

Die Fusion verwendet eine gewichtete Mittelung der Wahrscheinlichkeiten. Die Gewichte werden auf `fusion_val` selektiert und danach einmalig auf dem offiziellen `test`-Split evaluiert.

Für `occupation` wurde zusätzlich ein optionaler `creator_boost_alpha` getestet. Für `gender` und `birthyear` ist dieser Boost nicht relevant und bleibt `0.0`.

---

## 2. Occupation – Fair Weighted Fusion

### 2.1 Beste Validierungsselektion

Erster Lauf mit Creator-Boost:

```text
weights = [0.15, 0.69, 0.15]
creator_boost_alpha = 0.050
```

Testresultat:

```text
accuracy  = 0.6650
macro_f1  = 0.6511
```

Danach wurde ein Lauf ohne Creator-Boost durchgeführt:

```text
weights = [0.6428571428571429, 0.2142857142857143, 0.14285714285714288]
creator_boost_alpha = 0.000
```

Validierungsergebnis:

```text
fusion_val_accuracy = 0.7630
fusion_val_macro_f1 = 0.7601
```

Testresultat:

```text
accuracy  = 0.6475
macro_f1  = 0.6322
```

### 2.2 Vergleich der beiden Occupation-Fusionen

| Variante | BERTweet V3 | BERTweet V3.4 | Sparse Features | Creator Boost | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Mit Creator-Boost | 0.15 | 0.69 | 0.15 | 0.050 | 0.6650 | 0.6511 |
| Ohne Creator-Boost | 0.6429 | 0.2143 | 0.1429 | 0.000 | 0.6475 | 0.6322 |

**Zwischenfazit:**  
Die Variante mit Creator-Boost ist auf dem Testset besser als die Variante ohne Boost. Sie erreicht sowohl eine höhere Accuracy als auch einen höheren Macro-F1.

### 2.3 Occupation – Klassifikationsreport mit Creator-Boost

```text
              precision    recall  f1-score   support

      sports     0.8235    0.7000    0.7568       100
   performer     0.5443    0.8600    0.6667       100
     creator     0.5741    0.3100    0.4026       100
    politics     0.7670    0.7900    0.7783       100

    accuracy                         0.6650       400
   macro avg     0.6772    0.6650    0.6511       400
weighted avg     0.6772    0.6650    0.6511       400
```

Confusion Matrix:

```text
labels: ['sports', 'performer', 'creator', 'politics']
[[70 23  0  7]
 [ 2 86 11  1]
 [11 42 31 16]
 [ 2  7 12 79]]
```

Hauptfehler:

```text
creator    -> performer : 42
sports     -> performer : 23
creator    -> politics  : 16
politics   -> creator   : 12
creator    -> sports    : 11
performer  -> creator   : 11
```

### 2.4 Occupation – Interpretation

Das Occupation-Modell hat vor allem ein Creator-Problem. Die Klasse `creator` erreicht nur einen Recall von `0.3100`. Viele Creator werden als `performer` vorhergesagt. Das ist der deutlichste Fehlerpfad:

```text
creator -> performer: 42 Fälle
```

Gleichzeitig ist `performer` sehr recall-stark:

```text
performer recall = 0.8600
```

Das deutet darauf hin, dass das fusionierte Modell stark dazu tendiert, unsichere Profile als `performer` einzuordnen. Der Creator-Boost verbessert das Gesamtergebnis leicht, behebt aber das Grundproblem nicht vollständig.

---

## 3. Gender – Fair Weighted Fusion

### 3.1 Beste Validierungsselektion

```text
weights = [0.2777777777777778, 0.5555555555555556, 0.16666666666666666]
creator_boost_alpha = 0.0
```

Validierungsergebnis:

```text
fusion_val_accuracy = 0.7630
fusion_val_macro_f1 = 0.7594
```

Testresultat:

```text
accuracy  = 0.6725
macro_f1  = 0.6671
```

Die gespeicherten Dateien sind:

```text
outputs/hybrid_v4/fusion/predictions/gender_test_fair_weighted_fusion_predictions_best_val_w_0p28_0p56_0p17_alpha_0p000.json
outputs/hybrid_v4/fusion/metrics/gender_test_fair_weighted_fusion_metrics_best_val_w_0p28_0p56_0p17_alpha_0p000.json
outputs/hybrid_v4/fusion/metrics/gender_fair_weighted_fusion_final_report.json
```

### 3.2 Gender – Vergleich mit Einzelmodellen

| Modell | Accuracy | Macro-F1 |
|---|---:|---:|
| BERTweet V3 | 0.6675 | 0.6672 |
| BERTweet V3.4 | 0.6500 | 0.6346 |
| Feature-Modell | 0.5800 | 0.5749 |
| Fair Weighted Fusion | 0.6725 | 0.6671 |

**Zwischenfazit:**  
Die Fusion verbessert die Accuracy leicht gegenüber dem besten Einzelmodell. Der Macro-F1 bleibt aber praktisch gleich. Die Fusion ist deshalb nur eine kleine Verbesserung und kein klarer Durchbruch für `gender`.

### 3.3 Gender – Klassifikationsreport

```text
              precision    recall  f1-score   support

        male     0.6375    0.8000    0.7095       200
      female     0.7315    0.5450    0.6246       200

    accuracy                         0.6725       400
   macro avg     0.6845    0.6725    0.6671       400
weighted avg     0.6845    0.6725    0.6671       400
```

Confusion Matrix:

```text
labels: ['male', 'female']
[[160  40]
 [ 91 109]]
```

Fehler nach Richtung:

```text
female -> male   : 91
male   -> female : 40
```

### 3.4 Gender – Model Agreement

```text
all_base_agree_on_male        : 144
fusion_follows_v34            : 138
all_base_agree_on_female      : 71
fusion_follows_sparse         : 43
fusion_follows_v3             : 4
```

### 3.5 Gender – Interpretation

Das Gender-Modell zeigt eine deutliche Tendenz zu `male`:

```text
male recall   = 0.8000
female recall = 0.5450
```

Das bedeutet: Männliche Profile werden deutlich häufiger korrekt erkannt als weibliche Profile. Umgekehrt werden viele weibliche Profile fälschlich als männlich klassifiziert:

```text
female -> male: 91 Fälle
```

Die `female`-Precision ist mit `0.7315` höher als die `male`-Precision. Das bedeutet, dass `female` konservativer vorhergesagt wird: Wenn das Modell `female` vorhersagt, liegt es relativ häufig richtig; es verpasst aber viele tatsächliche weibliche Profile.

Die High-Confidence-Fehler bestätigen dieses Muster: Viele der sichersten Fehlklassifikationen sind `true=female pred=male`, oft mit Übereinstimmung aller drei Basismodelle (`v3=male`, `v34=male`, `sparse=male`).

---

## 4. Gesamtzwischenfazit nach Occupation und Gender

| Target | Beste Fusion | Test Accuracy | Test Macro-F1 | Hauptproblem |
|---|---|---:|---:|---|
| occupation | mit Creator-Boost | 0.6650 | 0.6511 | Creator wird oft als Performer erkannt |
| gender | ohne Boost | 0.6725 | 0.6671 | Female wird oft als Male erkannt |

Beide Targets zeigen, dass die Fusion zwar stabil funktioniert, aber die Fehlerstruktur der Basismodelle teilweise übernimmt. Besonders problematisch sind Fälle, in denen alle Basismodelle dieselbe falsche Richtung haben. In solchen Fällen kann die gewichtete Fusion kaum korrigieren.

---

## 5. Nächster Schritt: Birthyear

Als nächstes wird die Fair Weighted Fusion für `birthyear` ausgeführt:

```powershell
python -m Models.HybridV4.fusion.fair_weighted_fusion --target birthyear
```

Danach sollte wieder die Fehleranalyse auf den gespeicherten Predictions ausgeführt werden. Der genaue Dateiname ergibt sich aus den ausgewählten Gewichten, z. B. nach dem Schema:

```powershell
python -m Models.HybridV4.fusion.analyze_fair_fusion --predictions outputs/hybrid_v4/fusion/predictions/birthyear_test_fair_weighted_fusion_predictions_best_val_w_<...>_alpha_0p000.json
```

Nach dem Birthyear-Lauf sollten folgende Punkte ergänzt werden:

1. beste Gewichte auf `fusion_val`
2. Test Accuracy und Test Macro-F1
3. Klassifikationsreport
4. Confusion Matrix
5. wichtigste Fehlerpfade
6. Vergleich mit den Einzelmodellen
7. kurze Interpretation für die Bachelorarbeit
