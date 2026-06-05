# HybridV4 Occupation Fusion – Ablation Summary

## Kontext

Für das Target `occupation` wurden zwei Varianten der Fair Weighted Fusion verglichen:

1. **Ohne Creator Boost**  
   `creator_boost_alpha = 0.00`

2. **Mit Creator Boost**  
   `creator_boost_alpha = 0.05`

Die Gewichte und der Boost wurden jeweils auf dem `fusion_val` Split ausgewählt und anschließend einmalig auf dem offiziellen `test` Split evaluiert.

---

## Beste Validierungs-Konfigurationen

### Ohne Creator Boost

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

Finales Testergebnis:

```text
occupation fair fusion test_acc=0.6475 test_macro_f1=0.6322
```

### Mit Creator Boost

```json
{
  "weights": [
    0.15384615384615385,
    0.6923076923076923,
    0.15384615384615385
  ],
  "creator_boost_alpha": 0.05,
  "fusion_val_accuracy": 0.7760416666666666,
  "fusion_val_macro_f1": 0.7743837924360413
}
```

Finales Testergebnis:

```text
occupation fair fusion test_acc=0.6650 test_macro_f1=0.6511
```

---

## Vergleich auf dem Test Split

| Metrik | Ohne Boost `alpha=0.00` | Mit Boost `alpha=0.05` | Delta |
|---|---:|---:|---:|
| Accuracy | 0.6475 | **0.6650** | **+0.0175** |
| Macro-F1 | 0.6322 | **0.6511** | **+0.0189** |
| Sports F1 | 0.7449 | **0.7568** | **+0.0119** |
| Performer F1 | 0.6641 | **0.6667** | **+0.0026** |
| Creator F1 | 0.3791 | **0.4026** | **+0.0235** |
| Politics F1 | 0.7407 | **0.7783** | **+0.0376** |

---

## Classification Report ohne Creator Boost

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

Confusion Matrix:

```text
labels: ['sports', 'performer', 'creator', 'politics']
[[73 23  0  4]
 [ 3 87  9  1]
 [13 44 29 14]
 [ 7  8 15 70]]
```

---

## Classification Report mit Creator Boost

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

---

## Wichtigste Veränderungen durch den Creator Boost

| Fehlerklasse | Ohne Boost | Mit Boost | Veränderung |
|---|---:|---:|---:|
| creator → performer | 44 | 42 | besser |
| creator → politics | 14 | 16 | schlechter |
| creator → sports | 13 | 11 | besser |
| politics → sports | 7 | 2 | deutlich besser |
| politics → performer | 8 | 7 | leicht besser |
| politics → creator | 15 | 12 | besser |
| performer → creator | 9 | 11 | schlechter |

Der Creator Boost reduziert einige Verwechslungen, insbesondere bei `politics`, erhöht aber weiterhin nicht ausreichend den Recall der Klasse `creator`. Die größte verbleibende Schwäche ist weiterhin die Verwechslung `creator → performer`.

---

## Creator Boost Diagnostics

### Ohne Boost

```text
Predicted creator:       53
True creator examples:   100
False creator positives: 24
```

### Mit Boost

```text
Predicted creator:       54
True creator examples:   100
False creator positives: 23
all predicted creator   : mean=0.6288 median=0.6641 min=0.1725 max=0.9211
true creators           : mean=0.4556 median=0.4471 min=0.0121 max=0.8992
false creator positives : mean=0.6761 median=0.6960 min=0.1725 max=0.9211
```

Interpretation: Der Creator-Boost erhöht die Anzahl der vorhergesagten `creator`-Beispiele nur minimal von 53 auf 54, verbessert aber die Qualität leicht, da die False Creator Positives von 24 auf 23 sinken. Gleichzeitig steigt der Creator F1-Score von 0.3791 auf 0.4026.

---

## Kurzinterpretation für die Bachelorarbeit

The creator boost led to a small but consistent improvement of the occupation fusion model. Compared with the no-boost ablation, the selected boost increased the overall test accuracy from 0.6475 to 0.6650 and the macro-F1 from 0.6322 to 0.6511. The largest class-level improvements were observed for `creator` and `politics`: creator F1 increased from 0.3791 to 0.4026, while politics F1 increased from 0.7407 to 0.7783. The main remaining weakness is the confusion between `creator` and `performer`, where 42 true creator profiles were still classified as performer even with the boost.

Since both the fusion weights and the boost strength were selected on the `fusion_val` split and evaluated only afterwards on the test split, this result can be reported as a validation-selected ablation rather than a test-optimized configuration.

---

## Finale Entscheidung für Occupation

Die Variante **mit Creator Boost** wird als finale HybridV4-Fusion für `occupation` verwendet:

```text
weights = [0.15384615384615385, 0.6923076923076923, 0.15384615384615385]
creator_boost_alpha = 0.05
accuracy = 0.6650
macro_f1 = 0.6511
```

Gespeicherte Artefakte:

```text
outputs/hybrid_v4/fusion/predictions/occupation_test_fair_weighted_fusion_predictions_best_val_w_0p15_0p69_0p15_alpha_0p050.json
outputs/hybrid_v4/fusion/metrics/occupation_test_fair_weighted_fusion_metrics_best_val_w_0p15_0p69_0p15_alpha_0p050.json
outputs/hybrid_v4/fusion/metrics/occupation_fair_weighted_fusion_final_report.json
```
