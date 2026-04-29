# SBERT V2 – Final Summary & Usage

## 📌 Overview

Version 2 (V2) stellt eine Verbesserung gegenüber dem MVP dar, mit Fokus auf bessere Generalisierung statt Overfitting.

### MVP (V1)

* SBERT (all-MiniLM-L6-v2)
* Mean Pooling auf Celebrity-Level
* MLP Classifier
* Problem: hohe Validation, schlechte Test-Generalisation

### V2

* SBERT Chunk-Level Embeddings
* Logistic Regression
* Celebrity-Level Soft Voting
* Ziel: robustere, realistischere Predictions

---

## 🧠 Zentrale Designentscheidungen

### 1. Chunk-Level statt Mean Pooling

**Problem (V1):**

* Mean Pooling verliert Information
* Signal wird verwässert

**Lösung (V2):**

* Training auf Chunk-Level
* Mehr Trainingssamples
* bessere Nutzung der Daten

---

### 2. Logistic Regression statt MLP

**Grund:**

* MLP hat stark overfitted
* Logistic Regression generalisiert besser auf kleinen / noisy Daten

---

### 3. Celebrity-Level Soft Voting

```text
final_probs = mean(chunk_probs)
```

**Warum:**

* Nicht jeder Chunk ist gleich gut
* Aggregation reduziert Noise aus Follower-Tweets

---

### 4. Train/Validation Split auf Celebrity-Level

**Wichtig:**

* Verhindert Data Leakage
* Alle Chunks eines Celebrities bleiben zusammen

---

## ⚖️ Rebalancing Strategie (V2.1)

### Occupation

```python
None (oder minimal creator boost ~1.05–1.10)
```

* Stärkeres Rebalancing hat Modell destabilisiert
* Creator bleibt schwierig wegen semantischer Nähe zu Performer

---

### Gender

```python
class_weight = "balanced"
```

* Reduziert Bias Richtung "male"
* Verbessert Female Recall

---

### Birthyear

```python
manuell:
{
    "1994": 1.5,
    "1985": 0.9,
    "1975": 0.95,
    "1963": 1.0,
    "1947": 1.5,
}
```

**Ergebnis:**

* weniger Collapse auf extreme Klassen
* stabilere Verteilung

---

## 🔬 Regression Experiment (Birthyear)

Getestet:

* Ridge Regression
* Weighted Ridge
* RandomForest Regression

**Ergebnis:**

```text
→ Modelle kollabieren Richtung Mittelwert
→ schlechte Macro-F1 auf Buckets
```

**Fazit:**

* Regression ungeeignet für SBERT-Embeddings in diesem Setup
* Klassifikation performt besser

---

## 📊 Finale Ergebnisse (Test)

| Task       | Macro-F1   | Bemerkung                 |
| ---------- | ---------- | ------------------------- |
| Occupation | ~0.50–0.53 | stabil, Creator schwierig |
| Gender     | ~0.60      | balanced notwendig        |
| Birthyear  | ~0.25–0.31 | schwierigster Task        |

---

## ⚠️ Wichtige Erkenntnisse

### 1. SBERT Limitierung

* Gute Semantik
* Schwach bei:

  * feinen Klassen (creator vs performer)
  * indirekten Tasks (Age aus Followern)

---

### 2. Rebalancing ist kein Allheilmittel

```text
Class weights ändern die Entscheidungsgrenze,
aber erzeugen kein neues Signal
```

---

### 3. Follower-Based Profiling ist schwierig

* hoher Noise
* indirekte Information
* schwache Korrelation

---

### 4. Regression funktioniert nicht gut

* zieht zum Mittelwert
* verliert Diskriminationsfähigkeit

---

## 🧪 Was wurde ausprobiert?

✔ SBERT + MLP
✔ SBERT + Logistic Regression
✔ Soft Voting
✔ Class Weight Balancing
✔ Manuelles Rebalancing
✔ Birthyear Regression (Ridge, RF)

---

## 🚀 Usage

### Training (V2)

```bash
python -m Models.SBERT.train_sbert_v2_lr_voting --target all
```

### Prediction (Test)

```bash
python -m Models.SBERT.predict_sbert_v2 --target all
```

### Evaluation

```bash
python -m Models.SBERT.evaluate_sbert_v2_test_predictions --target all
```

---

## 📂 Output Struktur

```text
outputs/sbert_v2/
├── checkpoints/
├── predictions/
├── metrics/
└── test_metrics/
    ├── plots/
    └── tables/
```

---

## 🧩 Fazit

V2 zeigt:

```text
✔ bessere Generalisierung als MVP
✔ robustere Pipeline
✔ saubere experimentelle Analyse
```

Aber auch:

```text
→ SBERT alleine reicht nicht für komplexe Profiling Tasks
```

👉 Grundlage für V3 (BERTweet + Hybrid Modelle)
