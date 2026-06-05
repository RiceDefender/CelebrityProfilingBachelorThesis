# HybridV4 Semantic Error Analysis – Zwischenstand

## Ziel

Für HybridV4 wurden semantische Signale aus den PAN20 Follower-Feeds analysiert, um typische Fehlergruppen besser zu verstehen. Der Fokus lag auf `gender` und `occupation`, insbesondere auf:

- `female_to_male` und `male_to_female` bei Gender
- `creator_to_performer`, `sports_to_performer` und `politics_to_creator` bei Occupation

## Bisher erzeugte Artefakte

```text
outputs/hybrid_v4/error_analysis/semantic_signals/
├── gender_semantic_signal_report.md
├── gender_semantic_signal_report.json
├── occupation_semantic_signal_report.md
└── occupation_semantic_signal_report.json
```

```text
outputs/hybrid_v4/error_analysis/semantic_signal_ratios/
├── gender_semantic_ratio_report.md
├── gender_semantic_ratio_report.json
├── occupation_semantic_ratio_report.md
└── occupation_semantic_ratio_report.json
```

```text
outputs/hybrid_v4/error_analysis/correctability_candidates/
├── gender_correctability_candidates.md
├── gender_correctability_candidates.json
├── occupation_correctability_candidates.md
└── occupation_correctability_candidates.json
```

## Gender: wichtigste Beobachtungen

Die Gender-Ratios zeigen ein interessantes Muster. Besonders `relationship_minus_male_role` trennt die korrekt klassifizierten Gruppen sichtbar:

| group | n | female_minus_male_role | relationship_minus_male_role |
| --- | ---: | ---: | ---: |
| correct_female | 109 | 0.7493 | 1.9906 |
| correct_male | 160 | 0.0033 | 0.4850 |
| female_to_male | 91 | 0.3525 | 0.5987 |
| male_to_female | 40 | 0.3621 | 2.4442 |

Interpretation:

- `correct_female` hat deutlich höhere relationship-orientierte Sprache als `correct_male`.
- `male_to_female` liegt sogar höher als `correct_female` bei `relationship_minus_male_role`.
- `female_to_male` liegt näher an `correct_male`, aber mit leicht erhöhtem `female_minus_male_role`.

Damit ist `relationship_minus_male_role` ein gutes Erklärungssignal für Unsicherheit oder Fehler bei Gender, aber nicht automatisch eine robuste Korrekturregel.

Die Correctability-Analyse bestätigt das: Nur etwa ein Viertel der Gender-Fehler liegt näher am semantischen Zentrum der wahren Klasse als an der vorhergesagten Klasse.

| error_group | n | closer_to_true | rate | mean_margin |
| --- | ---: | ---: | ---: | ---: |
| female_to_male | 91 | 23 | 0.2527 | -0.6664 |
| male_to_female | 40 | 10 | 0.2500 | -0.8211 |

## Occupation: wichtigste Beobachtungen

Bei Occupation sind die Ratio-Features stärker. Besonders `creator_to_performer` zeigt ein klares Performer-/Fan-Muster:

| Feature | creator_to_performer | correct_creator | correct_performer |
| --- | ---: | ---: | ---: |
| performer_minus_creator | 2.2529 | 0.9036 | 2.2125 |
| fan_minus_creator | 0.5983 | -0.0413 | 0.8266 |
| performer_fan_minus_creator | 4.0400 | 2.2823 | 4.0655 |

Interpretation:

- `creator_to_performer` unterscheidet sich deutlich von `correct_creator`.
- Gleichzeitig ist `creator_to_performer` sehr ähnlich zu `correct_performer`.
- Das erklärt, warum diese Fehlergruppe schwer zu korrigieren ist: Viele Creator haben tatsächlich performer-/fan-nahe Follower-Sprache.

Die Correctability-Analyse zeigt:

| error_group | n | closer_to_true | rate | mean_margin |
| --- | ---: | ---: | ---: | ---: |
| creator_to_performer | 44 | 20 | 0.4545 | -0.3932 |
| creator_to_politics | 14 | 5 | 0.3571 | -0.3322 |
| creator_to_sports | 13 | 6 | 0.4615 | -0.4103 |
| sports_to_performer | 23 | 9 | 0.3913 | -0.2982 |
| politics_to_creator | 15 | 12 | 0.8000 | 0.6782 |

Der interessanteste Korrekturkandidat ist damit nicht `creator_to_performer`, sondern `politics_to_creator`: 80% dieser Fälle liegen semantisch näher an `correct_politics` als an `correct_creator`.

## Nächster technischer Schritt

Als nächster Schritt wurde ein Rule-Simulation-Skript erstellt:

```text
Models/HybridV4/analysis/simulate_semantic_correction_rules.py
```

Es simuliert vorsichtige semantische Post-hoc-Regeln, zum Beispiel:

- `pred=creator` und `politics_minus_creator` sehr hoch → simuliere `politics`
- `pred=performer` und Performer-/Fan-Vorteil gegenüber Creator sehr niedrig → simuliere `creator`
- bei Gender nur exploratorische Flags, keine harte Empfehlung

Output:

```text
outputs/hybrid_v4/error_analysis/semantic_rule_simulation/
├── gender_semantic_rule_simulation.md
├── gender_semantic_rule_simulation.json
├── occupation_semantic_rule_simulation.md
├── occupation_semantic_rule_simulation.json
└── semantic_rule_simulation_summary.md
```

## Wichtiger methodischer Hinweis

Diese Rule-Simulation nutzt bestehende Test-Predictions und darf daher nicht als final optimierte Test-Performance verkauft werden. Sie ist eine Error-Analysis-Simulation.

Saubere Formulierung für die Thesis:

> The semantic rule simulation is used as a post-hoc diagnostic probe. Since it is derived from existing test-set predictions, it is not reported as a tuned model improvement. Instead, it identifies which error types are semantically plausible candidates for future validation-based meta-fusion.

## Empfehlung danach

Falls die Simulation vielversprechende Regeln zeigt, sollte der nächste echte Modellschritt ein kleines Meta-Modell auf einem sauberen Validation-/Fusion-Split sein. Mögliche Features:

- BERTweet-v3 Wahrscheinlichkeiten
- BERTweet-v3.4 Wahrscheinlichkeiten
- Sparse-Feature-Wahrscheinlichkeiten
- Prediction margins
- Model-agreement flags
- semantische Signal- und Ratio-Features

Wichtig: Keine Schwellenwerte auf dem Testsplit optimieren.
