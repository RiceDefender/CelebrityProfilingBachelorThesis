# BERTweetFusionV5 – Controlled Confidence Gating

## Zweck

BERTweetFusionV5 kombiniert vorhandene BERTweet-Modelle ohne weiteres Training. Die 4-Class-Modelle liefern die robuste Basisentscheidung. Auxiliary-Modelle greifen nur dann ein, wenn sie ein bekanntes Fehlerbild korrigieren können.

## Modelle

Base-Modelle:

- V3 occupation
- V3.4 stopword occupation
- V3.5 controlled sampling occupation
- V3.6 attention pooling occupation
- V3.7 occupation

Auxiliary-Modelle:

- V3 creator_binary
- V3 occupation_3class
- V3.7 occupation_group3
- V3.7 creator_performer

## Gating D

1. Base Fusion aus allen vorhandenen 4-Class-Modellen.
2. Creator Rescue, wenn mehrere unabhängige Creator-Signale aktiv sind.
3. Sports/Politics Rescue, wenn die Base-Fusion `performer` sagt, aber mehrere Signale für `sports` oder `politics` sprechen.
4. V3.7 Creator-only Specialist Override, aber nur bei sicherem Entertainment-Gate und sicherem Creator-Signal.

Der Creator/Performer-Specialist darf nicht frei auf `performer` überschreiben, weil er in den bisherigen Tests performer-lastig war.

## Ausführen

```powershell
python -m Models.BERTweetFusionV5.run_fusion_v5 --split test
```

## Outputs

```text
outputs/bertweet_fusion_v5/predictions/occupation_test_fusion_v5_gating_d_predictions.json
outputs/bertweet_fusion_v5/metrics/occupation_test_fusion_v5_gating_d_metrics.json
outputs/bertweet_fusion_v5/reports/occupation_test_fusion_v5_gating_d_report.md
```
