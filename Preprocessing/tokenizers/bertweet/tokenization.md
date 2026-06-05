# BERTweet V3 Tokenization Summary

## Ziel

Für die nächste Modellversion der Celebrity-Profiling-Pipeline wurde eine eigene BERTweet-Tokenisierung erstellt. Ziel war es, das Truncation-Problem klassischer Transformer-Modelle zu vermeiden und längere Follower-Feeds in mehrere verarbeitbare BERTweet-Chunks zu zerlegen.

Im Gegensatz zur bestehenden BERT-uncased-Tokenisierung wurde eine neue Tokenisierung benötigt, da BERTweet ein eigenes Vocabulary und einen eigenen Tokenizer verwendet. Die BERT-uncased `input_ids` können daher nicht für BERTweet wiederverwendet werden.

---

## Modell und Tokenizer

Verwendetes Modell:

```text
vinai/bertweet-base
```

## Train:
- Celebrities: 1920
- Tokenized chunks: 481'162
- Avg chunks/celebrity: 250.61
- Missing labels: 0
- Empty chunks: 0

## Test:
- Celebrities: 400
- Tokenized chunks: 101'113
- Avg chunks/celebrity: 252.78
- Missing labels: 0
- Empty chunks: 0
