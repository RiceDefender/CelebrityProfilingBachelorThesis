To start the training process for the BERT model, you can run the following command in your terminal:

```bash
python -m Models.BERT.train_bert --target all
```

If you want to train the model on a specific target, you can replace `all` with the desired target name. For example, to train the model on the `occupation` target, you can run:

```bash
python -m Models.BERT.train_bert --target occupation
```