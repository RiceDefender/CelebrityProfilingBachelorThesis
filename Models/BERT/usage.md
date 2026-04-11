To start the training process for the BERT model, you can run the following command in your terminal:

```bash
python -m Models.BERT.train_bert --target all
```

If you want to train the model on a specific target, you can replace `all` with the desired target name. For example, to train the model on the `occupation` target, you can run:

```bash
python -m Models.BERT.train_bert --target occupation
```

For GPU usage you need to install the `torch` library with CUDA support. You can do this by running:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```