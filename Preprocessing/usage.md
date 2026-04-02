Here is an example of how to use the preprocessing functions in this module:

```bash
cd .. # Go to the root directory of the repository (CelebrityProfilingBachelorThesis)
python Preprocessing/build_hf_dataset.py --split train --max-followers 3 --max-tweets-per-follower 20 --max-chars 10000 --debug
```

This command will run the `build_hf_dataset_v2.py` script with the following arguments:
- `--split train`: This specifies that we want to preprocess the training split of the dataset.
- `--max-followers 3`: This limits the number of followers to 3 per celebrity.
- `--max-tweets-per-follower 20`: This limits the number of tweets to 20 per follower.
- `--max-chars 10000`: This limits the number of characters to 10,000 per celebrity.
- `--debug`: This enables debug mode, which will print additional information during preprocessing.

To save the preprocessed dataset, you can use the `--save` argument followed by the desired file name:

```bash
python Preprocessing/build_hf_dataset.py --split train --max-followers 3 --max-tweets-per-follower 20 --max-chars 10000 --save-jsonl artifacts/train_hf.jsonl
```
This command will save the preprocessed dataset in JSONL format to the specified path (`artifacts/train_hf.jsonl`).
