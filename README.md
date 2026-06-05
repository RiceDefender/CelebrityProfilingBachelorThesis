# Celebrity Profiling Bachelor Thesis

This repository contains code and experiments for the PAN 2020 Celebrity Profiling task, developed as part of a bachelor thesis.

## Project Structure

*   **`DataAnalyser/`**: Scripts for exploring and previewing the dataset.
*   **`ExistingEvaluationCode/`**: Evaluation scripts and baseline code.
*   **`Models/`**: Core implementations, training scripts, and configurations for the transformer models.
*   **`ModifiedPriceHodgeCode/`**: Adapted scripts used for extracting sparse features and computing baseline metrics. This code is originally based on the implementation provided by Hodge and Price.
*   **`Preprocessing/`**: Code for preprocessing the dataset into a format suitable for Hugging Face models.
*   **`data/`**: Directory for training and test datasets (not included in version control).

## Data

The dataset used in this project is the **PAN 2020 Celebrity Profiling Training Dataset**.
For details on the dataset structure, refer to the scripts in `DataAnalyser/`.

### Preprocessing (PAN 2020 Dataset)

The dataset is transformed into a Hugging Face–ready format using a modular preprocessing pipeline.

Each example is constructed by aggregating tweets from multiple followers of a celebrity profile.

Pipeline steps:
1. Load labels and follower feeds
2. Match by author ID
3. Normalize tweets (URLs, mentions)
4. Aggregate tweets per follower
5. Merge followers into a single text sequence
6. Apply length constraints (followers, tweets, characters)

Output format:
- id
- text
- birthyear
- gender
- occupation
### ⚠️ Memory Requirements

The preprocessing step can require a large amount of RAM depending on the parameters.

Reasons:
- Multiple followers per profile
- Multiple tweets per follower
- Large concatenated text sequences

Recommendations:
- Use `--max-followers`
- Use `--max-tweets-per-follower`
- Use `--max-chars`
- Avoid loading extremely large configurations on low-RAM systems

If memory errors occur:
- Reduce parameters
- Use smaller batch sizes
- Consider chunk-based processing

## References

*   **Task Overview**: [PAN 2020 Celebrity Profiling Overview](https://ceur-ws.org/Vol-2696/paper_259.pdf)
*   **Evaluation Code**: The code in `ExistingEvaluationCode/` is adapted from the official evaluation scripts provided at [Zenodo (Record 4461887)](https://zenodo.org/records/4461887).
*   **Hodge & Price Baseline**: The scripts located in `ModifiedPriceHodgeCode/` are an adapted version of the original source code developed by A. Hodge and S. Price for their classical machine learning baseline approach in the [Celebrity Profiling task](https://ceur-ws.org/Vol-2696/paper_230.pdf).
