# 🧩 Preprocessing Module Overview

This section describes the modular structure of the preprocessing pipeline used to build a Hugging Face–ready dataset from the PAN 2020 Celebrity Profiling data.

---

## 📂 Module Structure

```text
Preprocessing/
├── build_hf_dataset.py     # Main entry point (pipeline orchestration)
├── io_utils.py             # File loading / saving
├── normalize.py            # Text normalization
├── pan_parser.py           # PAN-specific parsing logic
├── aggregation.py          # Text construction from followers
```

---

## 🔧 Module Descriptions

---

### `build_hf_dataset.py`

**Role:**
Main entry point that orchestrates the full pipeline.

**Responsibilities:**

* Parse command-line arguments
* Load input data (labels + feeds)
* Call parsing and aggregation modules
* Build final examples
* Print debug information
* Optionally save dataset

**Important:**
This file is intentionally kept **small and readable**.
All heavy logic is delegated to modules.

---

### `io_utils.py`

**Role:**
Handles all file input/output operations.

**Functions:**

* `load_ndjson(path)`
* `save_jsonl(path, rows)`

---

### `normalize.py`

**Role:**
Standardizes tweet text.

**Special tokens:**

```
[URL]
[MENTION]
[TWEET_SEP]
[FOLLOWER_SEP]
```

---

### `pan_parser.py`

**Role:**
Handles PAN-specific parsing logic.

**Responsibilities:**

* Extract IDs
* Extract labels (gender, birthyear, occupation)
* Convert raw feed structure into:

  ```
  List[List[str]] → followers → tweets
  ```
* Group follower tweets per author

---

### `aggregation.py`

**Role:**
Builds final model input text.

**Responsibilities:**

* Combine tweets and followers using separators
* Apply limits (followers, tweets, characters)
* Avoid memory issues via incremental string building
* Create final dataset examples

---

## ⚙️ Design Principles

* Separation of concerns
* Memory-safe processing
* Modular and extensible
* Clean orchestration layer

---

## 🚀 Next Step

👉 Tokenization (Hugging Face)
👉 Token length & truncation analysis
