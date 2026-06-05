import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Preprocessing.tokenizers.bert.config_bert import INCLUDE_SUPP

splits = ["train", "test"]
if INCLUDE_SUPP:
    splits.append("supp")

script_path = os.path.join(PROJECT_ROOT, "Preprocessing", "tokenizers", "bert", "tokenize_bert.py")

for split in splits:
    print(f"Running BERT tokenizer for {split} split...")
    subprocess.run([sys.executable, script_path, "--split", split], check=True)