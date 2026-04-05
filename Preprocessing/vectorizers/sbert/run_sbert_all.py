import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Preprocessing.vectorizers.sbert.config_sbert import INCLUDE_SUPP

splits = ["train", "test"]
if INCLUDE_SUPP:
    splits.append("supp")

script_path = os.path.join(PROJECT_ROOT, "Preprocessing", "vectorizers", "sbert", "vectorize_sbert.py")

for split in splits:
    print(f"Running SBERT vectorizer for {split} split...")
    subprocess.run([sys.executable, script_path, "--split", split], check=True)