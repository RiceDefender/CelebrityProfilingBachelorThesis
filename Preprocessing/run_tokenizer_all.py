import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Removed "supp" to save memory
for split in ["train", "test"]:
    print(f"Running tokenizer for {split} split...")
    
    # Use subprocess.run with a list of arguments to safely handle spaces in paths on Windows
    script_path = os.path.join(PROJECT_ROOT, "Preprocessing", "tokenize_data.py")
    subprocess.run([sys.executable, script_path, "--split", split], check=True)
