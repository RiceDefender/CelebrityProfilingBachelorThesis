import os
import json

def preview_file(filepath, label):
    print(f"\n--- Previewing: {label} ---")
    print(f"File path: {filepath}\n")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i in range(100):
                line = f.readline()
                if not line:
                    print("--- End of file reached. ---")
                    break
                
                try:
                    data = json.loads(line)
                    print(f"--- Record {i+1} ---")
                    print(json.dumps(data, indent=4))
                except json.JSONDecodeError:
                    print(f"--- Record {i+1} (Raw Text) ---")
                    print(line.strip())
                print("-" * 20)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"Error reading file: {e}")
    print("\n" + "="*50 + "\n")

def main():
    # Base directory for the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust this path if your dataset is located elsewhere relative to the script
    base_path = os.path.join(script_dir, "..", "Data", "pan20-celebrity-profiling-training-dataset-2020-02-28", "pan20-celebrity-profiling-training-dataset-2020-02-28")

    files = [
        ("labels.ndjson", "Labels data"),
        ("follower-feeds.ndjson", "Follower Feeds data"),
        ("celebrity-feeds.ndjson", "Celebrity Feeds data")
    ]

    while True:
        print("Select a file to preview:")
        for idx, (filename, label) in enumerate(files):
            print(f"{idx + 1}. {label} ({filename})")
        print("q. Quit")

        choice = input("\nEnter your choice: ").strip().lower()

        if choice == 'q':
            break
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                filename, label = files[idx]
                filepath = os.path.join(base_path, filename)
                preview_file(filepath, label)
            else:
                print("Invalid selection. Please try again.\n")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.\n")

if __name__ == "__main__":
    main()
