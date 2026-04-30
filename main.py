# This file brings all the system components together in one file 
import subprocess
import sys
import os

# List of files to run in order
PIPELINE = [
    ("Generating synthetic labels",     "evaluation/generate_labels.py"),
    ("Preprocessing the data",              "preprocessing/preprocess_data.py"),
    ("Feature engineering",             "feature_engineering/feature_engineering.py"),
    ("Training the Isolation Forest model",       "models/training/isolation_forest.py"),
    ("Training the Autoencoder model",            "models/training/autoencoder.py"),]

def run_step(description, script_path):
    """Runs each stage and stop if it fails."""
    print(f"  STEP: {description}")
    print(f"  Running: {script_path}")

    if not os.path.exists(script_path):
        print(f"ERROR: Script not found at {script_path}")
        sys.exit(1)

    result = subprocess.run([sys.executable, script_path])

    if result.returncode != 0:
        print(f"\nERROR: {description} failed.")
        sys.exit(1)

def main():
    print("FULL PIPELINE OF THE ANOMALY DETECTION SYSTEM")

    for description, script in PIPELINE:
        run_step(description, script)

    print("PIPELINE COMPLETED SUCCESSFULLY")
    print()

if __name__ == "__main__":
    main()