import os
import shutil
import json
import csv
from pathlib import Path

# Configuration
PROJECT_ROOT = Path("c:/Users/savi2/OneDrive/Desktop/btp")
DATA_ROOT = PROJECT_ROOT / "data" / "nih_cxr14"
IMAGES_DIR = DATA_ROOT / "images"
METADATA_DIR = DATA_ROOT / "metadata"
ARCHIVES_DIR = DATA_ROOT / "archives"

# Dataset options
PRIMARY_DATASET = "khanfashee/nih-chest-x-ray-14-224x224-resized"
SECONDARY_DATASET = "nih-chest-xrays/data"

# 14 Target Classes for Tier 1
TARGET_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

def check_kaggle_credentials():
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json_path.exists():
        print(f"[X] Kaggle credentials not found at {kaggle_json_path}")
        print("Please place your kaggle.json file there and ensure it has correct permissions.")
        return False
    
    try:
        with open(kaggle_json_path, 'r') as f:
            creds = json.load(f)
            if 'username' in creds and 'key' in creds:
                print("[OK] Kaggle credentials found and validated.")
                return True
            else:
                print("[X] kaggle.json is missing 'username' or 'key'.")
                return False
    except Exception as e:
        print(f"[X] Error reading kaggle.json: {e}")
        return False

def check_disk_space(required_gb=10):
    # Check space on the drive where PROJECT_ROOT is located
    total, used, free = shutil.disk_usage(PROJECT_ROOT.anchor)
    free_gb = free / (2**30)
    print(f"Disk Space: {free_gb:.2f} GB available on {PROJECT_ROOT.anchor}")
    
    if free_gb < required_gb:
        print(f"[!] Warning: Less than {required_gb}GB available. Ensure you have enough space.")
        return False
    print(f"[OK] Sufficient disk space available (>{required_gb}GB).")
    return True

def setup_directories():
    for d in [IMAGES_DIR, METADATA_DIR, ARCHIVES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created/Verified directory: {d}")

def parse_metadata(csv_path):
    if not csv_path.exists():
        print(f"[!] Metadata file {csv_path} not found. (This is expected before download)")
        return
    
    print(f"[*] Parsing {csv_path} for 14 target classes...")
    class_counts = {cls: 0 for cls in TARGET_CLASSES}
    total_records = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_records += 1
            labels = row.get('Finding Labels', '').split('|')
            for label in labels:
                if label in class_counts:
                    class_counts[label] += 1
                    
    print(f"Total records processed: {total_records}")
    print("Class distribution:")
    for cls, count in class_counts.items():
        print(f"  - {cls}: {count}")

def generate_download_commands(dataset):
    print("\n[DOWNLOAD] Kaggle CLI Download Commands:")
    print(f"kaggle datasets download -d {dataset} -p {ARCHIVES_DIR}")
    print(f"Expand-Archive -Path {ARCHIVES_DIR}\\*.zip -DestinationPath {DATA_ROOT} -Force")
    print(f"Move-Item -Path {DATA_ROOT}\\Data_entry_2017.csv -Destination {METADATA_DIR} -ErrorAction SilentlyContinue")

def main():
    print("=== NIH ChestX-ray14 Pipeline Pre-Flight Check ===")
    
    # 1. Verify Credentials
    check_kaggle_credentials()
    
    # 2. Check Disk Space (10GB for resized dataset, 42GB+ for full)
    check_disk_space(required_gb=10) 
    
    # 3. Prepare Directory Structure
    setup_directories()
    
    # 4. Check/Parse Metadata (if already downloaded)
    csv_file = METADATA_DIR / "Data_entry_2017.csv"
    parse_metadata(csv_file)
    
    # 5. Output Download Instructions
    generate_download_commands(PRIMARY_DATASET)
    
    print("==================================================")

if __name__ == "__main__":
    main()
