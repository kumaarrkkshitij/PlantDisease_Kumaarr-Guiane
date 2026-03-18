import os
import shutil
from zipfile import ZipFile

DATA_DIR = "dataset/plantvillage"
ZIP_PATH = "downloads/PlantVillage.zip"

os.makedirs(DATA_DIR, exist_ok=True)

# Extract dataset if zip exists
if os.path.exists(ZIP_PATH):
    with ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print(f"Dataset extracted to {DATA_DIR}")
else:
    print(f"Place the PlantVillage.zip in {ZIP_PATH} first.")