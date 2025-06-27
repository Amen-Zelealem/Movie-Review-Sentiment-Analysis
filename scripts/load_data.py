import os
import zipfile
import pandas as pd
import subprocess

KAGGLE_DATASET = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
FILE_NAME = "IMDB Dataset.csv"
OUTPUT_FILE = "../imdb_reviews_sample.csv"


def download_from_kaggle():
    if os.path.exists(OUTPUT_FILE):
        print(f"{OUTPUT_FILE} already exists. Skipping download.")
        return

    print("Downloading dataset from Kaggle...")

    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET], check=True)

    zip_path = KAGGLE_DATASET.split("/")[-1] + ".zip"
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall()

    df = pd.read_csv(FILE_NAME)
    df = df.sample(n=5000, random_state=42).reset_index(drop=True) 
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved 5000 samples to {OUTPUT_FILE}")

    os.remove(zip_path)
    os.remove(FILE_NAME)


if __name__ == "__main__":
    download_from_kaggle()
