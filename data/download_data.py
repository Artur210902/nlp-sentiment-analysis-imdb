"""
Script to download and prepare the IMDB Movie Reviews dataset.
"""

import os
import urllib.request
import tarfile
import shutil
from pathlib import Path

def download_imdb_dataset(data_dir="data/raw"):
    """
    Download IMDB dataset from Stanford AI Lab.
    
    The dataset contains 50,000 movie reviews labeled as positive or negative.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = data_path / "aclImdb_v1.tar.gz"
    
    print("Downloading IMDB dataset...")
    if not tar_path.exists():
        urllib.request.urlretrieve(url, tar_path)
        print(f"Downloaded to {tar_path}")
    else:
        print(f"File already exists: {tar_path}")
    
    # Extract the archive
    extract_path = data_path / "aclImdb"
    if not extract_path.exists():
        print("Extracting archive...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_path)
        print(f"Extracted to {extract_path}")
    else:
        print(f"Archive already extracted: {extract_path}")
    
    return extract_path

def prepare_dataset_structure(source_dir, output_dir="data/processed"):
    """
    Prepare dataset structure for easier loading.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # The IMDB dataset structure:
    # aclImdb/
    #   train/
    #     pos/
    #     neg/
    #   test/
    #     pos/
    #     neg/
    
    print("Dataset structure is ready for processing.")
    print(f"Training data: {source_path / 'train'}")
    print(f"Test data: {source_path / 'test'}")
    
    return source_path

if __name__ == "__main__":
    print("=" * 50)
    print("IMDB Dataset Download Script")
    print("=" * 50)
    
    # Download dataset
    dataset_path = download_imdb_dataset()
    
    # Prepare structure
    prepare_dataset_structure(dataset_path)
    
    print("\n" + "=" * 50)
    print("Dataset download complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run preprocessing script to prepare data")
    print("2. Use notebooks for exploratory data analysis")
