"""
Text preprocessing utilities for IMDB sentiment analysis.
"""

import re
import os
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm

def clean_text(text: str) -> str:
    """
    Clean and preprocess text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def load_imdb_reviews(data_dir: str, split: str = "train") -> Tuple[List[str], List[int]]:
    """
    Load IMDB reviews from directory structure.
    
    Args:
        data_dir: Path to aclImdb directory
        split: 'train' or 'test'
        
    Returns:
        Tuple of (texts, labels) where labels are 0 (negative) or 1 (positive)
    """
    data_path = Path(data_dir)
    texts = []
    labels = []
    
    # Load positive reviews
    pos_dir = data_path / split / "pos"
    if pos_dir.exists():
        for file_path in tqdm(pos_dir.glob("*.txt"), desc=f"Loading {split} positive reviews"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append(clean_text(text))
                labels.append(1)
    
    # Load negative reviews
    neg_dir = data_path / split / "neg"
    if neg_dir.exists():
        for file_path in tqdm(neg_dir.glob("*.txt"), desc=f"Loading {split} negative reviews"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append(clean_text(text))
                labels.append(0)
    
    return texts, labels

def create_dataframe(texts: List[str], labels: List[int]) -> pd.DataFrame:
    """
    Create pandas DataFrame from texts and labels.
    
    Args:
        texts: List of text strings
        labels: List of labels (0 or 1)
        
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def prepare_imdb_dataset(data_dir: str = "data/raw/aclImdb", 
                         output_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare IMDB dataset for training.
    
    Args:
        data_dir: Path to aclImdb directory
        output_dir: Directory to save processed data
        
    Returns:
        Tuple of (train_df, test_df)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading training data...")
    train_texts, train_labels = load_imdb_reviews(data_dir, split="train")
    train_df = create_dataframe(train_texts, train_labels)
    
    print("Loading test data...")
    test_texts, test_labels = load_imdb_reviews(data_dir, split="test")
    test_df = create_dataframe(test_texts, test_labels)
    
    # Save processed data
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nDataset statistics:")
    print(f"Training set: {len(train_df)} samples")
    print(f"  - Positive: {train_df['label'].sum()}")
    print(f"  - Negative: {len(train_df) - train_df['label'].sum()}")
    print(f"Test set: {len(test_df)} samples")
    print(f"  - Positive: {test_df['label'].sum()}")
    print(f"  - Negative: {len(test_df) - test_df['label'].sum()}")
    
    print(f"\nSaved processed data to:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    
    return train_df, test_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess IMDB dataset")
    parser.add_argument("--data_dir", type=str, default="data/raw/aclImdb",
                       help="Path to aclImdb directory")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                       help="Output directory for processed data")
    
    args = parser.parse_args()
    
    train_df, test_df = prepare_imdb_dataset(args.data_dir, args.output_dir)
