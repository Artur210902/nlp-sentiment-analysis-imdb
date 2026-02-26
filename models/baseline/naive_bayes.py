"""
Baseline model: Naive Bayes with Count Vectorizer
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.evaluation import evaluate_model
from src.utils import set_seed, save_predictions

set_seed(42)

def load_data():
    """Load processed IMDB dataset."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, test_df

def train_naive_bayes(train_df, test_df, max_features=10000):
    """
    Train Naive Bayes model with Count Vectorizer.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        max_features: Maximum number of features
    """
    print("Training Naive Bayes model...")
    
    # Prepare data
    X_train = train_df['text'].values
    y_train = train_df['label'].values
    X_test = test_df['text'].values
    y_test = test_df['label'].values
    
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Count vectorization
    print("Fitting Count Vectorizer...")
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    X_train_counts = vectorizer.fit_transform(X_train_split)
    X_val_counts = vectorizer.transform(X_val)
    X_test_counts = vectorizer.transform(X_test)
    
    # Train Naive Bayes
    print("Training Multinomial Naive Bayes...")
    model = MultinomialNB(alpha=1.0)
    
    model.fit(X_train_counts, y_train_split)
    
    # Validation predictions
    y_val_pred = model.predict(X_val_counts)
    print("\nValidation Results:")
    evaluate_model(y_val, y_val_pred, "Naive Bayes (Validation)")
    
    # Test predictions
    y_test_pred = model.predict(X_test_counts)
    print("\nTest Results:")
    metrics = evaluate_model(
        y_test, y_test_pred,
        "Naive Bayes",
        save_path="results/metrics/naive_bayes.json"
    )
    
    # Save predictions
    save_predictions(y_test_pred, y_test, "naive_bayes")
    
    # Save model
    model_dir = Path(__file__).parent.parent.parent / "results" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / "naive_bayes_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    with open(model_dir / "naive_bayes_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"\nModel saved to {model_dir}")
    
    return model, vectorizer, metrics

if __name__ == "__main__":
    train_df, test_df = load_data()
    model, vectorizer, metrics = train_naive_bayes(train_df, test_df)
