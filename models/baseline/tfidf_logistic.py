"""
Baseline model: TF-IDF + Logistic Regression
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

def train_tfidf_logistic(train_df, test_df, max_features=10000):
    """
    Train TF-IDF + Logistic Regression model.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        max_features: Maximum number of features for TF-IDF
    """
    print("Training TF-IDF + Logistic Regression model...")
    
    # Prepare data
    X_train = train_df['text'].values
    y_train = train_df['label'].values
    X_test = test_df['text'].values
    y_test = test_df['label'].values
    
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # TF-IDF vectorization
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train_split)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver='lbfgs'
    )
    
    model.fit(X_train_tfidf, y_train_split)
    
    # Validation predictions
    y_val_pred = model.predict(X_val_tfidf)
    print("\nValidation Results:")
    evaluate_model(y_val, y_val_pred, "TF-IDF + Logistic Regression (Validation)")
    
    # Test predictions
    y_test_pred = model.predict(X_test_tfidf)
    print("\nTest Results:")
    metrics = evaluate_model(
        y_test, y_test_pred, 
        "TF-IDF + Logistic Regression",
        save_path="results/metrics/tfidf_logistic.json"
    )
    
    # Save predictions
    save_predictions(y_test_pred, y_test, "tfidf_logistic")
    
    # Save model
    model_dir = Path(__file__).parent.parent.parent / "results" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / "tfidf_logistic_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    with open(model_dir / "tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"\nModel saved to {model_dir}")
    
    return model, vectorizer, metrics

if __name__ == "__main__":
    train_df, test_df = load_data()
    model, vectorizer, metrics = train_tfidf_logistic(train_df, test_df)
