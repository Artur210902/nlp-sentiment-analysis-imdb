"""
Hybrid approach: Ensemble of transformer and traditional ML models
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.transformers.bert import IMDBDataset
from src.evaluation import evaluate_model
from src.utils import set_seed, save_predictions

set_seed(42)

class TransformerFeatureExtractor:
    """Extract features from transformer models."""
    
    def __init__(self, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        self.model.eval()
        self.model.to(device)
    
    def extract_features(self, texts, batch_size=32):
        """Extract features from texts using transformer."""
        features = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting features"):
            batch_texts = texts[i:i+batch_size]
            
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                # Use CLS token embedding (last hidden state)
                cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                features.append(cls_embeddings)
        
        return np.vstack(features)

def load_data():
    """Load processed IMDB dataset."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, test_df

def train_hybrid_ensemble(train_df, test_df):
    """
    Train hybrid ensemble model combining transformers and traditional ML.
    
    Strategy:
    1. Extract features from pre-trained transformers (BERT, RoBERTa)
    2. Combine with TF-IDF features
    3. Train ensemble classifier
    """
    print("Training Hybrid Ensemble Model...")
    
    # Prepare data
    X_train = train_df['text'].values
    y_train = train_df['label'].values
    X_test = test_df['text'].values
    y_test = test_df['label'].values
    
    # Split training data for validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print("\nStep 1: Extracting TF-IDF features...")
    # TF-IDF features
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_split)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print("\nStep 2: Extracting BERT features...")
    # BERT features
    bert_extractor = TransformerFeatureExtractor("bert-base-uncased")
    X_train_bert = bert_extractor.extract_features(X_train_split)
    X_val_bert = bert_extractor.extract_features(X_val)
    X_test_bert = bert_extractor.extract_features(X_test)
    
    print("\nStep 3: Extracting RoBERTa features...")
    # RoBERTa features
    roberta_extractor = TransformerFeatureExtractor("roberta-base")
    X_train_roberta = roberta_extractor.extract_features(X_train_split)
    X_val_roberta = roberta_extractor.extract_features(X_val)
    X_test_roberta = roberta_extractor.extract_features(X_test)
    
    print("\nStep 4: Combining features...")
    # Combine features
    from scipy.sparse import hstack
    X_train_combined = hstack([
        X_train_tfidf,
        X_train_bert,
        X_train_roberta
    ])
    X_val_combined = hstack([
        X_val_tfidf,
        X_val_bert,
        X_val_roberta
    ])
    X_test_combined = hstack([
        X_test_tfidf,
        X_test_bert,
        X_test_roberta
    ])
    
    # Convert to dense if needed (for some classifiers)
    print("Converting to dense format...")
    X_train_combined = X_train_combined.toarray()
    X_val_combined = X_val_combined.toarray()
    X_test_combined = X_test_combined.toarray()
    
    print("\nStep 5: Training ensemble classifier...")
    # Train ensemble classifier
    ensemble_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver='lbfgs',
        n_jobs=-1
    )
    
    ensemble_model.fit(X_train_combined, y_train_split)
    
    # Validation predictions
    y_val_pred = ensemble_model.predict(X_val_combined)
    print("\nValidation Results:")
    evaluate_model(y_val, y_val_pred, "Hybrid Ensemble (Validation)")
    
    # Test predictions
    y_test_pred = ensemble_model.predict(X_test_combined)
    print("\nTest Results:")
    metrics = evaluate_model(
        y_test, y_test_pred,
        "Hybrid Ensemble (Transformer + Traditional ML)",
        save_path="results/metrics/hybrid_ensemble.json"
    )
    
    # Save predictions
    save_predictions(y_test_pred, y_test, "hybrid_ensemble")
    
    # Save model components
    model_dir = Path(__file__).parent.parent.parent / "results" / "models" / "hybrid"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / "hybrid_ensemble_model.pkl", 'wb') as f:
        pickle.dump(ensemble_model, f)
    
    with open(model_dir / "tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    # Save feature extractors info (models are loaded from HuggingFace)
    extractor_info = {
        'bert_model': 'bert-base-uncased',
        'roberta_model': 'roberta-base',
        'tfidf_max_features': 5000
    }
    
    with open(model_dir / "extractor_info.json", 'w') as f:
        import json
        json.dump(extractor_info, f, indent=2)
    
    print(f"\nModel saved to {model_dir}")
    
    return ensemble_model, metrics

if __name__ == "__main__":
    train_df, test_df = load_data()
    model, metrics = train_hybrid_ensemble(train_df, test_df)
