"""
Transformer model: BERT for sentiment analysis
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.evaluation import evaluate_model
from src.utils import set_seed, save_predictions

set_seed(42)

class IMDBDataset(Dataset):
    """Dataset class for IMDB reviews."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def load_data():
    """Load processed IMDB dataset."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, test_df

def train_bert(train_df, test_df, model_name="bert-base-uncased", 
               batch_size=16, num_epochs=3, learning_rate=2e-5):
    """
    Train BERT model for sentiment analysis.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        model_name: HuggingFace model name
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
    """
    print(f"Training {model_name} model...")
    
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
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # Create datasets
    train_dataset = IMDBDataset(X_train_split, y_train_split, tokenizer)
    val_dataset = IMDBDataset(X_val, y_val, tokenizer)
    test_dataset = IMDBDataset(X_test, y_test, tokenizer)
    
    # Training arguments
    output_dir = Path(__file__).parent.parent.parent / "results" / "models" / "bert"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate on validation set
    print("\nValidation Results:")
    val_results = trainer.evaluate(val_dataset)
    print(f"Validation Accuracy: {val_results['eval_accuracy']:.4f}")
    print(f"Validation F1: {val_results['eval_f1']:.4f}")
    
    # Evaluate on test set
    print("\nTest Results:")
    test_results = trainer.evaluate(test_dataset)
    
    # Get predictions
    test_predictions = trainer.predict(test_dataset)
    y_test_pred = np.argmax(test_predictions.predictions, axis=1)
    
    # Evaluate with our metrics
    metrics = evaluate_model(
        y_test, y_test_pred,
        "BERT-base-uncased",
        save_path="results/metrics/bert.json"
    )
    
    # Save predictions
    save_predictions(y_test_pred, y_test, "bert")
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nModel saved to {output_dir}")
    
    return model, tokenizer, metrics

if __name__ == "__main__":
    train_df, test_df = load_data()
    model, tokenizer, metrics = train_bert(train_df, test_df)
