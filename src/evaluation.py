"""
Evaluation utilities for sentiment analysis models.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple
import json
from pathlib import Path

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                   model_name: str = "model",
                   save_path: str = None) -> Dict[str, float]:
    """
    Evaluate model and print/save results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save metrics JSON
        
    Returns:
        Dictionary with metrics
    """
    metrics = calculate_metrics(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results for {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Save metrics if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        metrics_dict = {
            'model_name': model_name,
            'metrics': metrics,
            'confusion_matrix': cm.tolist()
        }
        
        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"\nMetrics saved to {save_path}")
    
    return metrics

def compare_models(results: Dict[str, Dict[str, float]]) -> None:
    """
    Compare multiple models and display results in a table.
    
    Args:
        results: Dictionary mapping model names to their metrics
    """
    print(f"\n{'='*80}")
    print("Model Comparison")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Accuracy':<12} {'F1 (macro)':<12} {'F1 (weighted)':<12}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['accuracy']:<12.4f} "
              f"{metrics['f1_macro']:<12.4f} {metrics['f1_weighted']:<12.4f}")
    
    print("=" * 80)
