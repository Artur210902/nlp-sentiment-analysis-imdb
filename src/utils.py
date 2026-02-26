"""
Utility functions for the project.
"""

import random
import numpy as np
import torch
from pathlib import Path

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def save_predictions(y_pred: np.ndarray, y_true: np.ndarray, 
                    model_name: str, output_dir: str = "results/predictions"):
    """
    Save model predictions to file.
    
    Args:
        y_pred: Predicted labels
        y_true: True labels
        model_name: Name of the model
        output_dir: Output directory
    """
    ensure_dir(output_dir)
    
    output_path = Path(output_dir) / f"{model_name}_predictions.npz"
    np.savez(output_path, y_pred=y_pred, y_true=y_true)
    
    print(f"Predictions saved to {output_path}")
