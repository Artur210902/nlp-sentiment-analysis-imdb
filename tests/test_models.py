"""
Unit tests for model implementations.
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import clean_text, create_dataframe
from src.evaluation import calculate_metrics
from src.utils import set_seed

set_seed(42)

class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions."""
    
    def test_clean_text(self):
        """Test text cleaning function."""
        from src.preprocessing import clean_text
        
        # Test HTML removal
        text = "<p>This is a test</p>"
        cleaned = clean_text(text)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("</p>", cleaned)
        
        # Test URL removal
        text = "Check out https://example.com for more info"
        cleaned = clean_text(text)
        self.assertNotIn("https://example.com", cleaned)
        
        # Test whitespace normalization
        text = "This   has    multiple    spaces"
        cleaned = clean_text(text)
        self.assertNotIn("   ", cleaned)

class TestEvaluation(unittest.TestCase):
    """Test evaluation functions."""
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        from src.evaluation import calculate_metrics
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1_macro'], 1.0)
        
        # Test with errors
        y_pred = np.array([0, 0, 1, 0, 1])
        metrics = calculate_metrics(y_true, y_pred)
        self.assertLess(metrics['accuracy'], 1.0)

class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        from src.utils import set_seed
        import random
        
        set_seed(42)
        val1 = random.random()
        
        set_seed(42)
        val2 = random.random()
        
        self.assertEqual(val1, val2)

if __name__ == '__main__':
    unittest.main()
