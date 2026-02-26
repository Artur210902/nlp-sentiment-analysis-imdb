"""
Script to run all experiments sequentially.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import set_seed

set_seed(42)

def run_baseline_experiments():
    """Run baseline model experiments."""
    print("=" * 80)
    print("Running Baseline Experiments")
    print("=" * 80)
    
    from models.baseline.tfidf_logistic import train_tfidf_logistic, load_data
    from models.baseline.naive_bayes import train_naive_bayes
    
    train_df, test_df = load_data()
    
    print("\n1. TF-IDF + Logistic Regression")
    train_tfidf_logistic(train_df, test_df)
    
    print("\n2. Naive Bayes")
    train_naive_bayes(train_df, test_df)

def run_traditional_experiments():
    """Run traditional ML experiments."""
    print("\n" + "=" * 80)
    print("Running Traditional ML Experiments")
    print("=" * 80)
    
    from models.traditional.svm import train_svm, load_data
    from models.traditional.random_forest import train_random_forest
    
    train_df, test_df = load_data()
    
    print("\n1. TF-IDF + SVM")
    train_svm(train_df, test_df)
    
    print("\n2. TF-IDF + Random Forest")
    train_random_forest(train_df, test_df)

def run_transformer_experiments():
    """Run transformer experiments."""
    print("\n" + "=" * 80)
    print("Running Transformer Experiments")
    print("=" * 80)
    
    from models.transformers.bert import train_bert, load_data
    from models.transformers.roberta import train_roberta
    from models.transformers.distilbert import train_distilbert
    
    train_df, test_df = load_data()
    
    print("\n1. BERT-base-uncased")
    train_bert(train_df, test_df)
    
    print("\n2. RoBERTa-base")
    train_roberta(train_df, test_df)
    
    print("\n3. DistilBERT-base-uncased")
    train_distilbert(train_df, test_df)

def run_hybrid_experiment():
    """Run hybrid ensemble experiment."""
    print("\n" + "=" * 80)
    print("Running Hybrid Ensemble Experiment")
    print("=" * 80)
    
    from models.hybrid.ensemble_transformer_traditional import train_hybrid_ensemble, load_data
    
    train_df, test_df = load_data()
    train_hybrid_ensemble(train_df, test_df)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--baseline", action="store_true", help="Run baseline experiments")
    parser.add_argument("--traditional", action="store_true", help="Run traditional ML experiments")
    parser.add_argument("--transformers", action="store_true", help="Run transformer experiments")
    parser.add_argument("--hybrid", action="store_true", help="Run hybrid experiment")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    
    args = parser.parse_args()
    
    if args.all or args.baseline:
        run_baseline_experiments()
    
    if args.all or args.traditional:
        run_traditional_experiments()
    
    if args.all or args.transformers:
        run_transformer_experiments()
    
    if args.all or args.hybrid:
        run_hybrid_experiment()
    
    if not any([args.baseline, args.traditional, args.transformers, args.hybrid, args.all]):
        print("No experiments selected. Use --all to run everything or specify individual experiment types.")
        print("Available options: --baseline, --traditional, --transformers, --hybrid, --all")
