"""
Script to analyze and compare results from all experiments.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.evaluation import compare_models

def load_all_metrics(results_dir="results/metrics"):
    """Load all metrics from JSON files."""
    results_path = Path(__file__).parent.parent / results_dir
    metrics_dict = {}
    
    metric_files = [
        "tfidf_logistic.json",
        "naive_bayes.json",
        "svm.json",
        "random_forest.json",
        "bert.json",
        "roberta.json",
        "distilbert.json",
        "hybrid_ensemble.json"
    ]
    
    for metric_file in metric_files:
        file_path = results_path / metric_file
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                model_name = data.get('model_name', metric_file.replace('.json', ''))
                metrics_dict[model_name] = data['metrics']
    
    return metrics_dict

def create_comparison_table(metrics_dict, output_path="report/figures/model_comparison.png"):
    """Create comparison table and visualization."""
    # Prepare data for table
    models = []
    accuracies = []
    f1_macros = []
    f1_weighteds = []
    precisions = []
    recalls = []
    
    for model_name, metrics in metrics_dict.items():
        models.append(model_name)
        accuracies.append(metrics['accuracy'])
        f1_macros.append(metrics['f1_macro'])
        f1_weighteds.append(metrics['f1_weighted'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracies,
        'F1 (macro)': f1_macros,
        'F1 (weighted)': f1_weighteds,
        'Precision': precisions,
        'Recall': recalls
    })
    
    # Sort by accuracy
    df = df.sort_values('Accuracy', ascending=False)
    
    # Save to CSV
    csv_path = Path(__file__).parent.parent / "results" / "metrics" / "comparison_table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Comparison table saved to {csv_path}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy comparison
    axes[0, 0].barh(df['Model'], df['Accuracy'])
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_xlim([0.8, 1.0])
    
    # F1 macro comparison
    axes[0, 1].barh(df['Model'], df['F1 (macro)'])
    axes[0, 1].set_xlabel('F1 Score (macro)')
    axes[0, 1].set_title('Model F1 Score (macro) Comparison')
    axes[0, 1].set_xlim([0.8, 1.0])
    
    # F1 weighted comparison
    axes[1, 0].barh(df['Model'], df['F1 (weighted)'])
    axes[1, 0].set_xlabel('F1 Score (weighted)')
    axes[1, 0].set_title('Model F1 Score (weighted) Comparison')
    axes[1, 0].set_xlim([0.8, 1.0])
    
    # All metrics comparison
    x = np.arange(len(df))
    width = 0.2
    axes[1, 1].bar(x - 2*width, df['Accuracy'], width, label='Accuracy')
    axes[1, 1].bar(x - width, df['F1 (macro)'], width, label='F1 (macro)')
    axes[1, 1].bar(x, df['F1 (weighted)'], width, label='F1 (weighted)')
    axes[1, 1].bar(x + width, df['Precision'], width, label='Precision')
    axes[1, 1].bar(x + 2*width, df['Recall'], width, label='Recall')
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('All Metrics Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(df['Model'], rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0.8, 1.0])
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent.parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison visualization saved to {output_path}")
    plt.close()
    
    return df

def print_latex_table(df, output_path="report/figures/results_table.tex"):
    """Generate LaTeX table from results."""
    output_path = Path(__file__).parent.parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    latex_table = "\\begin{table}[!tbh]\n"
    latex_table += "    \\centering\n"
    latex_table += "    \\begin{tabular}{|l|cccc|}\n"
    latex_table += "        \\hline\n"
    latex_table += "        Model & Accuracy & F1 (macro) & F1 (weighted) & Precision \\\\\n"
    latex_table += "        \\hline\n"
    
    for _, row in df.iterrows():
        model_name = row['Model'].replace('_', '\\_')
        latex_table += f"        {model_name} & {row['Accuracy']:.4f} & {row['F1 (macro)']:.4f} & {row['F1 (weighted)']:.4f} & {row['Precision']:.4f} \\\\\n"
    
    latex_table += "        \\hline\n"
    latex_table += "    \\end{tabular}\n"
    latex_table += "    \\caption{Comparison of all models on IMDB test set.}\n"
    latex_table += "    \\label{tab:results}\n"
    latex_table += "\\end{table}\n"
    
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_path}")
    print("\nLaTeX Table:")
    print(latex_table)

if __name__ == "__main__":
    print("=" * 80)
    print("Analyzing Results from All Experiments")
    print("=" * 80)
    
    # Load metrics
    metrics_dict = load_all_metrics()
    
    if not metrics_dict:
        print("No metrics found! Please run experiments first.")
        sys.exit(1)
    
    # Print comparison
    print("\nModel Comparison:")
    compare_models(metrics_dict)
    
    # Create comparison table and visualization
    df = create_comparison_table(metrics_dict)
    
    # Generate LaTeX table
    print_latex_table(df)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
