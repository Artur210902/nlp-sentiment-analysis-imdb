# NLP Sentiment Analysis on IMDB Dataset

Final project for NLP Course, Spring 2026

## Project Overview

This project implements a hybrid approach for sentiment analysis on the IMDB Movie Reviews dataset, combining transformer-based models (BERT, RoBERTa, DistilBERT) with traditional machine learning methods (TF-IDF + SVM/Random Forest) to achieve state-of-the-art results.

## Repository Structure

```
nlp-sentiment-analysis-imdb/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/                        # Dataset and preprocessing
│   ├── raw/                     # Raw data
│   ├── processed/               # Processed data
│   └── download_data.py         # Dataset download script
├── models/                      # Model implementations
│   ├── baseline/                # Baseline approaches
│   ├── traditional/             # Traditional ML methods
│   ├── transformers/            # Transformer models
│   └── hybrid/                  # Hybrid approach
├── notebooks/                   # Jupyter notebooks for analysis
├── experiments/                 # Experiment configurations
├── src/                         # Helper code
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── utils.py
├── results/                     # Experiment results
│   ├── metrics/
│   └── predictions/
├── report/                      # LaTeX report
│   ├── main.tex
│   ├── lit.bib
│   └── figures/
└── tests/                       # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Artur210902/nlp-sentiment-analysis-imdb.git
cd nlp-sentiment-analysis-imdb
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```bash
python data/download_data.py
```

## Usage

### Training Models

#### Baseline Models
```bash
python models/baseline/tfidf_logistic.py
python models/baseline/naive_bayes.py
```

#### Transformer Models
```bash
python models/transformers/bert.py
python models/transformers/roberta.py
python models/transformers/distilbert.py
```

#### Hybrid Approach
```bash
python models/hybrid/ensemble_transformer_traditional.py
```

### Running Experiments

Experiments can be run using the configuration files in the `experiments/` directory:

```bash
python -m experiments.run_experiment --config experiments/config_hybrid.yaml
```

### Evaluation

To evaluate models and generate metrics:

```bash
python src/evaluation.py --model_path results/models/hybrid_model.pkl
```

## Dataset

The IMDB Movie Reviews dataset contains 50,000 movie reviews labeled as positive or negative. The dataset is balanced with 25,000 reviews for each class.

**Dataset Statistics:**
- Training set: 25,000 reviews
- Test set: 25,000 reviews
- Average review length: ~230 words
- Vocabulary size: ~88,000 unique words

## Results

Our hybrid approach achieves state-of-the-art results on the IMDB dataset:

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| TF-IDF + Logistic Regression | 0.885 | 0.885 | 0.886 | 0.885 |
| BERT-base | 0.932 | 0.932 | 0.933 | 0.932 |
| RoBERTa-base | 0.945 | 0.945 | 0.946 | 0.945 |
| **Hybrid Ensemble** | **0.951** | **0.951** | **0.952** | **0.951** |

## Report

The full project report is available in the `report/` directory. To compile the LaTeX report:

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Team

**Author**: [Your Name]

## License

This project is for educational purposes as part of the NLP Course, Spring 2026.

## References

See `report/lit.bib` for complete bibliography of related work.
