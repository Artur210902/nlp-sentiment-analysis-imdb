# Contributing

This is a course project for NLP Course, Spring 2026. 

## Project Structure

Please refer to the main README.md for the project structure.

## Running Experiments

1. Download the dataset:
```bash
python data/download_data.py
```

2. Preprocess the data:
```bash
python src/preprocessing.py
```

3. Run experiments:
```bash
python experiments/run_all_experiments.py --all
```

4. Analyze results:
```bash
python experiments/analyze_results.py
```

## Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular

## Testing

Run tests with:
```bash
pytest tests/
```

## Report

The LaTeX report is in the `report/` directory. To compile:
```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
