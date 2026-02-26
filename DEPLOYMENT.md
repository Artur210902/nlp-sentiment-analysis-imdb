# Deployment Guide

This guide explains how to set up and deploy the NLP Sentiment Analysis project on GitHub.

## Prerequisites

- Python 3.8 or higher
- Git
- GitHub account
- (Optional) LaTeX distribution for compiling the report

## Step 1: Initialize Git Repository

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: NLP Sentiment Analysis project"
```

## Step 2: Create GitHub Repository

1. Go to GitHub and create a new repository
2. Name it: `nlp-sentiment-analysis-imdb`
3. Do NOT initialize with README, .gitignore, or license (we already have these)

## Step 3: Connect Local Repository to GitHub

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/nlp-sentiment-analysis-imdb.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Update Report with GitHub Link

1. Edit `report/main.tex`
2. Replace `\url{https://github.com/yourusername/nlp-sentiment-analysis-imdb}` with your actual GitHub URL
3. Recompile the report

## Step 5: Generate PDF Report

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The PDF will be generated as `report/main.pdf`

## Step 6: Update README

1. Edit `README.md`
2. Replace placeholder GitHub URLs with your actual repository URL
3. Update author information

## Step 7: Add Project Description on GitHub

1. Go to your repository settings
2. Add a description: "NLP Final Project: Sentiment Analysis on IMDB Dataset using Hybrid Approach"
3. Add topics: `nlp`, `sentiment-analysis`, `bert`, `roberta`, `machine-learning`, `python`

## Step 8: Upload PDF Report

1. Create a `releases` folder or use GitHub Releases
2. Upload the compiled PDF report
3. Or add it to the repository root (though it's better to keep it in `report/`)

## Verification Checklist

- [ ] Repository is public (or accessible to course instructors)
- [ ] README.md is complete and accurate
- [ ] All code files are present
- [ ] requirements.txt is up to date
- [ ] Report PDF is generated and accessible
- [ ] GitHub link in report abstract is correct
- [ ] All experiments can be run (after dataset download)
- [ ] Tests pass (if applicable)

## Notes

- The dataset will need to be downloaded separately (it's large and not included in the repo)
- Model checkpoints are excluded via .gitignore (they're too large)
- Results will be generated when experiments are run
