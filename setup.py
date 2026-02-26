"""
Setup script for the NLP Sentiment Analysis project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nlp-sentiment-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Sentiment Analysis on IMDB Movie Reviews using Hybrid Approach",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Artur210902/nlp-sentiment-analysis-imdb",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
