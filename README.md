# Movie Review Sentiment Analysis

## Overview
This project builds a classifier to predict positive or negative sentiment from IMDb movie reviews using classical ML and deep learning (LSTM) approaches.

## Features
- Text preprocessing: tokenization, stopword removal, lemmatization
- ML models: Logistic Regression, Naive Bayes, SVM
- Deep learning: PyTorch LSTM with embeddings
- Evaluation: accuracy, precision, recall, F1-score

## Tech Stack
Python, PyTorch, TorchText, scikit-learn, NLTK, SpaCy, Matplotlib, Seaborn, Jupyter Notebook

## Setup & Run
Follow the environment setup above. Then run:

```bash
python src/preprocess.py
python src/train_ml.py
python src/train_lstm.py
python src/evaluate.py
