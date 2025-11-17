# Movie Review Sentiment Analysis

## Overview
This project builds a classifier to predict positive or negative sentiment from IMDb movie reviews using classical ML and deep learning (LSTM) approaches.

## Inspiration
- 👉 **RoadMap:** [Machine Learning Roadmap](https://roadmap.sh/machine-learning)  
- 🎯 **Kaggle Competition:** [Sentiment Analysis on Movie Reviews](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview)  
- 📚 **GeeksforGeeks Tutorial:** [Sentiment Analysis on IMDb Movie Reviews](https://www.geeksforgeeks.org/nlp/sentiment-analysis-on-imdb-movie-reviews/)  
- 🔥 **PyTorch Quickstart Tutorial:** [Beginner Basics](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

## Features
- Text preprocessing: tokenization, stopword removal, lemmatization
- ML models: Logistic Regression, Naive Bayes, SVM
- Deep learning: PyTorch LSTM with embeddings
- Evaluation: accuracy, precision, recall, F1-score

## Tech Stack
Python, PyTorch, TorchText, scikit-learn, NLTK, SpaCy, Matplotlib, Seaborn, Jupyter Notebook


## Project Folders Documentation

- [src folder README](movie-sentiment-analysis/README-ML.md) – contains scripts for training and evaluating models


## Setup & Run
Follow the environment setup above. Then run:

```bash
python src/preprocess.py
python src/train_ml.py
python src/train_lstm.py
python src/evaluate.py



