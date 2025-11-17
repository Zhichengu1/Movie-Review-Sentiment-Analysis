# Movie Review Sentiment Analysis

## Overview
This project builds a classifier to predict positive or negative sentiment from IMDb movie reviews using classical ML and deep learning (LSTM) approaches.


## Inspiration
This project was inspired by several resources on sentiment analysis:

- 👉 **RoadMap:** [Machine Learning Roadmap](https://roadmap.sh/machine-learning)  
- 🎯 **Kaggle Competition:** [Sentiment Analysis on Movie Reviews](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview)  
- 📚 **GeeksforGeeks Tutorial:** [Sentiment Analysis on IMDb Movie Reviews](https://www.geeksforgeeks.org/nlp/sentiment-analysis-on-imdb-movie-reviews/)  
- 🔥 **PyTorch Quickstart Tutorial:** [Beginner Basics](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)  
- 💡 **GitHub Repo:** [PyTorch Sentiment Analysis by Ben Trevett](https://github.com/bentrevett/pytorch-sentiment-analysis) — helped shape the LSTM implementation and overall project structure


## Project Folders Documentation

- [src folder README](movie-sentiment-analysis/README-ML.md) – contains scripts for training and evaluating models



## ✨ Features
- **Text preprocessing**: Tokenization, stopword removal, lemmatization
- **ML models**: Logistic Regression, Naive Bayes, SVM
- **Deep learning**: PyTorch LSTM with embeddings
- **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices, ROC curves

## 🛠️ Tech Stack
Python, PyTorch, TorchText, scikit-learn, NLTK, SpaCy, Matplotlib, Seaborn, Jupyter Notebook

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone/create project directory
mkdir movie-sentiment
cd movie-sentiment

# Create directories
mkdir data src models results

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Get the Dataset

1. Download **IMDB Dataset of 50K Movie Reviews** from:
   - Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
   
2. Place `IMDB Dataset.csv` in the `data/` folder:
   ```
   movie-sentiment/data/IMDB Dataset.csv
   ```

### 3. Add Project Files

Place these files in your project:
- `config.py` → Root directory
- `src/preprocess.py`
- `src/train_ml.py`
- `src/train_lstm.py`
- `src/evaluate.py`

### 4. Run the Pipeline

Execute these commands in order:

```bash
# Step 1: Preprocess data (5-10 minutes)
python src/preprocess.py

# Step 2: Train ML models (3-5 minutes)
python src/train_ml.py

# Step 3: Train LSTM model (30-60 min CPU, 5-10 min GPU)
python src/train_lstm.py

# Step 4: Evaluate and visualize (2 minutes)
python src/evaluate.py
```

### 5. View Results

Check the `results/` folder for:
- Confusion matrices for each model
- ROC curves comparison
- Model performance comparison chart
- Detailed metrics table (CSV)

---

## 📊 Expected Results

| Model               | Accuracy | F1-Score |
|---------------------|----------|----------|
| Logistic Regression | ~88%     | 0.88     |
| Naive Bayes         | ~86%     | 0.86     |
| SVM                 | ~89%     | 0.89     |
| **LSTM**            | **~90%** | **0.90** |

---

## 📁 Project Structure

```
movie-sentiment/
│
├── data/
│   ├── IMDB Dataset.csv           # Raw dataset (YOU ADD THIS)
│   └── preprocessed_data.pkl      # Auto-generated
│
├── src/
│   ├── preprocess.py              # Data cleaning & splitting
│   ├── train_ml.py                # Train ML models
│   ├── train_lstm.py              # Train LSTM model
│   └── evaluate.py                # Evaluation & visualization
│
├── models/                         # Saved models (auto-created)
├── results/                        # Plots & metrics (auto-created)
├── config.py                       # Configuration
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Data
DATA_PATH = 'data/IMDB Dataset.csv'
TEST_SIZE = 0.2  # 20% for testing

# ML Models
TFIDF_MAX_FEATURES = 5000

# LSTM
EMBED_DIM = 100
HIDDEN_DIM = 128
N_LAYERS = 2
BATCH_SIZE = 64
N_EPOCHS = 5
LEARNING_RATE = 0.001
```

---

## 📖 Documentation

Each Python file contains detailed documentation:

### `src/preprocess.py`
- Loads and cleans text data
- Removes HTML, special characters
- Tokenizes, removes stopwords, lemmatizes
- Splits into train/test sets (80/20)

### `src/train_ml.py`
- Converts text to TF-IDF features
- Trains Logistic Regression, Naive Bayes, SVM
- Evaluates on test set
- Saves trained models

### `src/train_lstm.py`
- Builds vocabulary from training data
- Encodes text as sequences
- Trains LSTM neural network
- Uses GPU if available
- Saves best model

### `src/evaluate.py`
- Loads all trained models
- Computes accuracy, precision, recall, F1
- Generates confusion matrices
- Creates ROC curves
- Saves comparison visualizations

---

## 🐛 Troubleshooting

### Dataset Not Found
```
❌ ERROR: File not found at data/IMDB Dataset.csv
```
**Solution**: Download and place the dataset in the `data/` folder

### Preprocessed Data Missing
```
❌ ERROR: preprocessed_data.pkl not found!
```
**Solution**: Run `python src/preprocess.py` first

### NLTK Data Missing
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Out of Memory (LSTM)
**Solution**: Reduce `BATCH_SIZE` in `config.py` (try 32 or 16)

### Slow Training
- **Check GPU**: `torch.cuda.is_available()` should return `True`
- **Reduce epochs**: Set `N_EPOCHS = 3` in `config.py`
- **Use smaller vocab**: Set `MAX_VOCAB_SIZE = 5000`

---

## 🎓 Key Concepts

### ML Models
- **Logistic Regression**: Linear classifier, fast and interpretable
- **Naive Bayes**: Probabilistic model based on word frequencies
- **SVM**: Finds optimal decision boundary between classes

### LSTM (Long Short-Term Memory)
- Neural network that processes sequences
- Maintains "memory" of previous words
- Captures context and word order
- Better than traditional RNN at long sequences

### Metrics
- **Accuracy**: % of correct predictions
- **Precision**: Of predicted positives, % that were correct
- **Recall**: Of actual positives, % that were detected
- **F1-Score**: Harmonic mean of precision and recall

---

## 📈 Next Steps

### Improve the Model
- Use pre-trained embeddings (GloVe, Word2Vec)
- Try bidirectional LSTM
- Implement attention mechanism
- Use Transformer models (BERT, RoBERTa)

### Add Features
- Web interface for predictions
- Real-time sentiment analysis
- Multi-class classification (1-5 stars)
- Aspect-based sentiment analysis

### Deploy
- Create Flask/FastAPI API
- Build Streamlit dashboard
- Deploy to AWS/Heroku/Google Cloud

---

## 📦 Requirements

```
numpy
pandas
scikit-learn
torch
torchtext
matplotlib
seaborn
nltk
spacy
jupyter
```

See `requirements.txt` for exact versions.

---

## 🤝 Contributing

This is a learning project. Feel free to:
- Experiment with different architectures
- Try different hyperparameters
- Add new features
- Improve documentation

---

## 📄 License

This project is for educational purposes.

Dataset: IMDB Movie Reviews (Kaggle)

---

## 🎯 Learning Objectives

After completing this project, you will understand:
- ✅ Text preprocessing techniques
- ✅ Feature extraction (TF-IDF)
- ✅ Classical ML for text classification
- ✅ Neural networks for NLP
- ✅ LSTM architecture and training
- ✅ Model evaluation metrics
- ✅ PyTorch basics
- ✅ Data pipelines

---

## 🌟 Acknowledgments

- Dataset: Andrew L. Maas et al. (Stanford)
- Frameworks: PyTorch, scikit-learn, NLTK
- Community: Kaggle, Stack Overflow

---





