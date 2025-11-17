"""
config.py - Project Configuration File
"""

import os

# DATA PATHS
DATA_PATH = 'data/IMDB Dataset.csv'  # Your CSV file
MODELS_DIR = 'models/'
RESULTS_DIR = 'results/'

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# DATA PREPROCESSING PARAMETERS
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_VOCAB_SIZE = 10000

# ML MODEL PARAMETERS
TFIDF_MAX_FEATURES = 5000
ML_MODELS = ['logistic_regression', 'naive_bayes', 'svm']

# LSTM MODEL PARAMETERS
EMBED_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 1
N_LAYERS = 2
DROPOUT = 0.5

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
N_EPOCHS = 5

# VISUALIZATION PARAMETERS
FIGSIZE = (10, 6)
DPI = 100

print("✅ Configuration loaded successfully!")
print(f"📁 Data path: {DATA_PATH}")
print(f"💾 Models will be saved to: {MODELS_DIR}")
print(f"📊 Results will be saved to: {RESULTS_DIR}")