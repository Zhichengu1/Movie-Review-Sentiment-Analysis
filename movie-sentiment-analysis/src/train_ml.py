"""
train_ml.py - Classical Machine Learning Models
================================================

PURPOSE:
Train and evaluate traditional ML models (Logistic Regression, Naive Bayes, SVM)
using TF-IDF text vectorization. These serve as baseline models.

WHAT YOU NEED TO DO:
1. Make sure you've run preprocess.py first
2. Run this script: python src/train_ml.py
3. Models will be saved in models/ folder
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
sys.path.append('..')
from config import *


def load_preprocessed_data():
    """
    Load preprocessed data from pickle file.
    
    RETURNS:
    --------
    X_train, X_test : pandas.Series
        Preprocessed review texts
    y_train, y_test : pandas.Series
        Binary labels (0=negative, 1=positive)
    """
    print("📂 Loading preprocessed data...")
    
    try:
        # TODO: Open the 'data/preprocessed_data.pkl' file in binary read ('rb') mode
        # HINT: with open(..., 'rb') as f:
        pass # Remove this 'pass'
            # TODO: Load the data from the file using pickle.load()
            # data = ...
            
    except FileNotFoundError:
        print("❌ ERROR: preprocessed_data.pkl not found!")
        print("Please run 'python src/preprocess.py' first")
        sys.exit(1)
    
    # TODO: Extract the four components from the 'data' dictionary
    X_train = None # Placeholder
    X_test = None # Placeholder
    y_train = None # Placeholder
    y_test = None # Placeholder
    
    print(f"✅ Loaded {len(X_train)} training samples")
    print(f"✅ Loaded {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test):
    """
    Convert text to TF-IDF numerical features.
    
    PARAMETERS:
    -----------
    X_train, X_test : pandas.Series
        Text data to vectorize
    
    RETURNS:
    --------
    X_train_vec, X_test_vec : sparse matrices
        TF-IDF feature matrices
    vectorizer : TfidfVectorizer
        Fitted vectorizer (saved for later use)
    """
    print(f"\n🔢 Vectorizing text with TF-IDF (max features: {TFIDF_MAX_FEATURES})...")
    
    # TODO: Create the TfidfVectorizer
    # HINT: Use TfidfVectorizer() and pass in the parameters:
    # max_features=TFIDF_MAX_FEATURES
    # ngram_range=(1, 2)
    # min_df=5
    # max_df=0.8
    vectorizer = None # Placeholder
    
    # TODO: Fit the vectorizer on the training data AND transform it
    # HINT: Use vectorizer.fit_transform(X_train)
    X_train_vec = None # Placeholder
    
    # TODO: Transform the test data using the *fitted* vectorizer
    # HINT: Use vectorizer.transform(X_test)
    X_test_vec = None # Placeholder
    
    print(f"✅ Vectorization complete")
    print(f"   Feature matrix shape: {X_train_vec.shape}")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # TODO: Save the vectorizer to a file
    # HINT: Use 'with open(..., 'wb') as f:' and pickle.dump(vectorizer, f)
    # File path: f'{MODELS_DIR}tfidf_vectorizer.pkl'
    
    print(f"💾 Vectorizer saved")
    
    return X_train_vec, X_test_vec, vectorizer


def train_models(X_train_vec, y_train):
    """
    Train multiple ML models.
    
    PARAMETERS:
    -----------
    X_train_vec : sparse matrix
        TF-IDF features
    y_train : pandas.Series
        Training labels
    
    RETURNS:
    --------
    models : dict
        Dictionary of trained models
    """
    # TODO: Initialize an empty dictionary to store the models
    models = {}
    
    print("\n🤖 Training ML models...")
    print("-" * 60)
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    # TODO: Initialize the LogisticRegression model
    # HINT: Use LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    lr_model = None # Placeholder
    
    # TODO: Fit the model on the training data
    # HINT: lr_model.fit(...)
    
    # TODO: Add the trained model to the 'models' dictionary
    # HINT: models['logistic_regression'] = lr_model
    
    print("✅ Logistic Regression trained")
    
    # 2. Naive Bayes
    print("\nTraining Naive Bayes...")
    # TODO: Initialize the MultinomialNB model
    nb_model = None # Placeholder
    
    # TODO: Fit the model
    
    # TODO: Add the model to the 'models' dictionary
    
    print("✅ Naive Bayes trained")
    
    # 3. Support Vector Machine
    print("\nTraining SVM...")
    # TODO: Initialize the LinearSVC model
    # HINT: Use LinearSVC(max_iter=1000, random_state=RANDOM_STATE)
    svm_model = None # Placeholder
    
    # TODO: Fit the model
    
    # TODO: Add the model to the 'models' dictionary
    
    print("✅ SVM trained")
    
    # Save all models
    # TODO: Loop through the 'models' dictionary (e.g., for name, model in models.items():)
    pass # Remove this 'pass'
        # TODO: Inside the loop, save each model using pickle.dump()
        # File path: f'{MODELS_DIR}{name}.pkl'
        
        # print(f"💾 {name} saved")
    
    return models


def evaluate_model(model, X_test_vec, y_test, model_name):
    """
    Evaluate a trained model and print metrics.
    
    PARAMETERS:
    -----------
    model : sklearn model
        Trained classifier
    X_test_vec : sparse matrix
        Test features
    y_test : pandas.Series
        True labels
    model_name : str
        Name of the model
    
    RETURNS:
    --------
    accuracy : float
        Test accuracy
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name.upper()}")
    print(f"{'='*60}")
    
    # TODO: Make predictions on the test data
    # HINT: model.predict(...)
    y_pred = None # Placeholder
    
    # TODO: Calculate the accuracy
    # HINT: Use accuracy_score(y_test, y_pred)
    accuracy = 0.0 # Placeholder
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    print(f"\n📊 Classification Report:")
    # TODO: Print the classification_report
    # HINT: print(classification_report(y_test, y_pred, ...))
    
    
    # Confusion matrix
    # TODO: Calculate the confusion_matrix
    cm = np.array([[0, 0], [0, 0]]) # Placeholder
    print(f"\n🔍 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Neg    Pos")
    print(f"Actual Neg      {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"Actual Pos      {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    return accuracy


def main():
    """
    Main execution function.
    """
    print("="*60)
    print("STEP 2: TRAINING ML MODELS")
    print("="*60)
    
    # TODO: Load data
    X_train, X_test, y_train, y_test = (None, None, None, None) # Placeholder
    
    # TODO: Vectorize text
    X_train_vec, X_test_vec, vectorizer = (None, None, None) # Placeholder
    
    # TODO: Train models
    models = {} # Placeholder
    
    # Evaluate all models
    # TODO: Initialize an empty dictionary 'results'
    results = {}
    
    # TODO: Loop through the 'models' dictionary
    # HINT: for name, model in models.items():
    pass # Remove this 'pass'
        # TODO: Inside the loop, call evaluate_model()
        # accuracy = ...
        
        # TODO: Store the accuracy in the 'results' dictionary
        # results[name] = accuracy
    
    # Summary
    print("\n" + "="*60)
    print("ML MODELS SUMMARY")
    print("="*60)
    
    # TODO: Loop through the 'results' dictionary and print a summary
    # HINT: for name, acc in results.items():
    pass # Remove this 'pass'
        # print(f"{name:20s}: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n✅ ML training complete!")
    print("Next step: Run 'python src/train_lstm.py' to train deep learning model")


if __name__ == "__main__":
    main()