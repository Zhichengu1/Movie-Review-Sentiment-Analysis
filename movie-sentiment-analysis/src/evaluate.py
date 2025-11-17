"""
evaluate.py - Comprehensive Model Evaluation
=============================================

PURPOSE:
Compare all trained models (ML and LSTM) with detailed metrics and visualizations.
Creates confusion matrices, ROC curves, and performance comparisons.

WHAT YOU NEED TO DO:
1. Make sure all previous scripts have been run:
   - preprocess.py
   - train_ml.py
   - train_lstm.py
2. Run this script: python src/evaluate.py
3. Results and plots will be saved in results/ folder

VISUALIZATIONS CREATED:
- Confusion matrices for each model
- ROC curves
- Model comparison bar chart
- Classification reports

FUNCTIONS TO UNDERSTAND:
- load_all_models(): Loads saved models
- evaluate_ml_model(): Evaluates sklearn models
- evaluate_lstm_model(): Evaluates PyTorch LSTM
- plot_confusion_matrix(): Creates heatmap
- plot_comparison(): Compares all models
"""

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
import sys
sys.path.append('..')
from config import *
from train_lstm import LSTMSentiment, ReviewDataset, collate_batch, encode_texts


def load_preprocessed_data():
    """Load preprocessed data."""
    print("📂 Loading preprocessed data...")
    with open('data/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print("✅ Data loaded")
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']


def load_ml_models():
    """
    Load all trained ML models and vectorizer.
    
    RETURNS:
    --------
    models : dict
        Dictionary of trained sklearn models
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer
    """
    print("\n📦 Loading ML models...")
    models = {}
    
    model_names = ['logistic_regression', 'naive_bayes', 'svm']
    for name in model_names:
        try:
            with open(f'{MODELS_DIR}{name}.pkl', 'rb') as f:
                models[name] = pickle.load(f)
            print(f"   ✅ {name} loaded")
        except FileNotFoundError:
            print(f"   ⚠️  {name} not found (run train_ml.py)")
    
    # Load vectorizer
    try:
        with open(f'{MODELS_DIR}tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("   ✅ Vectorizer loaded")
    except FileNotFoundError:
        print("   ⚠️  Vectorizer not found")
        vectorizer = None
    
    return models, vectorizer


def load_lstm_model():
    """
    Load trained LSTM model.
    
    RETURNS:
    --------
    model : LSTMSentiment
        Loaded PyTorch model
    word2idx : dict
        Vocabulary mapping
    """
    print("\n📦 Loading LSTM model...")
    
    # Load vocabulary
    try:
        with open(f'{MODELS_DIR}vocabulary.pkl', 'rb') as f:
            vocab_data = pickle.load(f)
        word2idx = vocab_data['word2idx']
        print(f"   ✅ Vocabulary loaded ({len(word2idx)} words)")
    except FileNotFoundError:
        print("   ⚠️  Vocabulary not found (run train_lstm.py)")
        return None, None
    
    # Initialize model architecture
    vocab_size = len(word2idx)
    model = LSTMSentiment(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    )
    
    # Load weights
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(f'{MODELS_DIR}lstm_model.pt', 
                                         map_location=device))
        model.to(device)
        model.eval()
        print(f"   ✅ LSTM model loaded (device: {device})")
    except FileNotFoundError:
        print("   ⚠️  LSTM weights not found (run train_lstm.py)")
        return None, None
    
    return model, word2idx


def evaluate_ml_model(model, X_test_vec, y_test, model_name):
    """
    Evaluate an ML model and return metrics.
    
    PARAMETERS:
    -----------
    model : sklearn model
        Trained classifier
    X_test_vec : sparse matrix
        TF-IDF features
    y_test : array
        True labels
    model_name : str
        Model name
    
    RETURNS:
    --------
    results : dict
        Dictionary with metrics and predictions
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Predictions
    y_pred = model.predict(X_test_vec)
    
    # Probabilities (for ROC curve)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_vec)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test_vec)
    else:
        y_proba = y_pred
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Classification report
    print(f"\n{classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])}")
    
    return {
        'name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def evaluate_lstm_model(model, word2idx, X_test, y_test):
    """
    Evaluate LSTM model.
    
    PARAMETERS:
    -----------
    model : LSTMSentiment
        Trained LSTM
    word2idx : dict
        Vocabulary
    X_test : pandas.Series
        Test texts
    y_test : pandas.Series
        True labels
    
    RETURNS:
    --------
    results : dict
        Metrics and predictions
    """
    print(f"\n{'='*60}")
    print("Evaluating: LSTM")
    print(f"{'='*60}")
    
    device = next(model.parameters()).device
    
    # Encode texts
    X_test_encoded = encode_texts(X_test, word2idx)
    
    # Create dataset and dataloader
    test_dataset = ReviewDataset(X_test_encoded, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    # Get predictions
    y_pred_proba = []
    y_pred_labels = []
    
    model.eval()
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            y_pred_proba.extend(outputs.cpu().numpy())
    
    # Convert to labels
    y_pred_proba = np.array(y_pred_proba).flatten()
    y_pred_labels = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels)
    recall = recall_score(y_test, y_pred_labels)
    f1 = f1_score(y_test, y_pred_labels)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Classification report
    print(f"\n{classification_report(y_test, y_pred_labels, target_names=['Negative', 'Positive'])}")
    
    return {
        'name': 'lstm',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred_labels,
        'y_proba': y_pred_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred_labels)
    }


def plot_confusion_matrix(cm, model_name, save_path):
    """
    Plot confusion matrix as heatmap.
    
    PARAMETERS:
    -----------
    cm : array
        Confusion matrix
    model_name : str
        Model name for title
    save_path : str
        Where to save figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"   💾 Saved: {save_path}")


def plot_roc_curves(results_list, y_test, save_path):
    """
    Plot ROC curves for all models.
    
    ROC CURVE EXPLANATION:
    - X-axis: False Positive Rate (FPR)
    - Y-axis: True Positive Rate (TPR/Recall)
    - Diagonal line: Random classifier
    - Higher AUC = Better model
    """
    plt.figure(figsize=(10, 8))
    
    for results in results_list:
        fpr, tpr, _ = roc_curve(y_test, results['y_proba'])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, linewidth=2, 
                label=f"{results['name'].upper()} (AUC = {roc_auc:.3f})")
    
    # Random classifier line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"   💾 Saved: {save_path}")


def plot_comparison(results_list, save_path):
    """
    Plot comparison bar chart of all models.
    
    METRICS COMPARED:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = [r['name'].upper() for r in results_list]
    
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results_list]
        ax.bar(x + i * width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"   💾 Saved: {save_path}")


def save_results_table(results_list, save_path):
    """
    Save results as CSV table.
    """
    df = pd.DataFrame([{
        'Model': r['name'].upper(),
        'Accuracy': f"{r['accuracy']:.4f}",
        'Precision': f"{r['precision']:.4f}",
        'Recall': f"{r['recall']:.4f}",
        'F1-Score': f"{r['f1']:.4f}"
    } for r in results_list])
    
    df.to_csv(save_path, index=False)
    print(f"   💾 Saved: {save_path}")


def main():
    """
    Main evaluation function.
    
    EXECUTION ORDER:
    1. Load all data and models
    2. Evaluate each model
    3. Generate visualizations
    4. Save results
    """
    print("="*60)
    print("STEP 4: MODEL EVALUATION")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Load models
    ml_models, vectorizer = load_ml_models()
    lstm_model, word2idx = load_lstm_model()
    
    # Vectorize for ML models
    if vectorizer:
        X_test_vec = vectorizer.transform(X_test)
    
    # Evaluate all models
    results_list = []
    
    # Evaluate ML models
    if ml_models and vectorizer:
        for name, model in ml_models.items():
            results = evaluate_ml_model(model, X_test_vec, y_test, name)
            results_list.append(results)
    
    # Evaluate LSTM
    if lstm_model and word2idx:
        results = evaluate_lstm_model(lstm_model, word2idx, X_test, y_test)
        results_list.append(results)
    
    # Generate visualizations
    if results_list:
        print("\n📊 Generating visualizations...")
        
        # Confusion matrices
        for results in results_list:
            plot_confusion_matrix(
                results['confusion_matrix'],
                results['name'],
                f"{RESULTS_DIR}confusion_matrix_{results['name']}.png"
            )
        
        # ROC curves
        plot_roc_curves(results_list, y_test, 
                       f"{RESULTS_DIR}roc_curves.png")
        
        # Comparison chart
        plot_comparison(results_list, 
                       f"{RESULTS_DIR}model_comparison.png")
        
        # Results table
        save_results_table(results_list, 
                          f"{RESULTS_DIR}results_table.csv")
        
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        for results in results_list:
            print(f"\n{results['name'].upper()}:")
            print(f"  Accuracy:  {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall:    {results['recall']:.4f}")
            print(f"  F1-Score:  {results['f1']:.4f}")
        
        print("\n✅ Evaluation complete!")
        print(f"📊 All results saved to {RESULTS_DIR}")
    else:
        print("\n⚠️  No models found to evaluate!")
        print("Please run train_ml.py and train_lstm.py first")


if __name__ == "__main__":
    main()