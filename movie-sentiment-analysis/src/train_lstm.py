"""
train_lstm.py - Deep Learning LSTM Model
=========================================

PURPOSE:
Train a PyTorch LSTM (Long Short-Term Memory) neural network for sentiment analysis.
LSTMs can capture sequential patterns and long-range dependencies in text.

WHAT YOU NEED TO DO:
1. Make sure preprocess.py has been run
2. Run this script: python src/train_lstm.py
3. Model will be saved in models/ folder
4. Training will use GPU if available (much faster)

ARCHITECTURE:
Input Text → Embedding Layer → LSTM Layers → Fully Connected → Sigmoid → Output

FUNCTIONS TO UNDERSTAND:
- build_vocabulary(): Creates word-to-index mapping
- ReviewDataset: Custom dataset for PyTorch
- LSTMSentiment: Neural network architecture
- train_epoch(): Single training pass
- evaluate_model(): Validation pass
"""

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
from collections import Counter
import sys
sys.path.append('..')
from config import *


def load_preprocessed_data():
    """
    Load preprocessed data from pickle file.
    
    RETURNS:
    --------
    X_train, X_test, y_train, y_test
    """
    print("📂 Loading preprocessed data...")
    
    try:
        with open('data/preprocessed_data.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("❌ ERROR: preprocessed_data.pkl not found!")
        print("Please run 'python src/preprocess.py' first")
        sys.exit(1)
    
    print("✅ Data loaded successfully")
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']


def build_vocabulary(X_train, max_vocab_size=MAX_VOCAB_SIZE):
    """
    Build vocabulary from training data.
    
    PARAMETERS:
    -----------
    X_train : pandas.Series
        Training text data
    max_vocab_size : int
        Maximum vocabulary size
    
    RETURNS:
    --------
    word2idx : dict
        Maps words to indices
    idx2word : dict
        Maps indices to words
    
    VOCABULARY STRUCTURE:
    - Index 0: <PAD> (padding token for sequences of different lengths)
    - Index 1: <UNK> (unknown token for words not in vocabulary)
    - Index 2+: Most frequent words
    
    EXAMPLE:
    "I love this movie" → [45, 123, 89, 234]
    """
    print(f"\n📚 Building vocabulary (max size: {max_vocab_size})...")
    
    # Tokenize all training texts
    all_words = []
    for text in X_train:
        tokens = word_tokenize(text)
        all_words.extend(tokens)
    
    # Count word frequencies
    word_counts = Counter(all_words)
    print(f"   Total unique words: {len(word_counts)}")
    
    # Keep only most frequent words
    most_common = word_counts.most_common(max_vocab_size - 2)  # -2 for PAD and UNK
    
    # Build word-to-index mapping
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, count) in enumerate(most_common, start=2):
        word2idx[word] = idx
    
    # Build index-to-word mapping (for debugging)
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    print(f"✅ Vocabulary built: {len(word2idx)} words")
    
    # Save vocabulary
    with open(f'{MODELS_DIR}vocabulary.pkl', 'wb') as f:
        pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)
    
    return word2idx, idx2word


def encode_texts(texts, word2idx):
    """
    Convert texts to sequences of indices.
    
    PARAMETERS:
    -----------
    texts : pandas.Series
        Text data
    word2idx : dict
        Word-to-index mapping
    
    RETURNS:
    --------
    encoded : list of torch.Tensors
        Sequences of word indices
    
    EXAMPLE:
    "great movie" with vocab {"<PAD>":0, "<UNK>":1, "great":2, "movie":3}
    → tensor([2, 3])
    """
    encoded = []
    unk_idx = word2idx['<UNK>']
    
    for text in texts:
        tokens = word_tokenize(text)
        indices = [word2idx.get(token, unk_idx) for token in tokens]
        encoded.append(torch.tensor(indices, dtype=torch.long))
    
    return encoded


class ReviewDataset(Dataset):
    """
    Custom PyTorch Dataset for movie reviews.
    
    WHAT THIS DOES:
    - Wraps encoded sequences and labels
    - Provides __len__ and __getitem__ for DataLoader
    - Enables batch loading during training
    
    USAGE:
    dataset = ReviewDataset(X_encoded, y)
    dataloader = DataLoader(dataset, batch_size=64)
    """
    
    def __init__(self, X_encoded, y):
        """
        Initialize dataset.
        
        PARAMETERS:
        -----------
        X_encoded : list of torch.Tensors
            Encoded text sequences
        y : pandas.Series or array
            Labels (0 or 1)
        """
        self.X = X_encoded
        self.y = torch.tensor(y.values, dtype=torch.float32)
    
    def __len__(self):
        """Return number of samples."""
        return len(self.y)
    
    def __getitem__(self, idx):
        """Get a single sample (sequence, label)."""
        return self.X[idx], self.y[idx]


def collate_batch(batch):
    """
    Collate function for DataLoader to handle variable-length sequences.
    
    WHAT THIS DOES:
    - Pads sequences to same length (longest in batch)
    - Stacks into batches
    
    EXAMPLE:
    Input:  [tensor([1,2]), tensor([3,4,5]), tensor([6])]
    Output: tensor([[1,2,0],    ← padded with 0s
                    [3,4,5],
                    [6,0,0]])
    """
    X_batch, y_batch = zip(*batch)
    X_batch = pad_sequence(X_batch, batch_first=True, padding_value=0)
    y_batch = torch.tensor(y_batch).unsqueeze(1)
    return X_batch, y_batch


class LSTMSentiment(nn.Module):
    """
    LSTM Neural Network for Sentiment Classification.
    
    ARCHITECTURE:
    1. Embedding Layer: Converts word indices to dense vectors
    2. LSTM Layers: Processes sequence, captures context
    3. Fully Connected: Maps LSTM output to single value
    4. Sigmoid: Converts to probability [0, 1]
    
    LSTM EXPLANATION:
    - Processes text sequentially (word by word)
    - Maintains hidden state (memory)
    - Can capture long-range dependencies
    - Better than simple RNN at handling long sequences
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, 
                 n_layers=N_LAYERS, dropout=DROPOUT):
        """
        Initialize the LSTM model.
        
        PARAMETERS:
        -----------
        vocab_size : int
            Size of vocabulary
        embed_dim : int
            Dimension of word embeddings
        hidden_dim : int
            LSTM hidden state dimension
        output_dim : int
            Output dimension (1 for binary classification)
        n_layers : int
            Number of LSTM layers
        dropout : float
            Dropout rate for regularization
        """
        super(LSTMSentiment, self).__init__()
        
        # Embedding layer: word index → dense vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Sigmoid activation for probability
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        PARAMETERS:
        -----------
        x : torch.Tensor
            Input batch of sequences [batch_size, seq_len]
        
        RETURNS:
        --------
        output : torch.Tensor
            Predicted probabilities [batch_size, 1]
        
        FLOW:
        x → embeddings → LSTM → take last hidden state → FC → sigmoid → output
        """
        # Embed: [batch, seq_len] → [batch, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # LSTM: outputs all hidden states, we only need final one
        # output: [batch, seq_len, hidden_dim]
        # hidden: [n_layers, batch, hidden_dim]
        output, (hidden, cell) = self.lstm(embedded)
        
        # Take last hidden state from last layer
        hidden = hidden[-1]  # [batch, hidden_dim]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Fully connected
        out = self.fc(hidden)  # [batch, 1]
        
        # Sigmoid to get probability
        out = self.sigmoid(out)
        
        return out


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    PARAMETERS:
    -----------
    model : LSTMSentiment
        The neural network
    dataloader : DataLoader
        Training data
    criterion : nn.BCELoss
        Loss function
    optimizer : torch.optim.Adam
        Optimizer
    device : torch.device
        CPU or GPU
    
    RETURNS:
    --------
    avg_loss : float
        Average loss over epoch
    
    TRAINING PROCESS:
    For each batch:
    1. Forward pass (get predictions)
    2. Calculate loss
    3. Backward pass (compute gradients)
    4. Update weights
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        # Move to device (GPU if available)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_batch)
        
        # Calculate loss
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 50 == 0:
            print(f"   Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model on test/validation data.
    
    RETURNS:
    --------
    avg_loss : float
        Average loss
    accuracy : float
        Accuracy score
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradient calculation (faster, less memory)
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            predictions = model(X_batch)
            
            # Calculate loss
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted_labels = (predictions > 0.5).float()
            correct += (predicted_labels == y_batch).sum().item()
            total += y_batch.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    """
    Main execution function.
    
    EXECUTION ORDER:
    1. Load data
    2. Build vocabulary
    3. Encode texts
    4. Create datasets and dataloaders
    5. Initialize model
    6. Train model
    7. Save model
    """
    print("="*60)
    print("STEP 3: TRAINING LSTM MODEL")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Build vocabulary
    word2idx, idx2word = build_vocabulary(X_train)
    vocab_size = len(word2idx)
    
    # Encode texts
    print("\n🔢 Encoding texts...")
    X_train_encoded = encode_texts(X_train, word2idx)
    X_test_encoded = encode_texts(X_test, word2idx)
    print(f"✅ Encoding complete")
    
    # Create datasets
    train_dataset = ReviewDataset(X_train_encoded, y_train)
    test_dataset = ReviewDataset(X_test_encoded, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    print(f"✅ DataLoaders created (batch size: {BATCH_SIZE})")
    
    # Initialize model
    print(f"\n🤖 Initializing LSTM model...")
    model = LSTMSentiment(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Embedding dim: {EMBED_DIM}")
    print(f"   Hidden dim: {HIDDEN_DIM}")
    print(f"   LSTM layers: {N_LAYERS}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\n🏋️  Training for {N_EPOCHS} epochs...")
    print("-" * 60)
    
    best_acc = 0
    for epoch in range(N_EPOCHS):
        print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'{MODELS_DIR}lstm_model.pt')
            print(f"   💾 Best model saved!")
    
    print("\n" + "="*60)
    print(f"✅ LSTM Training Complete!")
    print(f"🎯 Best Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print("="*60)
    print("\nNext step: Run 'python src/evaluate.py' for detailed evaluation")


if __name__ == "__main__":
    main()