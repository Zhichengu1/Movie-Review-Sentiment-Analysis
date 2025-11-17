"""
preprocess.py - Data Preprocessing Module
==========================================

PURPOSE:
This script loads the IMDb dataset, cleans the text data, and prepares it for model training.
It handles tokenization, stopword removal, lemmatization, and train/test splitting.

WHAT YOU NEED TO DO:
1. Place your 'IMDB Dataset.csv' file in the data/ folder
2. Run this script first: python src/preprocess.py
3. It will create preprocessed data files for later use

FUNCTIONS TO UNDERSTAND:
- load_data(): Loads the CSV file
- preprocess_text(): Cleans individual reviews
- prepare_data(): Orchestrates the full preprocessing pipeline

DATA ORGANIZATION:
Input:  data/IMDB Dataset.csv (columns: 'review', 'sentiment')
Output: data/preprocessed_data.pkl (contains X_train, X_test, y_train, y_test)
"""

import pandas as pd 
import numpy as np
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split
import sys
sys.path.append('..')
from config import *

# Download required NLTK data
print("📥 Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def load_data(file_path=DATA_PATH):
    """
    Load the IMDb dataset from CSV file.
    
    PARAMETERS:
    -----------
    file_path : str
        Path to the IMDB Dataset.csv file
    
    RETURNS:
    --------
    df : pandas.DataFrame
        DataFrame with columns ['review', 'sentiment']
    
    WHAT THIS DOES:
    - Reads CSV file
    - Checks for required columns
    - Displays basic statistics
    """
    print(f"📂 Loading data from {file_path}...")
    
    try:
        # TODO: Load the CSV file from file_path into a DataFrame called 'df'
        # HINT: Use pd.read_csv()
        df = None # Placeholder
        pass # Remove this 'pass' when you add your code
        
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {file_path}")
        print("Please ensure IMDB Dataset.csv is in the data/ folder")
        sys.exit(1)
    
    # Verify columns exist
    required_cols = ['review', 'sentiment']
    
    # TODO: Check if all columns in 'required_cols' are present in 'df.columns'
    # HINT: Use a list comprehension or a loop (e.g., if not all(col in ... for col in ...):)
    if True: # Placeholder
        # TODO: If columns are missing, print an error and exit
        # HINT: Use print() and sys.exit(1)
        pass # Remove this 'pass' when you add your code
    
    print(f"✅ Loaded {len(df)} reviews")
    print(f"📊 Class distribution:\n{df['sentiment'].value_counts()}")
    
    return df


def preprocess_text(text):
    """
    Clean and preprocess a single review text.
    
    PARAMETERS:
    -----------
    text : str
        Raw review text
    
    RETURNS:
    --------
    cleaned_text : str
        Preprocessed text
    
    PREPROCESSING STEPS:
    1. Convert to lowercase
    2. Remove HTML tags (e.g., <br />)
    3. Remove special characters and numbers
    4. Tokenize into words
    5. Remove stopwords
    6. Lemmatize words (convert to base form)
    7. Join back into string
    
    EXAMPLE:
    Input:  "This movie was AMAZING! <br /> Best film ever!"
    Output: "movie amazing best film ever"
    """
    
    # TODO 1: Convert text to lowercase
    # HINT: Use .lower()
    text = text 

    # TODO 2: Remove HTML tags
    # HINT: Use re.sub() with a pattern like r'<.*?>'
    text = text 
    
    # TODO 3: Remove special characters and numbers, keep only letters and spaces
    # HINT: Use re.sub() with a pattern like r'[^a-z\s]'
    text = text 
    
    # TODO 4: Tokenize
    # HINT: Use word_tokenize() from NLTK
    tokens = [] # Placeholder
    
    # TODO 5: Remove stopwords and short words
    # HINT: Use a list comprehension to check if a word is NOT in 'stop_words' and if len(word) > 2
    tokens = [word for word in tokens] # Placeholder
    
    # TODO 6: Lemmatize (convert words to base form)
    # HINT: Use a list comprehension and call lemmatizer.lemmatize(word) for each word
    tokens = [word for word in tokens] # Placeholder
    
    # TODO 7: Join back into string
    # HINT: Use ' '.join(...)
    cleaned_text = '' # Placeholder
    
    return cleaned_text


def prepare_data(df):
    """
    Orchestrate the full preprocessing pipeline.
    
    PARAMETERS:
    -----------
    df : pandas.DataFrame
        Raw dataframe with 'review' and 'sentiment' columns
    
    RETURNS:
    --------
    X_train, X_test : pandas.Series
        Training and test review texts
    y_train, y_test : pandas.Series
        Training and test labels (0=negative, 1=positive)
    
    PIPELINE:
    1. Apply text preprocessing to all reviews
    2. Convert sentiment labels to binary (0/1)
    3. Split into train/test sets (80/20)
    4. Save preprocessed data
    """
    print("\n🧹 Preprocessing reviews...")
    
    # TODO 1: Apply the 'preprocess_text' function to the 'review' column
    # HINT: Use df['review'].apply(...) and store it in a new column 'cleaned_review'
    df['cleaned_review'] = None # Placeholder
    
    # Show example
    print("\n📝 Example preprocessing:")
    print("ORIGINAL:", df['review'].iloc[0][:100], "...")
    print("CLEANED:", df['cleaned_review'].iloc[0][:100], "...")
    
    # TODO 2: Prepare features (X) and labels (y)
    # X should be the 'cleaned_review' column
    # y should be the 'sentiment' column mapped to 0 and 1
    # HINT: y = df['sentiment'].map({'positive': 1, 'negative': 0})
    X = None # Placeholder
    y = None # Placeholder
    
    # TODO 3: Train/test split
    print(f"\n✂️ Splitting data (test size: {TEST_SIZE})...")
    # HINT: Use train_test_split() from sklearn
    # Pass in X, y, test_size, random_state, and stratify
    X_train, X_test, y_train, y_test = (None, None, None, None) # Placeholder
    
    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Testing samples: {len(X_test)}")
    
    # TODO 4: Save preprocessed data
    output_path = 'data/preprocessed_data.pkl'
    # HINT: Use 'with open(output_path, 'wb') as f:'
    # Inside the 'with' block, use pickle.dump() to save a dictionary
    # The dictionary should be: {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    
    print(f"💾 Preprocessed data saved to {output_path}")
    
    return X_train, X_test, y_train, y_test


def main():
    """
    Main execution function.
    
    EXECUTION ORDER:
    1. Load raw data
    2. Preprocess and split data
    3. Save for later use
    """
    print("="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Preprocess and split
    # Note: This will fail until you implement the TODOs in prepare_data
    if df is not None:
        X_train, X_test, y_train, y_test = prepare_data(df)
    
    print("\n✅ Preprocessing complete!")
    print("Next step: Run 'python src/train_ml.py' to train ML models")


if __name__ == "__main__":
    main()