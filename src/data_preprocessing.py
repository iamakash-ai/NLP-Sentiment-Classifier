"""
Data preprocessing module for NLP tasks
Uses sklearn pipelines for text preprocessing and feature extraction
"""
import re
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from config import (
    TRAINING_DATA_FILE, PROCESSED_TRAIN_FILE, PROCESSED_TEST_FILE,
    VECTORIZER_MODEL_PATH, MAX_FEATURES, MIN_DF, MAX_DF, NGRAM_RANGE,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextCleaner:
    """Custom text cleaning transformer"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def fit(self, X, y=None):
        """Fit transformer"""
        return self
    
    def transform(self, X):
        """Transform data"""
        if isinstance(X, pd.Series):
            return X.apply(self.clean_text)
        return np.array([self.clean_text(text) for text in X])
    
    def fit_transform(self, X, y=None):
        """Fit and transform"""
        return self.fit(X, y).transform(X)


def create_preprocessing_pipeline():
    """
    Create sklearn pipeline for text preprocessing and feature extraction
    
    Returns:
        Pipeline: sklearn pipeline with text cleaning and TF-IDF vectorization
    """
    pipeline = Pipeline([
        ('text_cleaner', FunctionTransformer(
            lambda x: TextCleaner().fit_transform(x) if isinstance(x, (pd.Series, list)) else x,
            validate=False
        )),
        ('tfidf_vectorizer', TfidfVectorizer(
            max_features=MAX_FEATURES,
            min_df=MIN_DF,
            max_df=MAX_DF,
            ngram_range=NGRAM_RANGE,
            lowercase=True,
            stop_words='english',
            sublinear_tf=True
        ))
    ])
    
    return pipeline


def create_custom_preprocessing_pipeline():
    """
    Alternative pipeline with custom text cleaner
    """
    pipeline = Pipeline([
        ('text_cleaner', TextCleaner()),
        ('tfidf_vectorizer', TfidfVectorizer(
            max_features=MAX_FEATURES,
            min_df=MIN_DF,
            max_df=MAX_DF,
            ngram_range=NGRAM_RANGE,
            lowercase=False,
            stop_words=None,
            sublinear_tf=True
        ))
    ])
    
    return pipeline


def load_data(file_path):
    """
    Load data from CSV file
    
    Args:
        file_path: Path to CSV file with 'text' and 'label' columns
    
    Returns:
        tuple: (X, y) where X is text and y is labels
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    
    # Handle missing values
    df = df.dropna()
    
    X = df['text'].values
    y = df['label'].values
    
    logger.info(f"Loaded {len(X)} samples")
    return X, y


def preprocess_data(X, vectorizer=None, fit=True):
    """
    Preprocess text data using the pipeline
    
    Args:
        X: Input text data
        vectorizer: TfidfVectorizer instance (if None, creates new one)
        fit: Whether to fit the vectorizer
    
    Returns:
        tuple: (X_processed, vectorizer)
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            min_df=MIN_DF,
            max_df=MAX_DF,
            ngram_range=NGRAM_RANGE,
            lowercase=True,
            stop_words='english',
            sublinear_tf=True
        )
    
    cleaner = TextCleaner()
    X_cleaned = cleaner.fit_transform(X) if fit else cleaner.transform(X)
    
    if fit:
        X_processed = vectorizer.fit_transform(X_cleaned)
    else:
        X_processed = vectorizer.transform(X_cleaned)
    
    return X_processed, vectorizer


def split_data(X, y):
    """
    Split data into train, validation, and test sets
    
    Args:
        X: Input features
        y: Target labels
    
    Returns:
        tuple: (X_train, X_temp, y_train, y_temp) where temp contains val+test
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    logger.info(f"Train set size: {len(X_train)}")
    logger.info(f"Temp set size: {len(X_temp)}")
    
    return X_train, X_temp, y_train, y_temp


def prepare_training_data(csv_file_path):
    """
    Main function to prepare data for training
    
    Args:
        csv_file_path: Path to training CSV file
    
    Returns:
        dict: Dictionary with all data splits and vectorizer
    """
    logger.info("Starting data preparation...")
    
    # Load data
    X, y = load_data(csv_file_path)
    
    # Split data
    X_train, X_temp, y_train, y_temp = split_data(X, y)
    
    # Split temp into val and test
    val_size_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1-val_size_ratio,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )
    
    logger.info(f"Val set size: {len(X_val)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Preprocess data
    X_train_processed, vectorizer = preprocess_data(X_train, fit=True)
    X_val_processed, _ = preprocess_data(X_val, vectorizer=vectorizer, fit=False)
    X_test_processed, _ = preprocess_data(X_test, vectorizer=vectorizer, fit=False)
    
    # Save vectorizer
    with open(VECTORIZER_MODEL_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Vectorizer saved to {VECTORIZER_MODEL_PATH}")
    
    result = {
        'X_train': X_train_processed,
        'y_train': y_train,
        'X_val': X_val_processed,
        'y_val': y_val,
        'X_test': X_test_processed,
        'y_test': y_test,
        'vectorizer': vectorizer,
        'X_train_raw': X_train,
        'X_val_raw': X_val,
        'X_test_raw': X_test
    }
    
    logger.info("Data preparation completed")
    return result


def preprocess_new_data(texts, vectorizer_path=VECTORIZER_MODEL_PATH):
    """
    Preprocess new text data for prediction
    
    Args:
        texts: List of text strings or single string
        vectorizer_path: Path to saved vectorizer
    
    Returns:
        sparse matrix: Processed feature vectors
    """
    # Load vectorizer
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]
    
    # Clean text
    cleaner = TextCleaner()
    texts_cleaned = [cleaner.clean_text(text) for text in texts]
    
    # Vectorize
    X_processed = vectorizer.transform(texts_cleaned)
    
    return X_processed


if __name__ == "__main__":
    # Example usage
    print("Data preprocessing module ready")
