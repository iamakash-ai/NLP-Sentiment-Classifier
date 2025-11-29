"""
Configuration module for NLP project
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

# Model paths
PIPELINE_MODEL_PATH = MODELS_DIR / "nlp_pipeline.pkl"
VECTORIZER_MODEL_PATH = MODELS_DIR / "vectorizer.pkl"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "classifier.pkl"

# Data files
TRAINING_DATA_FILE = RAW_DATA_DIR / "training_data.csv"
TEST_DATA_FILE = RAW_DATA_DIR / "test_data.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "processed_train.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "processed_test.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Preprocessing parameters
MAX_FEATURES = 5000
MIN_DF = 2
MAX_DF = 0.95
NGRAM_RANGE = (1, 2)

# Hyperparameter tuning
CV_FOLDS = 5
N_JOBS = -1

# Model types
MODELS_TO_TEST = ['logistic_regression', 'naive_bayes', 'svm']

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET', 'nlp-project-bucket')
AWS_CODECOMMIT_REPO = os.getenv('AWS_CODECOMMIT_REPO', 'nlp-project-repo')
