"""
Data validation and health check module
Validates data quality and model performance
"""
import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import (
    TRAINING_DATA_FILE, PROCESSED_TRAIN_FILE,
    CLASSIFIER_MODEL_PATH, VECTORIZER_MODEL_PATH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data quality and integrity"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
    
    def check_csv_format(self, file_path):
        """Validate CSV file format"""
        logger.info(f"Checking CSV format: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Check required columns
            if 'text' not in df.columns or 'label' not in df.columns:
                self.errors.append(
                    f"CSV must contain 'text' and 'label' columns. Found: {df.columns.tolist()}"
                )
                return False
            
            # Check for empty values
            if df['text'].isna().sum() > 0:
                missing_count = df['text'].isna().sum()
                self.warnings.append(f"Found {missing_count} missing values in 'text' column")
            
            if df['label'].isna().sum() > 0:
                missing_count = df['label'].isna().sum()
                self.warnings.append(f"Found {missing_count} missing values in 'label' column")
            
            # Check label distribution
            label_counts = df['label'].value_counts()
            self.info.append(f"Label distribution: {label_counts.to_dict()}")
            
            # Check for class imbalance
            label_ratios = label_counts / len(df)
            if (label_ratios < 0.1).any():
                self.warnings.append("Potential class imbalance detected (< 10% minority class)")
            
            # Check text length
            text_lengths = df['text'].str.len()
            self.info.append(f"Text length stats - Min: {text_lengths.min()}, Max: {text_lengths.max()}, Mean: {text_lengths.mean():.1f}")
            
            if (text_lengths < 5).sum() > 0:
                short_texts = (text_lengths < 5).sum()
                self.warnings.append(f"Found {short_texts} texts with < 5 characters")
            
            self.info.append(f"Total samples: {len(df)}")
            
            return True
        
        except Exception as e:
            self.errors.append(f"Error reading CSV: {e}")
            return False
    
    def check_data_quality(self, file_path):
        """Check overall data quality"""
        logger.info(f"Checking data quality: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Remove missing values
            df = df.dropna()
            
            # Check for duplicates
            duplicate_count = df.duplicated(subset=['text']).sum()
            if duplicate_count > 0:
                self.warnings.append(f"Found {duplicate_count} duplicate texts")
            
            # Check for special characters
            special_char_texts = df['text'].str.contains(r'[^\w\s]', regex=True).sum()
            self.info.append(f"Texts with special characters: {special_char_texts}")
            
            # Check for URLs
            url_texts = df['text'].str.contains(r'http', regex=True).sum()
            if url_texts > 0:
                self.info.append(f"Texts containing URLs: {url_texts}")
            
            return True
        
        except Exception as e:
            self.errors.append(f"Error checking data quality: {e}")
            return False
    
    def validate_models(self):
        """Check if trained models exist and are valid"""
        logger.info("Validating models...")
        
        try:
            import joblib
            import pickle
            
            # Check classifier model
            if not Path(CLASSIFIER_MODEL_PATH).exists():
                self.errors.append(f"Classifier model not found: {CLASSIFIER_MODEL_PATH}")
            else:
                try:
                    model = joblib.load(CLASSIFIER_MODEL_PATH)
                    self.info.append(f"Classifier model loaded successfully: {type(model).__name__}")
                except Exception as e:
                    self.errors.append(f"Error loading classifier: {e}")
            
            # Check vectorizer model
            if not Path(VECTORIZER_MODEL_PATH).exists():
                self.errors.append(f"Vectorizer not found: {VECTORIZER_MODEL_PATH}")
            else:
                try:
                    with open(VECTORIZER_MODEL_PATH, 'rb') as f:
                        vectorizer = pickle.load(f)
                    self.info.append(f"Vectorizer loaded successfully: {type(vectorizer).__name__}")
                except Exception as e:
                    self.errors.append(f"Error loading vectorizer: {e}")
            
            return len(self.errors) == 0
        
        except Exception as e:
            self.errors.append(f"Error validating models: {e}")
            return False
    
    def generate_report(self):
        """Generate validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'status': 'PASS' if len(self.errors) == 0 else 'FAIL'
        }
        return report
    
    def print_report(self):
        """Print validation report"""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("DATA & MODEL VALIDATION REPORT")
        print("="*60)
        print(f"Status: {report['status']}")
        print(f"Timestamp: {report['timestamp']}\n")
        
        if report['errors']:
            print("❌ ERRORS:")
            for error in report['errors']:
                print(f"  - {error}")
        
        if report['warnings']:
            print("\n⚠️  WARNINGS:")
            for warning in report['warnings']:
                print(f"  - {warning}")
        
        if report['info']:
            print("\nℹ️  INFO:")
            for info in report['info']:
                print(f"  - {info}")
        
        print("\n" + "="*60)
        
        return report


class ModelMonitor:
    """Monitor model performance and drifts"""
    
    def __init__(self, metrics_file='models/classifier_metrics.json'):
        self.metrics_file = metrics_file
        self.current_metrics = None
        self.load_metrics()
    
    def load_metrics(self):
        """Load model metrics from file"""
        try:
            if Path(self.metrics_file).exists():
                with open(self.metrics_file, 'r') as f:
                    self.current_metrics = json.load(f)
                logger.info(f"Metrics loaded: {self.metrics_file}")
            else:
                logger.warning(f"Metrics file not found: {self.metrics_file}")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    def check_performance_thresholds(self, thresholds=None):
        """Check if model meets minimum performance thresholds"""
        if thresholds is None:
            thresholds = {
                'accuracy': 0.7,
                'f1': 0.65,
                'precision': 0.65,
                'recall': 0.65
            }
        
        if not self.current_metrics:
            logger.warning("No metrics loaded")
            return None
        
        results = {}
        for metric, threshold in thresholds.items():
            if metric in self.current_metrics:
                value = self.current_metrics[metric]
                passed = value >= threshold
                results[metric] = {
                    'value': value,
                    'threshold': threshold,
                    'passed': passed,
                    'status': '✓' if passed else '✗'
                }
        
        return results
    
    def print_performance_report(self):
        """Print model performance report"""
        thresholds = self.check_performance_thresholds()
        
        if not thresholds:
            print("No performance data available")
            return
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE REPORT")
        print("="*60)
        
        all_passed = True
        for metric, result in thresholds.items():
            status = result['status']
            value = result['value']
            threshold = result['threshold']
            
            print(f"{metric.upper():12} {status} {value:.4f} (threshold: {threshold:.4f})")
            
            if not result['passed']:
                all_passed = False
        
        print("\n" + "="*60)
        if all_passed:
            print("✓ All performance thresholds met!")
        else:
            print("✗ Some thresholds not met. Consider retraining.")
        print("="*60)


def main():
    """Run validation checks"""
    logger.info("Starting validation checks...")
    
    # Data validation
    validator = DataValidator()
    
    if Path(TRAINING_DATA_FILE).exists():
        validator.check_csv_format(TRAINING_DATA_FILE)
        validator.check_data_quality(TRAINING_DATA_FILE)
    else:
        validator.errors.append(f"Training data not found: {TRAINING_DATA_FILE}")
    
    # Model validation
    validator.validate_models()
    
    # Print validation report
    report = validator.generate_report()
    validator.print_report()
    
    # Model performance monitoring
    monitor = ModelMonitor()
    if monitor.current_metrics:
        monitor.print_performance_report()
    
    return report['status'] == 'PASS'


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
