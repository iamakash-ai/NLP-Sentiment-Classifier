"""
Model training module with hyperparameter tuning and evaluation
Uses sklearn pipelines with multiple classifiers
"""
import pickle
import logging
import joblib
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import prepare_training_data
from config import (
    PIPELINE_MODEL_PATH, CLASSIFIER_MODEL_PATH, VECTORIZER_MODEL_PATH,
    CV_FOLDS, N_JOBS, RANDOM_STATE, MODELS_TO_TEST, TRAINING_DATA_FILE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, tuning, and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_params = None
        self.training_history = []
        self.metrics = {}
    
    def get_model_options(self):
        """Get available model configurations"""
        models = {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                    n_jobs=N_JOBS
                ),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'naive_bayes': {
                'model': MultinomialNB(),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10]
                }
            },
            'linear_svm': {
                'model': LinearSVC(
                    random_state=RANDOM_STATE,
                    max_iter=2000,
                    dual=False
                ),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'loss': ['squared_hinge']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            }
        }
        return models
    
    def train_with_hyperparameter_tuning(self, X_train, y_train, X_val, y_val, model_name='logistic_regression'):
        """
        Train model with GridSearchCV for hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_name: Name of model to train
        
        Returns:
            dict: Trained model and best parameters
        """
        logger.info(f"Training {model_name} with hyperparameter tuning...")
        
        models = self.get_model_options()
        if model_name not in models:
            raise ValueError(f"Model {model_name} not available")
        
        model_config = models[model_name]
        base_model = model_config['model']
        params = model_config['params']
        
        # GridSearchCV with cross-validation
        grid_search = GridSearchCV(
            base_model,
            params,
            cv=CV_FOLDS,
            scoring='f1_macro',
            n_jobs=N_JOBS,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = grid_search.predict(X_val)
        val_score = f1_score(y_val, y_pred, average='macro')
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        logger.info(f"Validation F1 score: {val_score:.4f}")
        
        result = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'val_score': val_score,
            'model_name': model_name
        }
        
        self.models[model_name] = result
        return result
    
    def evaluate_model(self, model, X_test, y_test, model_name='model'):
        """
        Evaluate model on test set
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name for logging
        
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating {model_name} on test set...")
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Try to calculate ROC-AUC for binary classification
        try:
            if len(np.unique(y_test)) == 2:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        except:
            pass
        
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def compare_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Train and compare multiple models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
        
        Returns:
            dict: Comparison results
        """
        logger.info("Starting model comparison...")
        comparison_results = {}
        
        for model_name in MODELS_TO_TEST:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'='*50}")
            
            try:
                # Train with hyperparameter tuning
                train_result = self.train_with_hyperparameter_tuning(
                    X_train, y_train, X_val, y_val, model_name
                )
                
                # Evaluate on test set
                test_metrics = self.evaluate_model(
                    train_result['model'], X_test, y_test, model_name
                )
                
                comparison_results[model_name] = {
                    'train_result': train_result,
                    'test_metrics': test_metrics
                }
                
                # Track best model
                if self.best_model is None or test_metrics['f1'] > self.metrics.get('f1', 0):
                    self.best_model = train_result['model']
                    self.best_params = train_result['best_params']
                    self.metrics = test_metrics
                    logger.info(f"New best model: {model_name} with F1={test_metrics['f1']:.4f}")
            
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                comparison_results[model_name] = {'error': str(e)}
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Best Model: {self.models_by_name(comparison_results)}")
        logger.info(f"{'='*50}")
        
        return comparison_results
    
    def models_by_name(self, comparison_results):
        """Get best model name"""
        best_name = None
        best_f1 = 0
        for name, result in comparison_results.items():
            if 'test_metrics' in result:
                f1 = result['test_metrics'].get('f1', 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_name = name
        return best_name if best_name else 'None'
    
    def save_model(self, model_path=CLASSIFIER_MODEL_PATH):
        """Save best model to disk"""
        if self.best_model is None:
            raise ValueError("No model to save. Train a model first.")
        
        logger.info(f"Saving model to {model_path}")
        joblib.dump(self.best_model, model_path)
        
        # Save metrics as well
        metrics_file = str(model_path).replace('.pkl', '_metrics.json')
        with open(metrics_file, 'w') as f:
            # Convert numpy types to native Python types
            metrics_serializable = {}
            for key, value in self.metrics.items():
                if isinstance(value, dict):
                    metrics_serializable[key] = value
                elif isinstance(value, np.ndarray):
                    metrics_serializable[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    metrics_serializable[key] = float(value)
                else:
                    metrics_serializable[key] = value
            
            json.dump(metrics_serializable, f, indent=4)
        logger.info(f"Metrics saved to {metrics_file}")
    
    def load_model(self, model_path=CLASSIFIER_MODEL_PATH):
        """Load model from disk"""
        logger.info(f"Loading model from {model_path}")
        self.best_model = joblib.load(model_path)
        return self.best_model


def plot_confusion_matrix(confusion_matrix_data, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()


def main():
    """Main training pipeline"""
    logger.info("Starting NLP model training pipeline...")
    
    # Prepare data
    data = prepare_training_data(TRAINING_DATA_FILE)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Train and compare models
    trainer = ModelTrainer()
    comparison_results = trainer.compare_models(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Save best model
    trainer.save_model()
    
    # Plot confusion matrix
    cm = np.array(trainer.metrics['confusion_matrix'])
    plot_confusion_matrix(cm, 'confusion_matrix.png')
    
    # Save comparison results
    results_file = 'training_results.json'
    results_to_save = {}
    for name, result in comparison_results.items():
        if 'test_metrics' in result:
            test_metrics = result['test_metrics']
            results_to_save[name] = {
                'accuracy': float(test_metrics['accuracy']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'f1': float(test_metrics['f1']),
                'best_params': str(result['train_result']['best_params'])
            }
    
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    logger.info(f"Training results saved to {results_file}")
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
