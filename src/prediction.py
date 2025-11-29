"""
Prediction module for making inferences with trained models
"""
import pickle
import logging
import joblib
import numpy as np
import pandas as pd
from data_preprocessing import preprocess_new_data
from config import CLASSIFIER_MODEL_PATH, VECTORIZER_MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPPredictor:
    """Handles model predictions and inference"""
    
    def __init__(self, model_path=CLASSIFIER_MODEL_PATH, vectorizer_path=VECTORIZER_MODEL_PATH):
        """
        Initialize predictor with trained model and vectorizer
        
        Args:
            model_path: Path to trained classifier
            vectorizer_path: Path to trained vectorizer
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.load_models()
    
    def load_models(self):
        """Load model and vectorizer from disk"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            logger.info(f"Loading vectorizer from {self.vectorizer_path}")
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            logger.info("Models loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise
    
    def predict_single(self, text):
        """
        Make prediction for a single text
        
        Args:
            text: Input text string
        
        Returns:
            dict: Prediction result with label and probability
        """
        try:
            # Preprocess text
            X_processed = preprocess_new_data(text, self.vectorizer_path)
            
            # Make prediction
            prediction = self.model.predict(X_processed)[0]
            
            # Get probability if available
            try:
                probabilities = self.model.predict_proba(X_processed)[0]
                confidence = float(np.max(probabilities))
            except:
                confidence = 1.0
            
            result = {
                'text': text,
                'prediction': prediction,
                'confidence': confidence,
                'success': True
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'text': text,
                'prediction': None,
                'confidence': 0.0,
                'error': str(e),
                'success': False
            }
    
    def predict_batch(self, texts):
        """
        Make predictions for multiple texts
        
        Args:
            texts: List of text strings
        
        Returns:
            list: List of prediction results
        """
        logger.info(f"Making predictions for {len(texts)} texts")
        results = []
        
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        
        return results
    
    def predict_with_scores(self, text):
        """
        Make prediction with detailed scores for all classes
        
        Args:
            text: Input text string
        
        Returns:
            dict: Detailed prediction with all class scores
        """
        try:
            # Preprocess text
            X_processed = preprocess_new_data(text, self.vectorizer_path)
            
            # Make prediction
            prediction = self.model.predict(X_processed)[0]
            
            # Get class labels
            classes = self.model.classes_
            
            # Get probabilities
            try:
                probabilities = self.model.predict_proba(X_processed)[0]
                confidence_scores = {str(label): float(prob) 
                                   for label, prob in zip(classes, probabilities)}
            except:
                confidence_scores = {str(label): 1.0/len(classes) 
                                   for label in classes}
            
            result = {
                'text': text,
                'prediction': prediction,
                'confidence_scores': confidence_scores,
                'predicted_probability': confidence_scores.get(str(prediction), 0.0),
                'success': True
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error making detailed prediction: {str(e)}")
            return {
                'text': text,
                'prediction': None,
                'confidence_scores': {},
                'error': str(e),
                'success': False
            }
    
    def predict_dataframe(self, df, text_column='text'):
        """
        Make predictions for a dataframe
        
        Args:
            df: Pandas dataframe
            text_column: Name of column containing text
        
        Returns:
            DataFrame: Original dataframe with predictions added
        """
        logger.info(f"Making predictions for dataframe with {len(df)} rows")
        
        df_copy = df.copy()
        
        predictions = []
        confidences = []
        
        for text in df_copy[text_column]:
            result = self.predict_single(text)
            predictions.append(result['prediction'])
            confidences.append(result['confidence'])
        
        df_copy['prediction'] = predictions
        df_copy['confidence'] = confidences
        
        return df_copy


def batch_predict(texts, model_path=CLASSIFIER_MODEL_PATH, vectorizer_path=VECTORIZER_MODEL_PATH):
    """
    Utility function for quick batch predictions
    
    Args:
        texts: List of text strings
        model_path: Path to trained model
        vectorizer_path: Path to trained vectorizer
    
    Returns:
        list: List of predictions
    """
    predictor = NLPPredictor(model_path, vectorizer_path)
    return predictor.predict_batch(texts)


def single_predict(text, model_path=CLASSIFIER_MODEL_PATH, vectorizer_path=VECTORIZER_MODEL_PATH):
    """
    Utility function for quick single prediction
    
    Args:
        text: Text string
        model_path: Path to trained model
        vectorizer_path: Path to trained vectorizer
    
    Returns:
        dict: Prediction result
    """
    predictor = NLPPredictor(model_path, vectorizer_path)
    return predictor.predict_single(text)


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This is a great product!",
        "I really dislike this.",
        "This movie is awesome!"
    ]
    
    try:
        predictor = NLPPredictor()
        
        print("\nSingle predictions:")
        for text in sample_texts[:1]:
            result = predictor.predict_single(text)
            print(f"Text: {result['text']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}\n")
        
        print("\nDetailed prediction:")
        result = predictor.predict_with_scores(sample_texts[0])
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence scores: {result['confidence_scores']}\n")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Models not yet trained. Run model_training.py first.")
