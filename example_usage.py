"""
Example usage script demonstrating all features
Shows how to use the NLP ML project
"""
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.prediction import NLPPredictor
from src.config import CLASSIFIER_MODEL_PATH, VECTORIZER_MODEL_PATH
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_single_prediction():
    """Example: Single text prediction"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Text Prediction")
    print("="*60)
    
    try:
        predictor = NLPPredictor()
        
        texts = [
            "This is an amazing product! I love it!",
            "Terrible experience. Won't buy again.",
            "The quality is okay, nothing special.",
        ]
        
        for text in texts:
            result = predictor.predict_single(text)
            print(f"\nText: {text}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            if not result['success']:
                print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"Error in single prediction: {e}")
        print("Make sure to train the model first: python train.py")


def example_batch_prediction():
    """Example: Batch predictions"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Prediction")
    print("="*60)
    
    try:
        predictor = NLPPredictor()
        
        texts = [
            "Great service!",
            "Poor quality",
            "Excellent!",
            "Not satisfied",
            "Perfect purchase"
        ]
        
        results = predictor.predict_batch(texts)
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                'Text': r['text'],
                'Prediction': r['prediction'],
                'Confidence': f"{r['confidence']:.2%}",
                'Status': '✓' if r['success'] else '✗'
            }
            for r in results
        ])
        
        print("\nResults:")
        print(df.to_string(index=False))
    
    except Exception as e:
        print(f"Error in batch prediction: {e}")
        print("Make sure to train the model first: python train.py")


def example_detailed_scores():
    """Example: Get detailed confidence scores"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Detailed Scores for All Classes")
    print("="*60)
    
    try:
        predictor = NLPPredictor()
        
        text = "This product exceeded my expectations!"
        result = predictor.predict_with_scores(text)
        
        print(f"\nText: {text}")
        print(f"Predicted: {result['prediction']}")
        print(f"Confidence: {result['predicted_probability']:.2%}")
        print("\nAll Class Scores:")
        for label, score in result['confidence_scores'].items():
            bar = '█' * int(score * 30)
            print(f"  {label:12} [{bar:30}] {score:.2%}")
    
    except Exception as e:
        print(f"Error: {e}")


def example_dataframe_predictions():
    """Example: Predict on DataFrame"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Predict on DataFrame")
    print("="*60)
    
    try:
        predictor = NLPPredictor()
        
        df = pd.DataFrame({
            'text': [
                'Great quality!',
                'Poor experience',
                'Excellent service',
                'Not satisfied'
            ]
        })
        
        print("\nInput DataFrame:")
        print(df)
        
        results_df = predictor.predict_dataframe(df)
        
        print("\nResults with Predictions:")
        print(results_df)
        
        print("\nPrediction Distribution:")
        print(results_df['prediction'].value_counts())
    
    except Exception as e:
        print(f"Error: {e}")


def example_detailed_prediction():
    """Example: Detailed prediction with class scores"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Detailed Prediction with Class Scores")
    print("="*60)
    
    try:
        predictor = NLPPredictor()
        
        text = "This product exceeded my expectations! Highly recommend."
        result = predictor.predict_with_scores(text)
        
        print(f"\nText: {text}")
        print(f"\nPredicted Label: {result['prediction']}")
        print(f"Confidence: {result['predicted_probability']:.2%}")
        
        print("\nConfidence Scores by Class:")
        for label, score in result['confidence_scores'].items():
            bar = '█' * int(score * 20)
            print(f"  {label:10} {bar:20} {score:.4f} ({score*100:.2f}%)")
    
    except Exception as e:
        print(f"Error in detailed prediction: {e}")
        print("Make sure to train the model first: python train.py")


def example_csv_prediction():
    """Example: Predict from CSV file"""
    print("\n" + "="*60)
    print("EXAMPLE 4: CSV File Prediction")
    print("="*60)
    
    try:
        # Create sample CSV
        sample_data = pd.DataFrame({
            'text': [
                'This is fantastic!',
                'I hate this product',
                'Average quality',
                'Best purchase ever!',
                'Complete waste of money'
            ],
            'id': range(1, 6)
        })
        
        # Save to CSV
        csv_file = 'sample_predictions.csv'
        sample_data.to_csv(csv_file, index=False)
        print(f"\nCreated sample CSV: {csv_file}")
        
        # Make predictions
        predictor = NLPPredictor()
        results_df = predictor.predict_dataframe(sample_data)
        
        print("\nPredictions:")
        print(results_df[['id', 'text', 'prediction', 'confidence']].to_string(index=False))
        
        # Save results
        results_file = 'predictions_output.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
        
        # Cleanup
        os.remove(csv_file)
        if os.path.exists(results_file):
            print(f"✓ Output file created successfully")
    
    except Exception as e:
        print(f"Error in CSV prediction: {e}")
        print("Make sure to train the model first: python train.py")


def example_performance_comparison():
    """Example: Compare model performance"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Model Performance Analysis")
    print("="*60)
    
    try:
        import json
        from pathlib import Path
        
        metrics_file = 'models/classifier_metrics.json'
        
        if Path(metrics_file).exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            print("\nModel Performance Metrics:")
            print(f"  Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
            print(f"  Recall:    {metrics.get('recall', 'N/A'):.4f}")
            print(f"  F1 Score:  {metrics.get('f1', 'N/A'):.4f}")
            
            if 'roc_auc' in metrics:
                print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        else:
            print(f"Metrics file not found: {metrics_file}")
            print("Train the model first: python train.py")
    
    except Exception as e:
        print(f"Error loading metrics: {e}")


def example_real_time_predictions():
    """Example: Real-time interactive predictions"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Interactive Real-Time Predictions")
    print("="*60)
    
    try:
        predictor = NLPPredictor()
        
        print("\nEnter texts for prediction (type 'quit' to exit):")
        print("-" * 60)
        
        while True:
            text = input("\nEnter text: ").strip()
            
            if text.lower() == 'quit':
                print("Exiting...")
                break
            
            if not text:
                continue
            
            result = predictor.predict_with_scores(text)
            
            if result['success']:
                print(f"\n✓ Prediction: {result['prediction']}")
                print(f"  Confidence: {result['predicted_probability']:.2%}")
                
                print("  Scores by class:")
                for label, score in result['confidence_scores'].items():
                    print(f"    {label:10} {score*100:6.2f}%")
            else:
                print(f"✗ Error: {result['error']}")
    
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to train the model first: python train.py")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  NLP ML PROJECT - USAGE EXAMPLES".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    # Check if models exist
    if not Path(CLASSIFIER_MODEL_PATH).exists():
        print("\n⚠️  Models not found!")
        print("\nTo use these examples, first:")
        print("  1. python create_sample_data.py    # Create sample data")
        print("  2. python train.py                  # Train models")
        print("  3. python example_usage.py          # Run examples")
        return
    
    # Run examples
    example_single_prediction()
    example_batch_prediction()
    example_detailed_prediction()
    example_csv_prediction()
    example_performance_comparison()
    
    # Interactive example (optional)
    try:
        response = input("\n\nRun interactive real-time predictions? (y/n): ").strip().lower()
        if response == 'y':
            example_real_time_predictions()
    except:
        pass
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("\nNext steps:")
    print("  - Run Streamlit app: streamlit run streamlit_app.py")
    print("  - Deploy to AWS: python ci_cd/aws_deploy.py --action create")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
