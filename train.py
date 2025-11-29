"""
Main training script to orchestrate the entire ML pipeline
"""
import sys
import os
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import prepare_training_data
from src.model_training import ModelTrainer
from src.config import TRAINING_DATA_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run complete training pipeline"""
    logger.info("="*60)
    logger.info("Starting NLP ML Training Pipeline")
    logger.info("="*60)
    
    # Check if training data exists
    if not Path(TRAINING_DATA_FILE).exists():
        logger.error(f"Training data not found at {TRAINING_DATA_FILE}")
        logger.info("Please create a CSV file with 'text' and 'label' columns")
        logger.info("Run create_sample_data.py to generate sample data")
        return False
    
    try:
        # Step 1: Data Preprocessing
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Data Preprocessing")
        logger.info("="*60)
        data = prepare_training_data(TRAINING_DATA_FILE)
        
        # Step 2: Model Training
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Model Training & Hyperparameter Tuning")
        logger.info("="*60)
        trainer = ModelTrainer()
        comparison_results = trainer.compare_models(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            data['X_test'], data['y_test']
        )
        
        # Step 3: Save Model
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Saving Best Model")
        logger.info("="*60)
        trainer.save_model()
        
        logger.info("\n" + "="*60)
        logger.info("✓ Training Pipeline Completed Successfully!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Review model metrics in models/classifier_metrics.json")
        logger.info("2. Run Streamlit app: streamlit run streamlit_app.py")
        logger.info("3. Deploy to AWS: python ci_cd/aws_deploy.py --action create")
        
        return True
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
