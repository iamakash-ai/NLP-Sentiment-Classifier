#!/usr/bin/env python3
"""
Quick Start Script - Run the entire NLP project locally
This script helps beginners get up and running in minutes
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║   NLP ML Project - Quick Start                            ║
    ║   End-to-End Machine Learning with AWS CI/CD             ║
    ╚═══════════════════════════════════════════════════════════╝
    """)


def check_prerequisites():
    """Check if all prerequisites are installed"""
    logger.info("Checking prerequisites...")
    
    requirements = {
        'Python 3.8+': sys.version_info >= (3, 8),
    }
    
    all_good = True
    for req, status in requirements.items():
        symbol = '✓' if status else '✗'
        print(f"  {symbol} {req}")
        if not status:
            all_good = False
    
    return all_good


def install_dependencies():
    """Install Python dependencies"""
    logger.info("Installing dependencies...")
    
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
            check=True,
            capture_output=True
        )
        
        requirements_file = Path('requirements.txt')
        if requirements_file.exists():
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                check=True
            )
            logger.info("✓ Dependencies installed successfully")
            return True
        else:
            logger.error("requirements.txt not found")
            return False
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def download_nltk_data():
    """Download required NLTK data"""
    logger.info("Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("✓ NLTK data downloaded")
        return True
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {e}")
        return False


def create_sample_data():
    """Create sample training data"""
    logger.info("Creating sample training data...")
    
    try:
        subprocess.run(
            [sys.executable, 'create_sample_data.py'],
            check=True
        )
        logger.info("✓ Sample data created")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating sample data: {e}")
        return False


def train_model():
    """Train the model"""
    logger.info("Training model (this may take 2-5 minutes)...")
    
    try:
        subprocess.run(
            [sys.executable, 'train.py'],
            check=True
        )
        logger.info("✓ Model training completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error training model: {e}")
        return False


def validate_setup():
    """Validate that everything is set up correctly"""
    logger.info("Validating setup...")
    
    try:
        subprocess.run(
            [sys.executable, 'validate_data.py'],
            check=True
        )
        logger.info("✓ Setup validation completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Validation failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                    SETUP COMPLETED! ✓                     ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Your NLP ML project is ready to use!
    
    📊 NEXT STEPS:
    
    1️⃣  RUN STREAMLIT APP (Interactive Inference)
        Command: streamlit run streamlit_app.py
        Browser: http://localhost:8501
    
    2️⃣  MAKE PREDICTIONS IN PYTHON
        Command: python -c "
        from src.prediction import NLPPredictor
        predictor = NLPPredictor()
        result = predictor.predict_single('This is amazing!')
        print(result)
        "
    
    3️⃣  DEPLOY TO AWS (See AWS_DEPLOYMENT_GUIDE.md)
        Command: python ci_cd/aws_deploy.py --action create
    
    4️⃣  CHECK MODEL METRICS
        File: models/classifier_metrics.json
        Visualization: confusion_matrix.png
    
    📁 IMPORTANT FILES:
    
    - streamlit_app.py         → Web UI for predictions
    - src/model_training.py    → Model training logic
    - src/data_preprocessing.py → Text preprocessing
    - src/prediction.py        → Make predictions
    - AWS_DEPLOYMENT_GUIDE.md  → Deploy to AWS (step-by-step)
    - README.md                → Full documentation
    
    🚀 QUICK COMMANDS:
    
    # Make single prediction
    python -c "
    from src.prediction import single_predict
    result = single_predict('Great product!')
    print(result)
    "
    
    # Batch predictions from file
    python -c "
    from src.prediction import NLPPredictor
    predictor = NLPPredictor()
    results = predictor.predict_batch(['Text 1', 'Text 2'])
    print(results)
    "
    
    # Re-train with new data
    python create_sample_data.py
    python train.py
    
    💡 TIPS:
    
    - Edit src/config.py to customize model parameters
    - Update create_sample_data.py with your own data
    - Use AWS_DEPLOYMENT_GUIDE.md for cloud deployment
    - Check README.md for advanced usage
    
    ❓ NEED HELP?
    
    - Local issues? Run: python validate_data.py
    - AWS issues? See: AWS_DEPLOYMENT_GUIDE.md
    - API reference? See: README.md
    
    ═══════════════════════════════════════════════════════════
    """)


def main():
    """Main quick start function"""
    print_banner()
    
    # Step 1: Check prerequisites
    print("\n1️⃣  CHECKING PREREQUISITES")
    print("="*60)
    if not check_prerequisites():
        logger.error("Some prerequisites are missing")
        sys.exit(1)
    
    # Step 2: Install dependencies
    print("\n2️⃣  INSTALLING DEPENDENCIES")
    print("="*60)
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        sys.exit(1)
    
    # Step 3: Download NLTK data
    print("\n3️⃣  DOWNLOADING NLP DATA")
    print("="*60)
    if not download_nltk_data():
        logger.warning("Some NLTK data failed to download (may continue anyway)")
    
    # Step 4: Create sample data
    print("\n4️⃣  CREATING SAMPLE DATA")
    print("="*60)
    if not create_sample_data():
        logger.error("Failed to create sample data")
        sys.exit(1)
    
    # Step 5: Train model
    print("\n5️⃣  TRAINING MODEL")
    print("="*60)
    if not train_model():
        logger.error("Failed to train model")
        sys.exit(1)
    
    # Step 6: Validate setup
    print("\n6️⃣  VALIDATING SETUP")
    print("="*60)
    if not validate_setup():
        logger.warning("Some validation checks failed (setup may still work)")
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
