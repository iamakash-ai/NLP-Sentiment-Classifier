"""
Generate sample NLP dataset for training
Creates a sentiment classification dataset with positive and negative examples
"""
import os
import sys
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample data - sentiment classification dataset
SAMPLE_DATA = [
    # Positive examples
    ("This product is absolutely amazing! I love it.", "positive"),
    ("Great quality and fast shipping. Highly recommend!", "positive"),
    ("Best purchase I've made in years. Very satisfied.", "positive"),
    ("Excellent service and wonderful experience overall.", "positive"),
    ("I'm so happy with this. It exceeded my expectations!", "positive"),
    ("Outstanding quality and great value for money.", "positive"),
    ("Perfect! Exactly what I was looking for.", "positive"),
    ("Fantastic product. Will definitely buy again.", "positive"),
    ("Very impressed with the quality and customer service.", "positive"),
    ("This is the best thing ever! Absolutely love it!", "positive"),
    ("Incredible experience from start to finish.", "positive"),
    ("Superior quality, fantastic value.", "positive"),
    ("I'm delighted with my purchase!", "positive"),
    ("Wonderful experience, would highly recommend.", "positive"),
    ("Amazing quality and excellent service.", "positive"),
    ("This exceeded all my expectations!", "positive"),
    ("Absolutely thrilled with this product.", "positive"),
    ("Best decision ever! Very happy.", "positive"),
    ("Outstanding! Couldn't ask for better.", "positive"),
    ("Perfect quality and great service.", "positive"),
    
    # Negative examples
    ("This is terrible. Waste of money.", "negative"),
    ("Really disappointed with this product.", "negative"),
    ("Poor quality and terrible customer service.", "negative"),
    ("Absolutely horrible experience. Don't buy!", "negative"),
    ("This is the worst product I've ever bought.", "negative"),
    ("Broken after one week. Very upset.", "negative"),
    ("Not worth the price at all.", "negative"),
    ("Extremely dissatisfied with my purchase.", "negative"),
    ("Awful quality and slow delivery.", "negative"),
    ("I hate this so much. Complete waste.", "negative"),
    ("Terrible experience overall. Avoid!", "negative"),
    ("Not as described. Very disappointed.", "negative"),
    ("Worst purchase ever made.", "negative"),
    ("Completely unsatisfied. Bad investment.", "negative"),
    ("Poor quality, won't buy again.", "negative"),
    ("Horrible product, terrible service.", "negative"),
    ("This is garbage. Don't waste your money.", "negative"),
    ("So disappointed. Major letdown.", "negative"),
    ("Awful experience from beginning to end.", "negative"),
    ("Completely defective. Very frustrated.", "negative"),
]

def create_sample_dataset(output_file, num_samples=None):
    """
    Create sample CSV dataset for training
    
    Args:
        output_file: Path to output CSV file
        num_samples: Number of samples to generate (None for all)
    """
    logger.info(f"Creating sample dataset...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Use all samples or subset
    data = SAMPLE_DATA
    if num_samples and num_samples < len(SAMPLE_DATA):
        data = SAMPLE_DATA[:num_samples]
    
    # Write to CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'label'])  # Header
            writer.writerows(data)
        
        logger.info(f"Sample dataset created: {output_file}")
        logger.info(f"Total samples: {len(data)}")
        logger.info(f"Positive samples: {sum(1 for _, label in data if label == 'positive')}")
        logger.info(f"Negative samples: {sum(1 for _, label in data if label == 'negative')}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return False


def main():
    """Main function"""
    # Get path for sample data
    base_dir = Path(__file__).parent
    data_file = base_dir / "data" / "raw" / "training_data.csv"
    
    # Create dataset
    success = create_sample_dataset(str(data_file))
    
    if success:
        logger.info("\n✓ Sample data created successfully!")
        logger.info(f"You can now run: python train.py")
    else:
        logger.error("✗ Failed to create sample data")
        sys.exit(1)


if __name__ == "__main__":
    main()
