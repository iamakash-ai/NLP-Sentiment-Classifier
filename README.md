# NLP Machine Learning Project with AWS CI/CD Pipeline

A complete end-to-end machine learning project for NLP tasks with scikit-learn pipelines, AWS CI/CD automation, and Streamlit inference interface.

## Features

### ✨ Core ML Features
- **Data Preprocessing Pipeline**: Text cleaning, lemmatization, stopword removal using sklearn Pipeline
- **Feature Extraction**: TF-IDF vectorization with customizable parameters
- **Model Training**: Multiple classifier comparison (Logistic Regression, Naive Bayes, SVM, Random Forest)
- **Hyperparameter Tuning**: GridSearchCV with cross-validation for optimal parameters
- **Model Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
- **Pickle Serialization**: Models and vectorizers saved for production use

### 🚀 CI/CD & Deployment
- **AWS CodeBuild**: Automated model training on code commits
- **AWS CodePipeline**: Orchestrates build, train, and deploy stages
- **CloudFormation**: Infrastructure as Code for AWS resources
- **S3 Integration**: Artifact storage and model versioning
- **Auto-retraining**: Pipeline triggers when data changes

### 🎯 Inference Interface
- **Streamlit App**: Interactive web UI for predictions
- **Single Prediction**: Test individual texts with confidence scores
- **Batch Prediction**: Process multiple texts at once
- **File Upload**: CSV batch processing with results download
- **Real-time Visualization**: Plotly charts for confidence distribution

## Project Structure

```
nlp_project/
├── src/
│   ├── config.py                 # Configuration and paths
│   ├── data_preprocessing.py      # Data pipeline and vectorization
│   ├── model_training.py          # Model training and tuning
│   └── prediction.py              # Inference module
├── data/
│   ├── raw/                       # Raw CSV data files
│   └── processed/                 # Processed features
├── models/                        # Trained models (pickle files)
├── ci_cd/
│   ├── buildspec.yml              # CodeBuild specification
│   ├── cloudformation_template.py # Infrastructure template
│   └── aws_deploy.py              # Deployment script
├── streamlit_app.py               # Streamlit inference UI
├── train.py                       # Main training pipeline
├── create_sample_data.py           # Generate sample dataset
└── requirements.txt               # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda
- AWS Account (for CI/CD deployment)
- Git

### Local Setup

1. **Clone/Create Project**
```bash
cd nlp_project
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK Data**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Quick Start

### 1. Create Sample Data
```bash
python create_sample_data.py
```
This generates a sample sentiment classification dataset.

### 2. Train Model
```bash
python train.py
```
This runs the complete pipeline:
- Data preprocessing with TF-IDF vectorization
- Model comparison with hyperparameter tuning
- Evaluation on test set
- Saves best model to `models/classifier.pkl`

### 3. Run Streamlit App
```bash
streamlit run streamlit_app.py
```
Access the app at `http://localhost:8501`

### 4. Make Predictions
```python
from src.prediction import NLPPredictor

predictor = NLPPredictor()
result = predictor.predict_single("This is amazing!")
print(result)
```

## Data Format

Training data should be a CSV file with two columns:

```csv
text,label
"This is great!",positive
"I hate this.",negative
"Amazing product!",positive
```

Place data in `data/raw/training_data.csv`

## Pipeline Architecture

### Data Preprocessing Pipeline
```
Raw Text
    ↓
Text Cleaning (lowercase, remove URLs, special chars)
    ↓
Tokenization & Lemmatization
    ↓
Stopword Removal
    ↓
TF-IDF Vectorization
    ↓
Feature Matrix
```

### Model Training Pipeline
```
Training Data
    ↓
Data Split (Train/Val/Test)
    ↓
Model Selection
    ↓
Hyperparameter Tuning (GridSearchCV)
    ↓
Cross-Validation
    ↓
Evaluation on Test Set
    ↓
Best Model Selection
    ↓
Model Serialization (Pickle)
```

## Configuration

Edit `src/config.py` to customize:
- Model parameters (MAX_FEATURES, MIN_DF, MAX_DF)
- Hyperparameter ranges
- Cross-validation folds
- AWS S3 bucket names
- Random seed and test split ratios

## AWS CI/CD Deployment

### Prerequisites
- AWS CLI configured with credentials
- AWS Account with appropriate permissions
- GitHub or CodeCommit repository

### Setup AWS Pipeline

1. **Deploy Infrastructure**
```bash
python ci_cd/aws_deploy.py --action create \
    --stack-name nlp-ml-stack \
    --region us-east-1 \
    --environment dev
```

2. **Upload Initial Data**
```bash
aws s3 cp data/raw/training_data.csv \
    s3://<bucket-name>/data/
```

3. **Trigger Pipeline**
When you push to the repository, CodePipeline automatically:
- Checks out code
- Installs dependencies
- Runs training script
- Uploads models to S3
- Generates metrics reports

### CloudFormation Resources

The template creates:
- **S3 Buckets**: Artifact storage and data storage
- **IAM Roles**: CodeBuild and CodePipeline execution roles
- **CodeBuild Project**: Model training automation
- **CloudWatch Logs**: Build logs and metrics
- **EventBridge Rules**: Automatic triggers (optional)

### Update Models

Models are automatically retrained when:
- Code is pushed to main branch
- Data files in S3 are updated
- Scheduled pipeline execution (if configured)

Models are stored in S3 with version control and automatically downloaded for inference.

## Model Evaluation

After training, check:
- `models/classifier_metrics.json` - Model performance metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `training_results.json` - Comparison results from all models

Metrics include:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC (for binary classification)
- Confusion Matrix
- Classification Report

## Streamlit Features

### Pages

1. **Single Prediction**
   - Input: Text string
   - Output: Label + Confidence + Class scores
   - Visualization: Bar chart of confidence scores

2. **Batch Prediction**
   - Input: Multiple texts (one per line)
   - Output: Table with predictions and confidence
   - Download: Results as CSV

3. **File Upload**
   - Input: CSV with 'text' column
   - Output: Predictions added to data
   - Stats: Distribution charts and metrics
   - Download: Results as CSV

4. **About**
   - Project information
   - Model details
   - Quick start guide

## Production Deployment

### Streamlit Cloud
```bash
# Push to GitHub
git push origin main

# Deploy via https://streamlit.io/cloud
```

### AWS SageMaker
The trained models can be deployed to SageMaker for production inference:
```bash
# Create SageMaker endpoint from S3 model
aws sagemaker create-model \
    --model-name nlp-classifier \
    --primary-container Image=<container-uri>,ModelDataUrl=s3://<bucket>/models/
```

### Docker Container
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "streamlit_app.py"]
```

## Monitoring & Logging

- **CloudWatch Logs**: Access build logs via AWS Console
- **Model Metrics**: JSON files with training history
- **Prediction Logs**: Streamlit app logs for debugging

## Troubleshooting

### Models not found
```bash
# Verify model files exist
ls -la models/
python create_sample_data.py
python train.py
```

### AWS deployment fails
```bash
# Check AWS credentials
aws sts get-caller-identity

# Verify S3 bucket access
aws s3 ls s3://<bucket-name>/

# Check CloudFormation stack status
aws cloudformation describe-stacks --stack-name nlp-ml-stack
```

### Streamlit app not loading
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Clear Streamlit cache
streamlit cache clear

# Check Python path
python -c "import sys; print(sys.path)"
```

## Advanced Usage

### Custom Text Preprocessing
Edit `TextCleaner` class in `src/data_preprocessing.py`:
```python
class TextCleaner:
    def clean_text(self, text):
        # Add custom preprocessing here
        return cleaned_text
```

### Add New Classifiers
Update `get_model_options()` in `src/model_training.py`:
```python
def get_model_options(self):
    models = {
        'custom_model': {
            'model': CustomClassifier(),
            'params': {'param1': [v1, v2]}
        }
    }
```

### Modify Pipeline Triggers
Edit `ci_cd/buildspec.yml` and CloudFormation template for custom build steps.

## Performance Optimization

- **Feature Reduction**: Adjust MAX_FEATURES in config.py
- **Model Selection**: Try different classifiers in MODELS_TO_TEST
- **Parallel Processing**: Increase N_JOBS for multi-threading
- **Batch Size**: Optimize for memory constraints

## Security Best Practices

- Use AWS IAM roles for services
- Enable S3 bucket versioning
- Enable CloudTrail for audit logging
- Use environment variables for sensitive data
- Implement rate limiting in Streamlit (if exposed publicly)

## License

MIT License - Feel free to use and modify

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Submit pull request

## Support

For issues and questions:
- Check troubleshooting section
- Review AWS documentation
- Check Streamlit docs: https://docs.streamlit.io
- Check scikit-learn docs: https://scikit-learn.org

## Roadmap

- [ ] Multi-label classification support
- [ ] Custom neural network models
- [ ] Auto-scaling SageMaker endpoints
- [ ] Real-time data monitoring
- [ ] A/B testing framework
- [ ] Model explainability (LIME/SHAP)
- [ ] Advanced hyperparameter optimization (Bayesian)

---

**Created**: November 2024
**Version**: 1.0.0
**Status**: Production Ready
