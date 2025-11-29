# 📂 File Directory & Description

Complete guide to every file in the NLP ML project and what it does.

---

## 📍 START HERE

| File | Purpose | When to Use |
|------|---------|------------|
| **GETTING_STARTED.md** | Overview of the entire project | Read first! |
| **QUICKSTART_BEGINNER.md** | Step-by-step local setup guide | Beginners starting locally |
| **AWS_DEPLOYMENT_GUIDE.md** | Step-by-step AWS deployment | Deploying to cloud |
| **AWS_DEPLOYMENT_CHECKLIST.md** | Checklist for AWS deployment | Tracking AWS progress |

---

## 🚀 MAIN SCRIPTS (Run These)

### For Local Development

| File | Purpose | How to Run | Time |
|------|---------|-----------|------|
| **quickstart.py** | Automated setup for beginners | `python quickstart.py` | 5 min |
| **create_sample_data.py** | Generate sample training data | `python create_sample_data.py` | 1 min |
| **train.py** | Train the ML model | `python train.py` | 3-5 min |
| **streamlit_app.py** | Web interface for predictions | `streamlit run streamlit_app.py` | Instant |
| **example_usage.py** | Show code examples | `python example_usage.py` | 2 min |
| **validate_data.py** | Check if setup is correct | `python validate_data.py` | 1 min |

### For AWS Deployment

| File | Purpose | How to Run |
|------|---------|-----------|
| **ci_cd/aws_deploy.py** | Deploy infrastructure to AWS | `python ci_cd/aws_deploy.py --action create` |

---

## 📚 DOCUMENTATION FILES

| File | Purpose | Read When |
|------|---------|-----------|
| **README.md** | Full project documentation | Want detailed info |
| **GETTING_STARTED.md** | Project overview & paths | Starting the project |
| **QUICKSTART_BEGINNER.md** | Beginner step-by-step guide | New to the project |
| **AWS_DEPLOYMENT_GUIDE.md** | AWS setup guide for beginners | Deploying to cloud |
| **AWS_DEPLOYMENT_CHECKLIST.md** | AWS deployment checklist | Tracking AWS progress |
| **requirements.txt** | Python dependencies | Installing packages |
| **.gitignore** | Git ignore patterns | Git configuration |

---

## 🧠 SOURCE CODE (ML Logic)

### src/ Folder - Core ML Pipeline

#### src/config.py
**Purpose:** Configuration and settings
**What it does:**
- Defines file paths
- ML model parameters
- AWS settings
- Hyperparameter ranges

**When to edit:**
- Change model parameters
- Modify data paths
- Adjust hyperparameters

**Key variables:**
```python
MAX_FEATURES = 5000        # Max features for vectorizer
MIN_DF = 2                 # Minimum document frequency
RANDOM_STATE = 42          # For reproducibility
MODELS_TO_TEST = [...]     # Which models to train
```

---

#### src/data_preprocessing.py
**Purpose:** Text cleaning and feature extraction
**What it does:**
- Cleans text (removes URLs, special chars)
- Tokenizes and lemmatizes
- Removes stopwords
- Creates TF-IDF features

**Key functions:**
```python
TextCleaner()              # Custom text cleaner
create_preprocessing_pipeline()  # Creates sklearn pipeline
preprocess_data()          # Applies preprocessing
prepare_training_data()    # End-to-end data prep
```

**When to edit:**
- Change text preprocessing rules
- Modify vectorization parameters
- Add custom text cleaning

---

#### src/model_training.py
**Purpose:** Model training and evaluation
**What it does:**
- Trains multiple classifiers
- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Model comparison and selection
- Saves best model

**Key classes:**
```python
ModelTrainer()             # Handles training
```

**Key methods:**
```python
train_with_hyperparameter_tuning()  # Train with tuning
evaluate_model()           # Evaluate on test set
compare_models()           # Compare all models
save_model()               # Save best model
```

**When to edit:**
- Add new classifiers
- Change hyperparameter ranges
- Modify evaluation metrics
- Add new models

---

#### src/prediction.py
**Purpose:** Make predictions on new data
**What it does:**
- Loads trained models
- Preprocesses input text
- Makes predictions
- Returns confidence scores

**Key classes:**
```python
NLPPredictor()             # Main prediction class
```

**Key methods:**
```python
predict_single()           # Single text prediction
predict_batch()            # Multiple texts
predict_with_scores()      # Detailed scores
predict_dataframe()        # CSV predictions
```

**When to use:**
- Making predictions in code
- Building applications
- Integration with other systems

---

## 🌐 WEB INTERFACE

#### streamlit_app.py
**Purpose:** Interactive web UI for predictions
**What it does:**
- Single text prediction
- Batch text prediction
- CSV file upload
- Confidence score visualization
- Results download

**How to run:**
```bash
streamlit run streamlit_app.py
```

**Features:**
- Real-time predictions
- Plotly visualizations
- File upload & download
- Model metrics display

**When to use:**
- Making interactive predictions
- Sharing with non-technical users
- Demonstrating the model
- Testing predictions

---

## ☁️ AWS CI/CD PIPELINE

### ci_cd/ Folder - Automation & Deployment

#### ci_cd/aws_deploy.py
**Purpose:** Deploy infrastructure to AWS
**What it does:**
- Creates CloudFormation stacks
- Manages AWS resources
- Handles deployment lifecycle

**How to run:**
```bash
python ci_cd/aws_deploy.py --action create --stack-name nlp-ml-stack
```

**Commands:**
```bash
--action create      # Create new stack
--action update      # Update existing stack
--action delete      # Delete stack
```

**When to use:**
- Deploying to AWS
- Creating infrastructure
- Managing AWS resources

---

#### ci_cd/buildspec.yml
**Purpose:** AWS CodeBuild configuration
**What it does:**
- Defines build process
- Installs dependencies
- Runs model training
- Uploads artifacts to S3

**Key phases:**
- **install:** Install dependencies
- **pre_build:** Check data
- **build:** Train model
- **post_build:** Upload results

**When to edit:**
- Change build process
- Add build steps
- Modify artifact upload

---

#### ci_cd/cloudformation_template.py
**Purpose:** Infrastructure as Code (IaC)
**What it does:**
- Defines AWS resources (S3, IAM, CodeBuild)
- Creates complete infrastructure
- Enables reproducible deployments

**Resources created:**
- S3 buckets (artifacts & data)
- IAM roles (permissions)
- CodeBuild project
- CloudWatch logs

**When to edit:**
- Add new AWS resources
- Change security settings
- Modify infrastructure

---

## 📊 DATA FOLDERS

#### data/ Folder
```
data/
├── raw/
│   └── training_data.csv        ← Your training data
└── processed/
    └── (Created during training)
```

**raw/:** Place raw data here
- CSV format: `text`, `label` columns
- Example: `training_data.csv`

**processed/:** Auto-generated processed features
- Created after running `train.py`

---

#### models/ Folder
```
models/
├── classifier.pkl              ← Trained classifier (created by train.py)
├── vectorizer.pkl              ← TF-IDF vectorizer (created by train.py)
└── classifier_metrics.json     ← Performance metrics (created by train.py)
```

**All files auto-generated after training!**

---

## 📋 PROJECT CONFIGURATION FILES

| File | Purpose |
|------|---------|
| **requirements.txt** | Python package dependencies |
| **.gitignore** | Git ignore patterns |
| **src/__init__.py** | Python package marker |

---

## 🔄 FILE RELATIONSHIPS

```
┌─ GETTING_STARTED.md (Read First!)
│
├─ Local Path:
│  ├─ QUICKSTART_BEGINNER.md
│  ├─ create_sample_data.py → data/raw/training_data.csv
│  ├─ train.py → models/classifier.pkl + vectorizer.pkl
│  ├─ streamlit_app.py (uses models/)
│  └─ example_usage.py (demonstrates prediction)
│
└─ AWS Path:
   ├─ AWS_DEPLOYMENT_GUIDE.md
   ├─ AWS_DEPLOYMENT_CHECKLIST.md
   ├─ ci_cd/aws_deploy.py (uses cloudformation_template.py)
   ├─ ci_cd/buildspec.yml (runs train.py)
   └─ ci_cd/cloudformation_template.py (defines resources)
```

---

## 📈 Data Flow

```
Raw Text
    ↓
data_preprocessing.py
    ├─ TextCleaner (cleans text)
    └─ TfidfVectorizer (creates features)
    ↓
X_train (features), y_train (labels)
    ↓
model_training.py
    ├─ ModelTrainer
    ├─ GridSearchCV (tuning)
    ├─ Cross-validation
    └─ Model evaluation
    ↓
Best model → models/classifier.pkl
    ↓
prediction.py
    ├─ Load model
    ├─ Preprocess input
    ├─ Make prediction
    └─ Return result
    ↓
streamlit_app.py or API
```

---

## 🔑 Key Files by Task

### I want to...

| Task | File(s) to Use |
|------|----------------|
| **Setup locally** | QUICKSTART_BEGINNER.md → quickstart.py |
| **Deploy to AWS** | AWS_DEPLOYMENT_GUIDE.md → ci_cd/aws_deploy.py |
| **Make predictions** | src/prediction.py or streamlit_app.py |
| **Train model** | train.py (uses src/data_preprocessing.py + src/model_training.py) |
| **Change parameters** | src/config.py |
| **Add custom preprocessing** | src/data_preprocessing.py |
| **Add new models** | src/model_training.py |
| **Use web interface** | streamlit_app.py |
| **See code examples** | example_usage.py |
| **Check setup** | validate_data.py |

---

## 📦 Dependency Graph

```
requirements.txt
    ↓
    ├─→ numpy, pandas, scipy
    ├─→ scikit-learn (sklearn)
    ├─→ nltk
    ├─→ joblib
    ├─→ boto3 (AWS)
    ├─→ streamlit (web UI)
    ├─→ plotly (visualization)
    └─→ others...
```

---

## ⚙️ Configuration Hierarchy

```
DEFAULTS (in config.py)
    ↓
ENVIRONMENT (AWS env vars)
    ↓
COMMAND LINE (argparse)
    ↓
CODE (values used)
```

---

## 📝 File Naming Convention

| Pattern | Meaning |
|---------|---------|
| `*.py` | Python source code |
| `*.md` | Markdown documentation |
| `*.txt` | Text files (requirements) |
| `*.pkl` | Pickle (serialized Python objects) |
| `*.json` | JSON (metrics, config) |
| `*.csv` | Data files |
| `*.yml` | YAML configuration |

---

## 🗂️ Complete File Tree

```
nlp_project/
│
├── 📘 GETTING_STARTED.md                  ← Read this first!
├── 📘 QUICKSTART_BEGINNER.md              ← Local setup
├── 📘 AWS_DEPLOYMENT_GUIDE.md             ← AWS setup
├── 📘 AWS_DEPLOYMENT_CHECKLIST.md         ← AWS checklist
├── 📘 README.md                           ← Full docs
│
├── 🚀 quickstart.py                       ← Auto-setup
├── 🚀 train.py                            ← Train model
├── 🚀 streamlit_app.py                    ← Web UI
├── 🚀 create_sample_data.py               ← Generate data
├── 🚀 example_usage.py                    ← Examples
├── 🚀 validate_data.py                    ← Validation
│
├── requirements.txt                       ← Dependencies
├── .gitignore                             ← Git config
│
├── src/                                   ← Source code
│   ├── __init__.py
│   ├── config.py                          ← Settings
│   ├── data_preprocessing.py              ← Data pipeline
│   ├── model_training.py                  ← ML training
│   └── prediction.py                      ← Predictions
│
├── ci_cd/                                 ← AWS automation
│   ├── aws_deploy.py                      ← Deploy script
│   ├── buildspec.yml                      ← Build config
│   └── cloudformation_template.py         ← Infrastructure
│
├── data/                                  ← Data files
│   ├── raw/
│   │   └── training_data.csv              ← Input here
│   └── processed/
│
├── models/                                ← Model files
│   ├── classifier.pkl                     ← (auto-created)
│   ├── vectorizer.pkl                     ← (auto-created)
│   └── classifier_metrics.json            ← (auto-created)
│
└── tests/                                 ← Tests (optional)
```

---

## 🎯 Quick Reference

**To start locally:**
1. Read `QUICKSTART_BEGINNER.md`
2. Run `python quickstart.py` (automatic setup)

**To deploy to AWS:**
1. Read `AWS_DEPLOYMENT_GUIDE.md`
2. Run `python ci_cd/aws_deploy.py --action create`

**To understand the code:**
1. Read `src/config.py` (settings)
2. Read `src/data_preprocessing.py` (data pipeline)
3. Read `src/model_training.py` (ML logic)
4. Read `src/prediction.py` (predictions)

**To see examples:**
1. Run `python example_usage.py`
2. Read code comments in source files

---

## 📞 File Lookup Table

| I need... | Find it here |
|-----------|------------|
| Setup instructions | QUICKSTART_BEGINNER.md |
| AWS instructions | AWS_DEPLOYMENT_GUIDE.md |
| Code examples | example_usage.py |
| Configuration | src/config.py |
| Data preparation | src/data_preprocessing.py |
| Model training | src/model_training.py |
| Predictions | src/prediction.py |
| Web interface | streamlit_app.py |
| Full documentation | README.md |

---

**Status:** ✅ All files documented  
**Last Updated:** November 2024
