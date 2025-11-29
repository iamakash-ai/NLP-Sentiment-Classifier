# 🚀 BEGINNER'S GUIDE - Step-by-Step Local Setup

Follow these steps exactly in order. This guide is for **Windows, Mac, or Linux**.

---

## STEP 1: Prerequisites (5 minutes)

### 1.1 Check Python Installation
Open terminal/PowerShell and type:
```bash
python --version
```

**Expected output:** `Python 3.8` or higher

❌ If you don't have Python:
- Download from https://www.python.org/downloads/
- On Windows: Check "Add Python to PATH" during installation
- Verify: Restart terminal and run `python --version` again

### 1.2 Check Git Installation (Optional but Recommended)
```bash
git --version
```

❌ If you don't have Git:
- Download from https://git-scm.com/download
- Install with default settings

---

## STEP 2: Set Up Project Folder (2 minutes)

### 2.1 Navigate to Project
```bash
# On Windows (PowerShell)
cd c:\Users\YourUsername\Desktop\NLP\nlp_project

# On Mac/Linux
cd ~/Desktop/NLP/nlp_project
```

### 2.2 Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it - WINDOWS
venv\Scripts\activate

# Activate it - MAC/LINUX
source venv/bin/activate
```

✓ **You should see `(venv)` in your terminal prompt**

---

## STEP 3: Install Dependencies (3-5 minutes)

### 3.1 Install Python Packages
```bash
pip install -r requirements.txt
```

⏳ This may take 2-5 minutes (lots of packages to download)

### 3.2 Download NLP Data (1 minute)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

✓ You should see: `[nltk_data] Downloading package...`

---

## STEP 4: Create Sample Training Data (1 minute)

### 4.1 Run the Data Generator
```bash
python create_sample_data.py
```

✓ **Expected output:**
```
Creating sample dataset...
Sample dataset created: data/raw/training_data.csv
Total samples: 40
Positive samples: 20
Negative samples: 20
```

### 4.2 Verify Data File
```bash
# Windows
dir data\raw\

# Mac/Linux
ls -la data/raw/
```

✓ You should see: `training_data.csv` file

---

## STEP 5: Train the Model (3-5 minutes)

### 5.1 Run Training
```bash
python train.py
```

⏳ This will:
- Preprocess data ✓
- Train multiple models ✓
- Tune hyperparameters ✓
- Save best model ✓

⏳ **This takes 3-5 minutes - be patient!**

✓ **Expected output at the end:**
```
==================================================
✓ Training Pipeline Completed Successfully!
==================================================
Models saved to: models/classifier.pkl
Metrics saved to: models/classifier_metrics.json
```

### 5.2 Verify Model Files
```bash
# Windows
dir models\

# Mac/Linux
ls -la models/
```

✓ You should see:
- `classifier.pkl` (the trained model)
- `vectorizer.pkl` (text processor)
- `classifier_metrics.json` (performance scores)

---

## STEP 6: Make Your First Prediction (1 minute)

### 6.1 Quick Test
```bash
python -c "
from src.prediction import single_predict
result = single_predict('This is amazing!')
print('Prediction:', result['prediction'])
print('Confidence:', f\"{result['confidence']:.0%}\")
"
```

✓ You should see output like:
```
Prediction: positive
Confidence: 85%
```

---

## STEP 7: Run Streamlit Web App (2 minutes)

### 7.1 Start the App
```bash
streamlit run streamlit_app.py
```

✓ You should see:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

### 7.2 Open in Browser
Click the link or go to: **http://localhost:8501**

🎉 **You should see the web app with three tabs:**
- Single Prediction
- Batch Prediction
- File Upload

### 7.3 Test the App
1. Go to **Single Prediction** tab
2. Type: "This is fantastic!"
3. Click **Predict** button
4. You should see the prediction result!

### 7.4 Stop the App
Press `Ctrl + C` in terminal

---

## STEP 8: Run Example Usage (2 minutes)

### 8.1 See All Examples
```bash
python example_usage.py
```

This will show you:
- ✓ Single text prediction
- ✓ Batch predictions
- ✓ Predictions from CSV file
- ✓ Model information
- ✓ And more!

---

## STEP 9: Verify Everything Works (1 minute)

### 9.1 Run Validation
```bash
python validate_data.py
```

✓ You should see green checkmarks ✓ for:
- ✓ Classifier model loaded
- ✓ Vectorizer loaded
- ✓ Data file valid
- ✓ All thresholds met

---

## QUICK REFERENCE - Common Commands

### 🎯 Make Predictions

**Single text:**
```bash
python -c "from src.prediction import single_predict; print(single_predict('Great!'))"
```

**Multiple texts:**
```bash
python -c "
from src.prediction import batch_predict
texts = ['Good product', 'Bad quality', 'Excellent!']
results = batch_predict(texts)
for r in results:
    print(f\"{r['text']:20} → {r['prediction']:10} ({r['confidence']:.0%})\")
"
```

**From file:**
```bash
python -c "
from src.prediction import NLPPredictor
import pandas as pd
predictor = NLPPredictor()
df = pd.read_csv('your_file.csv')
results = predictor.predict_dataframe(df)
results.to_csv('predictions.csv')
print('Saved to predictions.csv')
"
```

### 📊 Train Model

**With sample data:**
```bash
python create_sample_data.py
python train.py
```

**With your own CSV:**
1. Create CSV with 'text' and 'label' columns
2. Save to `data/raw/your_data.csv`
3. Edit `config.py` to point to your file
4. Run: `python train.py`

### 🔧 Utilities

**Check data quality:**
```bash
python validate_data.py
```

**View examples:**
```bash
python example_usage.py
```

**Check model metrics:**
```bash
python -c "
import json
with open('models/classifier_metrics.json') as f:
    metrics = json.load(f)
    for k, v in metrics.items():
        print(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}')
"
```

### 🌐 Run Web App
```bash
streamlit run streamlit_app.py
```
Then open browser to: **http://localhost:8501**

---

## 📁 Project Structure Overview

```
nlp_project/
├── streamlit_app.py          ← Web interface (run with streamlit)
├── train.py                  ← Train model (run first)
├── create_sample_data.py     ← Generate sample data
├── example_usage.py          ← Show examples
├── validate_data.py          ← Check setup
│
├── data/
│   └── raw/
│       └── training_data.csv ← Your training data goes here
│
├── models/
│   ├── classifier.pkl        ← Trained model (created after train.py)
│   └── vectorizer.pkl        ← Text processor (created after train.py)
│
├── src/
│   ├── config.py             ← Settings (edit here to customize)
│   ├── data_preprocessing.py ← Text cleaning
│   ├── model_training.py     ← Model training logic
│   └── prediction.py         ← Make predictions
│
└── README.md                 ← Full documentation
```

---

## 🚨 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'sklearn'"
**Solution:**
```bash
pip install -r requirements.txt
```

### Error: "No such file or directory: 'data/raw/training_data.csv'"
**Solution:**
```bash
python create_sample_data.py
```

### Error: "No such file: models/classifier.pkl"
**Solution:**
```bash
python train.py
```
Wait 5 minutes for training to complete.

### Error: "Port 8501 is already in use"
**Solution:**
```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502
```
Then open: http://localhost:8502

### Error on Windows: "venv\Scripts\activate is not recognized"
**Solution:**
```bash
# Try PowerShell approach
venv\Scripts\Activate.ps1
```

### NLTK Data Download Issues
**Solution:**
```bash
python -m nltk.downloader punkt stopwords wordnet
```

---

## ✅ Checklist - Did You Complete Everything?

- [ ] Python installed (version 3.8+)
- [ ] Virtual environment created and activated (see `(venv)` in prompt)
- [ ] Dependencies installed (no errors from pip install)
- [ ] NLTK data downloaded
- [ ] Sample data created (`data/raw/training_data.csv` exists)
- [ ] Model trained successfully
- [ ] First prediction works
- [ ] Streamlit web app runs
- [ ] Example usage scripts work

**If all checked:** 🎉 **You're ready for AWS deployment!**

---

## 🎓 Learning Path

1. **Beginner:** Run quickstart.py → Make predictions via web app
2. **Intermediate:** Edit config.py → Train with own data
3. **Advanced:** Modify model_training.py → Add custom classifiers
4. **Expert:** Deploy on AWS using AWS_DEPLOYMENT_GUIDE.md

---

## 📚 Next Steps

### Option 1: Deploy to AWS Cloud ☁️
See: **AWS_DEPLOYMENT_GUIDE.md**

### Option 2: Use Your Own Data 📊
1. Create CSV with 'text' and 'label' columns
2. Place in `data/raw/`
3. Edit `src/config.py` to point to your file
4. Run: `python train.py`

### Option 3: Customize the Model 🔧
1. Edit `src/config.py` to change parameters
2. Edit `src/model_training.py` to add models
3. Run: `python train.py`

### Option 4: Share the Web App 🌐
1. Deploy to Streamlit Cloud (free): https://streamlit.io/cloud
2. Or deploy to AWS (see AWS_DEPLOYMENT_GUIDE.md)

---

## 🆘 Need Help?

1. **Check README.md** for detailed documentation
2. **Run validate_data.py** to check your setup
3. **Check AWS_DEPLOYMENT_GUIDE.md** for cloud deployment
4. **See example_usage.py** for code examples
5. **Check scikit-learn docs**: https://scikit-learn.org

---

## 🎉 Congratulations!

You now have a working NLP ML project with:
- ✓ Data preprocessing pipeline
- ✓ Trained classifier
- ✓ Web interface (Streamlit)
- ✓ Ready for AWS deployment

**Next:** Deploy to AWS Cloud! See **AWS_DEPLOYMENT_GUIDE.md**

---

**Last Updated:** November 2024
**Status:** Ready for Production
