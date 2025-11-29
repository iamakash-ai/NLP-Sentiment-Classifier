"""
Streamlit app for NLP model inference
Interactive UI for making predictions with the trained model
"""
import sys
import os
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prediction import NLPPredictor
from config import CLASSIFIER_MODEL_PATH, VECTORIZER_MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NLP Sentiment Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .error {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load model once and cache it"""
    try:
        predictor = NLPPredictor(CLASSIFIER_MODEL_PATH, VECTORIZER_MODEL_PATH)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please train the model first by running `python src/model_training.py`")
        return None


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("<h1 class='main-header'>🤖 NLP Sentiment Classifier</h1>", unsafe_allow_html=True)
    
    # Load model
    predictor = load_model()
    
    if predictor is None:
        st.error("Model not loaded. Please ensure the model files exist.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Single Prediction", "Batch Prediction", "File Upload", "About"]
    )
    
    if page == "Single Prediction":
        single_prediction_page(predictor)
    
    elif page == "Batch Prediction":
        batch_prediction_page(predictor)
    
    elif page == "File Upload":
        file_upload_page(predictor)
    
    elif page == "About":
        about_page()


def single_prediction_page(predictor):
    """Single text prediction page"""
    st.header("📝 Single Text Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input text area
        user_text = st.text_area(
            "Enter text for prediction:",
            height=150,
            placeholder="Type or paste your text here...",
            key="single_text"
        )
    
    with col2:
        st.info("""
        ### How to use:
        1. Enter your text
        2. Click "Predict"
        3. View results
        """)
    
    if st.button("🔮 Predict", key="single_predict_btn"):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Making prediction..."):
                result = predictor.predict_with_scores(user_text)
                
                if result['success']:
                    st.success("Prediction completed!")
                    
                    # Display prediction
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Predicted Label",
                            result['prediction'],
                            delta=None
                        )
                    
                    with col2:
                        confidence_pct = result['predicted_probability'] * 100
                        st.metric(
                            "Confidence",
                            f"{confidence_pct:.2f}%",
                            delta=None
                        )
                    
                    with col3:
                        # Color code based on confidence
                        if confidence_pct >= 80:
                            st.metric("Trust Level", "🟢 High", delta=None)
                        elif confidence_pct >= 60:
                            st.metric("Trust Level", "🟡 Medium", delta=None)
                        else:
                            st.metric("Trust Level", "🔴 Low", delta=None)
                    
                    # Display confidence scores for all classes
                    st.subheader("Confidence Scores by Class")
                    
                    scores_df = pd.DataFrame([
                        {
                            'Class': label,
                            'Confidence': score,
                            'Percentage': f"{score*100:.2f}%"
                        }
                        for label, score in result['confidence_scores'].items()
                    ])
                    
                    # Bar chart
                    fig = px.bar(
                        scores_df,
                        x='Class',
                        y='Confidence',
                        title='Confidence Scores by Class',
                        labels={'Confidence': 'Score', 'Class': 'Predicted Class'},
                        color='Confidence',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display table
                    st.table(scores_df)
                
                else:
                    st.error(f"Error in prediction: {result.get('error', 'Unknown error')}")


def batch_prediction_page(predictor):
    """Batch text prediction page"""
    st.header("📚 Batch Text Prediction")
    
    st.info("Enter multiple texts, one per line, to get predictions for all of them.")
    
    # Text area for multiple texts
    batch_text = st.text_area(
        "Enter multiple texts (one per line):",
        height=200,
        placeholder="Text 1\nText 2\nText 3\n...",
        key="batch_text"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔮 Predict Batch", key="batch_predict_btn"):
            if not batch_text.strip():
                st.warning("Please enter some texts first.")
            else:
                texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
                
                with st.spinner(f"Making predictions for {len(texts)} texts..."):
                    results = predictor.predict_batch(texts)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame([
                        {
                            'Text': r['text'][:50] + '...' if len(r['text']) > 50 else r['text'],
                            'Prediction': r['prediction'] if r['success'] else 'Error',
                            'Confidence': f"{r['confidence']*100:.2f}%" if r['success'] else 'N/A',
                            'Status': '✓' if r['success'] else '✗'
                        }
                        for r in results
                    ])
                    
                    st.success(f"Predictions completed for {len(results)} texts!")
                    st.table(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
    
    with col2:
        st.info("""
        ### Tips:
        - One text per line
        - Empty lines are skipped
        - Results can be downloaded
        """)


def file_upload_page(predictor):
    """File upload prediction page"""
    st.header("📂 File Upload Prediction")
    
    st.info("Upload a CSV file with a 'text' column to get predictions for all rows.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="file_upload"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("CSV must contain a 'text' column")
                st.info(f"Available columns: {', '.join(df.columns.tolist())}")
            else:
                st.write(f"Loaded {len(df)} rows")
                st.dataframe(df.head())
                
                if st.button("🔮 Predict All Rows", key="file_predict_btn"):
                    with st.spinner(f"Making predictions for {len(df)} rows..."):
                        # Make predictions
                        results_df = predictor.predict_dataframe(df)
                        
                        st.success(f"Predictions completed!")
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="predictions_with_results.csv",
                            mime="text/csv"
                        )
                        
                        # Statistics
                        st.subheader("Prediction Statistics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Total Predictions",
                                len(results_df),
                                delta=None
                            )
                        
                        with col2:
                            avg_confidence = results_df['confidence'].mean()
                            st.metric(
                                "Avg Confidence",
                                f"{avg_confidence*100:.2f}%",
                                delta=None
                            )
                        
                        with col3:
                            success_count = results_df['prediction'].notna().sum()
                            st.metric(
                                "Successful Predictions",
                                success_count,
                                delta=None
                            )
                        
                        # Prediction distribution
                        pred_dist = results_df['prediction'].value_counts()
                        fig = px.pie(
                            values=pred_dist.values,
                            names=pred_dist.index,
                            title='Distribution of Predictions'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")


def about_page():
    """About page"""
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## NLP Sentiment Classification System
    
    This application provides predictions using a machine learning model trained on NLP data.
    
    ### Features:
    - **Single Prediction**: Analyze individual texts
    - **Batch Prediction**: Process multiple texts at once
    - **File Upload**: Process CSV files with bulk predictions
    - **Confidence Scores**: View prediction confidence for all classes
    
    ### Model Information:
    - **Algorithm**: Multiple classifier models (Logistic Regression, Naive Bayes, SVM)
    - **Features**: TF-IDF vectorization with text preprocessing
    - **Text Processing**: Lemmatization, stopword removal, special character handling
    
    ### How it works:
    1. Text input is cleaned and normalized
    2. TF-IDF features are extracted
    3. Trained classifier makes prediction
    4. Confidence scores are calculated
    
    ### Data Pipeline:
    - Raw data → Preprocessing → Feature extraction → Model training → Evaluation → Deployment
    
    ### Technologies:
    - **Backend**: Python, scikit-learn
    - **Frontend**: Streamlit
    - **ML Libraries**: pandas, numpy, scikit-learn
    
    ### Model Performance:
    Check the model metrics file for detailed evaluation results.
    
    ---
    
    **Version**: 1.0.0  
    **Last Updated**: November 2024
    """)
    
    st.markdown("---")
    
    st.subheader("Quick Start Guide")
    
    with st.expander("📖 How to use the app"):
        st.markdown("""
        1. **Single Prediction**:
           - Enter text in the text area
           - Click "Predict" button
           - View confidence scores
        
        2. **Batch Prediction**:
           - Enter multiple texts (one per line)
           - Click "Predict Batch"
           - Download results as CSV
        
        3. **File Upload**:
           - Prepare CSV with 'text' column
           - Upload file
           - View predictions and statistics
        """)


if __name__ == "__main__":
    main()
