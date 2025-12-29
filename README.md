# Side Effects Drug Analytics

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)](#)
[![Status](https://img.shields.io/badge/Project-Active-success)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![Contributors](https://img.shields.io/badge/Contributors-1-orange.svg)](#)

## Project Description

**Side Effects Drug Analytics** is a comprehensive machine learning and data science project that analyzes patient-generated drug reviews to extract actionable insights about adverse drug reactions (ADRs). This project combines natural language processing, predictive modeling, and explainable AI to help healthcare professionals, researchers, and patients understand medication side effects and their impact on patient satisfaction.

### Key Value Propositions
- **Real-world Evidence**: Leverages authentic patient experiences from WebMD reviews
- **Predictive Analytics**: Machine learning models predict satisfaction and identify side effect patterns
- **Explainable AI**: Transparent model insights using SHAP, permutation importance, and partial dependence
- **Clinical Decision Support**: Dashboard-style visualizations for healthcare stakeholders
- **Reproducible Research**: End-to-end pipeline with documented methodology and open-source code

Actionable insights on adverse drug reactions (ADRs) from real‑world WebMD reviews, with end‑to‑end analytics, machine learning, deep learning, and explainable AI.

## Quick Links
- Notebook: [Side_Effects_Drug_Analytics.ipynb](file:///c:/Users/Shankar/Desktop/New%20folder%20(3)/Side_Effects_Drug_Analytics.ipynb)
- README: [readme.md](file:///c:/Users/Shankar/Desktop/New%20folder%20(3)/readme.md)
- Dataset: [webmd.csv](file:///c:/Users/Shankar/Desktop/New%20folder%20(3)/webmd.csv)
- Original Dataset: [Kaggle - WebMD Drug Reviews Dataset](https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset)

## Table of Contents
- Overview
- Features
- Executive Summary
- Objectives & Research Questions
- Methodology
- Architecture & Workflow
- Repository Structure
- Dataset
- Data Dictionary
- Background & Motivation
- Stakeholders & Use Cases
- Success Metrics (KPIs)
- Setup
- Quickstart
- Usage
- Notebook Highlights
- Visual Catalogue
- Explainability
- Reproducibility
- Troubleshooting
- Environment
- Data Provenance & Quality
- Preprocessing Pipeline
- NLP Pipeline
- ML Modeling Details
- Dashboard Overview
- Performance & Compute
- Security, Privacy & Ethics
- Limitations & Bias
- Roadmap
- Contributing
- Citation
- License
- Author

## Overview
This project analyzes patient‑generated drug reviews to understand side effects and their relationships to satisfaction, effectiveness, ease of use, and demographics. It includes a comprehensive Jupyter Notebook that walks through 40 sections: data mining, EDA, NLP, ML/DL models, explainability, and dashboard‑style summaries.

- Main notebook: [Side_Effects_Drug_Analytics.ipynb](file:///c:/Users/Shankar/Desktop/New%20folder%20(3)/Side_Effects_Drug_Analytics.ipynb)
- Dataset: [webmd.csv](file:///c:/Users/Shankar/Desktop/New%20folder%20(3)/webmd.csv)

## Features
- Data cleaning, statistical summaries, and correlation analysis
- Side effects categorization, frequency tracking, and visualization
- Text preprocessing, TF‑IDF, sentiment analysis, and topic modeling
- Predictive ML (Logistic Regression, Random Forest) and DL (LSTM/CNN, conceptual)
- Explainable AI: permutation importance, partial dependence, SHAP feature importance
- Dashboard‑style figures summarizing dataset metrics and model insights

## Executive Summary
- Aggregates patient reviews to reveal ADR patterns and their impact on satisfaction.
- Identifies top drivers of predictions using permutation importance and SHAP.
- Provides interpretable visualizations and a summary dashboard for stakeholders.
- Establishes a reproducible pipeline with clear data handling and modeling steps.

## Objectives & Research Questions
- Extract and categorize ADRs from patient‑generated text.
- Quantify demographic patterns (age, sex) in ADR reporting.
- Analyze relationships between satisfaction, effectiveness, ease of use, and side effects.
- Build transparent models to predict sentiment/satisfaction.
- RQs include frequency of side effects across drugs, demographic influences, and predictability of sentiment from side effect text.

## Methodology
- Data ingestion and cleaning, statistical profiling, and EDA.
- NLP pipeline: preprocessing, TF‑IDF, sentiment, topic modeling.
- ML: logistic regression baseline, Random Forest with tuning and evaluation.
- Explainability: permutation importance, partial dependence, SHAP summaries.
- Dashboard‑style wrap‑up figures for communication and decision support.

## Architecture & Workflow
The notebook provides a visual workflow (Figure 1.1) and lifecycle (Figure 1.2) illustrating the pipeline from data collection through modeling and explainability to dashboards.

## Repository Structure
```
Side_Effects_Drug_Analytics.ipynb     # End-to-end analysis notebook
webmd.csv                             # WebMD drug reviews dataset
python/                               # Optional Python assets
jupyter/                              # Optional Jupyter assets
*.png                                 # Generated figures across sections
Side effects of drug analytics.txt    # Notes
```

## Dataset
- **Source**: WebMD patient reviews compiled for research/analysis
- **Format**: CSV with structured fields and free‑text reviews
- **Location**: [webmd.csv](file:///c:/Users/Shankar/Desktop/New%20folder%20(3)/webmd.csv)
- **Original Dataset**: Available on [Kaggle - WebMD Drug Reviews Dataset](https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset)

## Data Dictionary
- Drug: Drug name
- Condition: Patient‑reported condition or indication
- Reviews: Free‑text review content
- Year: Review year (numeric)
- Satisfaction: Patient satisfaction score (e.g., 1–5)
- Effectiveness: Patient‑reported effectiveness (e.g., 1–5)
- Ease of Use: Usability score (e.g., 1–5)
- Side_Effects (if present): Extracted or tagged mentions of side effects
- Additional derived features appear during feature engineering (TF‑IDF, sentiment, etc.)

## Installation

### System Requirements
- **Python**: 3.10 or higher (3.11 recommended)
- **Operating System**: Windows 10/11, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **Storage**: 2GB free space for dependencies and data

### Step-by-Step Installation

#### 1. Clone or Download the Project
```bash
# Option A: Download ZIP from GitHub
# Download and extract to your desired directory

# Option B: If using Git
git clone <repository-url>
cd Side-Effects-Drug-Analytics
```

#### 2. Set Up Python Environment

**Option A: Using Conda (Recommended)**
```bash
# Create new environment with Python 3.11
conda create -n drug-adr python=3.11

# Activate the environment
conda activate drug-adr

# Verify Python version
python --version
```

**Option B: Using Virtual Environment**
```bash
# Create virtual environment
python -m venv drug-adr-env

# Activate environment
# On Windows:
drug-adr-env\Scripts\activate
# On macOS/Linux:
source drug-adr-env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### 3. Install Core Dependencies
```bash
# Install essential packages for data analysis
pip install pandas==2.1.0 numpy==1.24.3 matplotlib==3.7.2 seaborn==0.12.2 scikit-learn==1.3.0

# Install Jupyter for notebook execution
pip install jupyter==1.0.0 jupyterlab==4.0.5

# Install text processing and NLP libraries
pip install nltk==3.8.1 textblob==0.17.1 wordcloud==1.9.2
```

#### 4. Install Advanced Analytics Packages
```bash
# Install explainable AI libraries
pip install shap==0.42.1 lime==0.2.0.1

# Install visualization and dashboard libraries
pip install plotly==5.15.0 dash==2.11.1 streamlit==1.25.0

# Install additional utilities
pip install tqdm==4.65.0 openpyxl==3.1.2
```

#### 5. Verify Installation
```bash
# Test Python and key packages
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, shap; print('All packages installed successfully!')"

# Check Jupyter installation
jupyter --version
```

#### 6. Download Dataset
Ensure the dataset file is in the project root:
- Download `webmd.csv` from the [Kaggle - WebMD Drug Reviews Dataset](https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset)
- Place the downloaded file in the project directory
- Verify the file exists: `ls webmd.csv` (or `dir webmd.csv` on Windows)

#### 7. Launch Jupyter Notebook
```bash
# Start Jupyter Lab (recommended)
jupyter lab

# Or start classic Jupyter Notebook
jupyter notebook
```

### Troubleshooting Installation

**Common Issues:**
- **Package conflicts**: Use a fresh virtual environment
- **SHAP installation fails**: Try `pip install shap --no-deps` then install dependencies separately
- **Jupyter not found**: Ensure you're in the activated environment
- **Memory errors**: Increase virtual memory/swap space for large datasets

**Getting Help:**
- Check the [Troubleshooting](#troubleshooting) section below
- Review [Issues](https://github.com/your-repo/issues) for known problems
- Create a new issue with your error details and system information

## Configuration

### Environment Variables

The project supports configuration through environment variables. Create a `.env` file in the project root:

```bash
# .env file
# Data Configuration
DATA_PATH=webmd.csv
DATA_ENCODING=utf-8
MAX_ROWS_DISPLAY=1000

# Model Configuration
RANDOM_STATE=42
TEST_SIZE=0.2
N_ESTIMATORS=100
MAX_DEPTH=10

# Visualization Configuration
PLOT_STYLE=seaborn
FIGURE_DPI=300
FIGURE_FORMAT=png

# SHAP Configuration
SHAP_SAMPLES=1000
SHAP_MAX_DISPLAY=20

# Performance Configuration
N_JOBS=-1  # Use all available cores
MEMORY_LIMIT=4G

# Dashboard Configuration
DASHBOARD_THEME=light
COLOR_PALETTE=viridis
```

### Configuration File (config.json)

For more complex configurations, use a JSON configuration file:

```json
{
  "data": {
    "file_path": "webmd.csv",
    "encoding": "utf-8",
    "max_display_rows": 1000,
    "missing_value_threshold": 0.05
  },
  "preprocessing": {
    "text_cleaning": {
      "lowercase": true,
      "remove_punctuation": true,
      "remove_stopwords": true,
      "min_word_length": 3
    },
    "feature_engineering": {
      "create_sentiment": true,
      "create_review_length": true,
      "create_word_count": true
    }
  },
  "modeling": {
    "random_forest": {
      "n_estimators": 100,
      "max_depth": 10,
      "min_samples_split": 5,
      "random_state": 42
    },
    "logistic_regression": {
      "max_iter": 1000,
      "C": 1.0,
      "random_state": 42
    }
  },
  "explainability": {
    "shap": {
      "samples": 1000,
      "max_display": 20,
      "plot_type": "summary"
    },
    "permutation_importance": {
      "n_repeats": 10,
      "random_state": 42
    }
  },
  "visualization": {
    "style": "seaborn",
    "figure_size": [12, 8],
    "dpi": 300,
    "color_palette": "viridis"
  }
}
```

### Loading Configuration in Code

```python
import json
import os
from pathlib import Path

def load_config():
    """Load configuration from JSON file and environment variables."""
    
    # Load base configuration
    config_path = Path('config.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'data': {'file_path': 'webmd.csv', 'encoding': 'utf-8'},
            'modeling': {'random_state': 42, 'test_size': 0.2}
        }
    
    # Override with environment variables
    config['data']['file_path'] = os.getenv('DATA_PATH', config['data']['file_path'])
    config['modeling']['random_state'] = int(os.getenv('RANDOM_STATE', config['modeling']['random_state']))
    
    return config

# Usage
config = load_config()
df = pd.read_csv(config['data']['file_path'], encoding=config['data']['encoding'])
```

### Custom Settings for Different Environments

#### Development Environment
```bash
# Development settings
export RANDOM_STATE=42
export MAX_ROWS_DISPLAY=500
export SHAP_SAMPLES=500
export N_JOBS=2
```

#### Production Environment
```bash
# Production settings
export RANDOM_STATE=42
export MAX_ROWS_DISPLAY=10000
export SHAP_SAMPLES=5000
export N_JOBS=-1
export MEMORY_LIMIT=8G
```

#### Testing Environment
```bash
# Testing settings
export RANDOM_STATE=123
export MAX_ROWS_DISPLAY=100
export SHAP_SAMPLES=100
export TEST_SIZE=0.3
export N_JOBS=1
```

## API Reference

### Core Functions

#### Data Loading and Validation
```python
def load_drug_data(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load and validate the WebMD drug reviews dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing drug reviews
    encoding : str, default='utf-8'
        File encoding to use when reading the CSV
    
    Returns:
    --------
    pd.DataFrame
        Loaded and validated dataset with standardized column names
    
    Raises:
    -------
    FileNotFoundError
        If the specified file_path does not exist
    ValueError
        If the file format is invalid or missing required columns
    """
    pass

def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate the loaded dataset and return quality metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to validate
    
    Returns:
    --------
    dict
        Dictionary containing validation results:
        - 'missing_values': Percentage of missing values per column
        - 'data_types': Data type information
        - 'completeness_score': Overall data completeness (0-1)
        - 'warnings': List of validation warnings
    """
    pass
```

#### Text Processing and NLP
```python
def preprocess_text(text: str, 
                   lowercase: bool = True,
                   remove_punctuation: bool = True,
                   remove_stopwords: bool = True) -> str:
    """
    Preprocess text data for NLP analysis.
    
    Parameters:
    -----------
    text : str
        Input text to preprocess
    lowercase : bool, default=True
        Convert text to lowercase
    remove_punctuation : bool, default=True
        Remove punctuation marks
    remove_stopwords : bool, default=True
        Remove common stopwords
    
    Returns:
    --------
    str
        Preprocessed text
    """
    pass

def extract_sentiment_features(df: pd.DataFrame, 
                             text_column: str = 'Reviews') -> pd.DataFrame:
    """
    Extract sentiment-related features from text reviews.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing text reviews
    text_column : str, default='Reviews'
        Name of the column containing text data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional sentiment features:
        - 'sentiment_polarity': TextBlob polarity score (-1 to 1)
        - 'sentiment_subjectivity': TextBlob subjectivity score (0 to 1)
        - 'word_count': Number of words in review
        - 'avg_word_length': Average word length
    """
    pass
```

#### Machine Learning Pipeline
```python
def build_ml_pipeline(model_type: str = 'random_forest',
                     random_state: int = 42) -> Pipeline:
    """
    Build a machine learning pipeline for drug review analysis.
    
    Parameters:
    -----------
    model_type : str, default='random_forest'
        Type of model to build ('random_forest', 'logistic_regression')
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    Pipeline
        Scikit-learn pipeline with preprocessing and model
    """
    pass

def train_model(X_train: pd.DataFrame, 
                y_train: pd.Series,
                pipeline: Pipeline) -> Pipeline:
    """
    Train the machine learning model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target variable
    pipeline : Pipeline
        Machine learning pipeline to train
    
    Returns:
    --------
    Pipeline
        Trained pipeline
    """
    pass

def evaluate_model(model: Pipeline,
                  X_test: pd.DataFrame,
                  y_test: pd.Series) -> dict:
    """
    Evaluate model performance and return metrics.
    
    Parameters:
    -----------
    model : Pipeline
        Trained model to evaluate
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target variable
    
    Returns:
    --------
    dict
        Evaluation metrics:
        - 'accuracy': Model accuracy
        - 'precision': Weighted precision
        - 'recall': Weighted recall
        - 'f1_score': Weighted F1 score
        - 'confusion_matrix': Confusion matrix
        - 'classification_report': Full classification report
    """
    pass
```

#### Explainable AI Functions
```python
def calculate_shap_values(model: Pipeline,
                         X_test: pd.DataFrame,
                         samples: int = 1000) -> np.ndarray:
    """
    Calculate SHAP values for model interpretability.
    
    Parameters:
    -----------
    model : Pipeline
        Trained model
    X_test : pd.DataFrame
        Test data for SHAP calculation
    samples : int, default=1000
        Number of samples to use for SHAP calculation
    
    Returns:
    --------
    np.ndarray
        SHAP values for each feature and prediction
    """
    pass

def get_feature_importance(model: Pipeline,
                         X_test: pd.DataFrame,
                         method: str = 'permutation') -> pd.DataFrame:
    """
    Calculate feature importance using specified method.
    
    Parameters:
    -----------
    model : Pipeline
        Trained model
    X_test : pd.DataFrame
        Test data
    method : str, default='permutation'
        Method to use ('permutation', 'shap', 'built_in')
    
    Returns:
    --------
    pd.DataFrame
        Feature importance rankings with scores
    """
    pass

def create_explainability_dashboard(model: Pipeline,
                                  X_test: pd.DataFrame,
                                  y_test: pd.Series) -> dict:
    """
    Create a comprehensive explainability dashboard.
    
    Parameters:
    -----------
    model : Pipeline
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target variable
    
    Returns:
    --------
    dict
        Dashboard components:
        - 'shap_summary': SHAP summary plot data
        - 'feature_importance': Top features by importance
        - 'partial_dependence': PDP plots for top features
        - 'model_performance': Key performance metrics
    """
    pass
```

### Usage Examples

#### Basic Data Analysis
```python
# Load and analyze data
df = load_drug_data('webmd.csv')
validation_results = validate_dataset(df)

# Preprocess text reviews
df['processed_reviews'] = df['Reviews'].apply(preprocess_text)
sentiment_features = extract_sentiment_features(df)
```

#### Model Training and Evaluation
```python
# Build and train model
pipeline = build_ml_pipeline(model_type='random_forest')
trained_model = train_model(X_train, y_train, pipeline)

# Evaluate performance
metrics = evaluate_model(trained_model, X_test, y_test)
print(f"Model Accuracy: {metrics['accuracy']:.3f}")
```

#### Explainability Analysis
```python
# Calculate SHAP values
shap_values = calculate_shap_values(trained_model, X_test)

# Get feature importance
importance_df = get_feature_importance(trained_model, X_test, method='shap')

# Create dashboard
dashboard_data = create_explainability_dashboard(trained_model, X_test, y_test)
```

## Quickstart

### 🚀 Get Started in 5 Minutes

1. **Ensure dataset is available**
   ```bash
   # Check if webmd.csv exists in project root
   ls webmd.csv  # Windows: dir webmd.csv
   ```

2. **Launch Jupyter environment**
   ```bash
   jupyter lab Side_Effects_Drug_Analytics.ipynb
   ```

3. **Run the complete analysis**
   - Click "Run All" in Jupyter menu, or
   - Press `Ctrl+Enter` through each cell sequentially

4. **View results**: Generated PNG figures and analysis outputs appear throughout the notebook

## Usage Guide

### 📊 Data Analysis Workflow

The notebook is organized into 40 logical sections. Here's how to navigate and use each component:

#### Phase 1: Data Exploration & Cleaning
```python
# Example: Load and validate data (Section 2-5)
import pandas as pd

# Load the dataset
df = pd.read_csv('webmd.csv')

# Quick data validation
print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Data types:\n{df.dtypes}")
```

#### Phase 2: Statistical Analysis
```python
# Example: Basic statistics (Section 6-10)
# Satisfaction distribution analysis
satisfaction_stats = df['Satisfaction'].describe()
side_effect_counts = df['Side_Effects'].value_counts().head(10)

print("Patient Satisfaction Statistics:")
print(satisfaction_stats)
print("\nTop 10 Most Reported Side Effects:")
print(side_effect_counts)
```

#### Phase 3: Advanced Analytics & ML
```python
# Example: Machine Learning Pipeline (Section 24-27)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Prepare features and target
X = df[['Satisfaction', 'Effectiveness', 'Ease_of_Use', 'review_length']]
y = df['has_side_effects']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
predictions = rf_model.predict(X_test)
print(classification_report(y_test, predictions))
```

#### Phase 4: Explainable AI
```python
# Example: SHAP Analysis (Section 36)
import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Visualize feature importance
shap.summary_plot(shap_values[1], X_test, feature_names=X.columns)
```

### 🎯 Common Use Cases

#### Use Case 1: Healthcare Professional
**Goal**: Understand side effect patterns for specific drugs
```python
# Filter data for specific drug
drug_analysis = df[df['Drug'] == 'Metformin']
side_effects_by_condition = drug_analysis.groupby('Condition')['Side_Effects'].value_counts()

print("Side Effects by Condition for Metformin:")
print(side_effects_by_condition.head(10))
```

#### Use Case 2: Researcher
**Goal**: Analyze sentiment patterns in patient reviews
```python
# Example sentiment analysis (Section 20-23)
from textblob import TextBlob

# Calculate sentiment polarity
df['sentiment'] = df['Reviews'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Analyze sentiment by satisfaction level
sentiment_by_satisfaction = df.groupby('Satisfaction')['sentiment'].mean()
print("Average Sentiment by Satisfaction Level:")
print(sentiment_by_satisfaction)
```

#### Use Case 3: Patient Advocate
**Goal**: Generate summary reports for stakeholders
```python
# Create summary statistics
summary_stats = {
    'Total Reviews': len(df),
    'Average Satisfaction': df['Satisfaction'].mean(),
    'Most Common Side Effect': df['Side_Effects'].mode()[0],
    'Reviews with Side Effects': (df['has_side_effects'].sum() / len(df) * 100)
}

print("Patient Experience Summary:")
for key, value in summary_stats.items():
    print(f"{key}: {value:.1f}" if isinstance(value, float) else f"{key}: {value}")
```

### 📈 Expected Outputs

After running the complete notebook, you'll generate:

1. **Statistical Reports**: Distribution plots, correlation matrices, summary tables
2. **Visualizations**: Bar charts, word clouds, heatmaps, scatter plots
3. **Model Performance**: Accuracy scores, confusion matrices, ROC curves
4. **Explainability Charts**: SHAP summary plots, feature importance rankings
5. **Dashboard Panels**: Consolidated overview figures for presentations

### 🖼️ Sample Outputs

**Dashboard Overview** (generated in Section 37-40):
```
DATASET STATISTICS
==================
Total Reviews: 5,000
Unique Drugs: 150
Unique Conditions: 75
Date Range: 2020 - 2023

QUALITY SCORES
==================
Data Completeness: 94.2%
Avg Review Length: 156 chars
```

**Top Predictive Features** (from ML analysis):
1. Review sentiment score
2. Medication effectiveness rating
3. Ease of use score
4. Review length
5. Drug category
 
## Notebook Highlights
 - Section 24–27: Baseline and tuned ML models with metrics
 - Section 28: Deep learning model concept and guidance
 - Section 36: Explainable ML (Permutation, SHAP, Partial Dependence)
 - Section 37–40: Dashboard setup and wrap‑up visuals
 
## Background & Motivation
- ADRs are a significant clinical and economic burden; post‑marketing data is essential.
- Patient‑generated reviews provide real‑world evidence complementing clinical trials.
- This project operationalizes review data to surface safety signals and satisfaction drivers.

## Stakeholders & Use Cases
- Clinicians: understand side effect prevalence and satisfaction trade‑offs by drug/condition.
- Pharmacovigilance teams: track emergent ADR patterns and severity indicators.
- Patients & advocates: transparent summaries of experiences and likely side effects.
- Researchers: reproducible pipeline for NLP and explainability on patient narratives.

## Success Metrics (KPIs)
- Coverage: % of reviews successfully cleaned and analyzed.
- Model quality: accuracy, F1 for sentiment/satisfaction classification.
- Explainability: stability of feature rankings across methods and samples.
- Usability: clarity of dashboard summaries and visualizations.

## Visual Catalogue
Representative figures (saved to PNGs in the project root):
 - workflow_diagram.png
 - healthcare_lifecycle.png
 - feature_engineering.png
 - evaluation_metrics.png
 - explainable_ml.png
 - sentiment_analysis.png
 - condition_analysis.png
 - drug_comparison.png
 
## Explainability
- Permutation Importance: global feature influence on Random Forest
- SHAP: local and global feature contributions; mean |SHAP| for ranking
- Partial Dependence: average effect of top features on predictions
 
## Reproducibility
- Random seeds set where applicable (e.g., random_state=42)
- Figures saved to PNG for review
- Self‑contained notebook with explicit dependencies
- Reproducibility checklist:
  - Confirm Python version (3.10/3.11)
  - Install dependencies
  - Verify dataset path
  - Run notebook sequentially without skipping cells

## Troubleshooting
- SHAP shapes: For multi‑class models, SHAP may return lists or 3D arrays. The notebook normalizes to 1D feature‑wise values for summaries and printing.
- f‑Strings: Avoid conditionals inside format specifiers. The dashboard cell precomputes conditional strings before interpolation.
- Missing columns: Conditional logic handles absent columns (Drug, Condition, Reviews, Year) gracefully in summaries.
- Windows paths: Use absolute paths as linked above when opening files locally.

## Environment
- Tested on Windows 10/11 with Python 3.11
- JupyterLab/Notebook recommended for best experience
- GPU not required; SHAP computations can be limited via sampling

## Data Provenance & Quality
- Source: WebMD patient reviews dataset; see dataset link above.
- Integrity checks: null counts, type coercion, range validation for ratings.
- Completeness score: % non‑null cells across all columns.
- Outlier handling: review length bounds, abnormal rating values flagged.

## Preprocessing Pipeline
- Load CSV and normalize column names/types.
- Clean text: lowercasing, punctuation stripping, optional stop‑word removal.
- Feature engineering: sentiment lexicon features, TF‑IDF vectors, side effect tags.
- Train/test split with stratification for balanced evaluation.

## NLP Pipeline
- Text preprocessing (tokenization, normalization).
- TF‑IDF vectorization for bag‑of‑words features.
- Sentiment classification workflow; optional topic modeling for latent themes.
- Word clouds and frequency plots to summarize language patterns.

## ML Modeling Details
- Baseline: Logistic Regression on engineered features.
- Advanced: Random Forest with hyperparameter tuning; feature importance computed.
- Evaluation: accuracy, precision, recall, F1, ROC where applicable.
- Explainability: permutation importance and SHAP; PDP for top features.

## Dashboard Overview
- Consolidated panels: dataset stats, satisfaction gauge, top features.
- Visual bar charts for permutation and SHAP feature rankings.
- Narrative interpretation block summarizing model behavior.

## Performance & Compute
- SHAP computations can be heavy; the notebook samples X_test for tractability.
- Parallelization: Random Forest and permutation importance support n_jobs.
- Memory: large TF‑IDF matrices may require sparse representations (handled by scikit‑learn).

## Security, Privacy & Ethics
- Patient privacy: do not attempt re‑identification; aggregate insights only.
- Bias: reviews are self‑reported and may be non‑representative.
- Transparency: explainability methods help mitigate black‑box concerns.

## Limitations & Bias
- Dataset may lack clinical verification and structured adverse event coding.
- Sentiment models on lay text can conflate side effects and general dissatisfaction.
- Temporal coverage depends on dataset availability; trends may shift.

## Roadmap
- Add gradient boosting models (XGBoost/LightGBM)
- Experiment with modern embeddings for sentiment/ADR detection
- Promote the dashboard to a working Dash/Streamlit app
- Add automated tests and CI workflows
- Provide model cards and data statements

## Contributing

We welcome contributions from the community! This section provides detailed guidelines for contributing to the Side Effects Drug Analytics project.

### 🎯 Types of Contributions

**We actively seek contributions in the following areas:**

- **🐛 Bug Reports**: Report issues with code, documentation, or functionality
- **💡 Feature Requests**: Suggest new analytical methods, visualizations, or features
- **📊 Data Analysis**: Improve existing analyses or add new analytical approaches
- **🎨 Visualizations**: Create new plots, charts, or interactive dashboards
- **📚 Documentation**: Enhance README, add tutorials, or improve code comments
- **🔧 Code Improvements**: Refactor code, improve performance, or add tests
- **🧪 Model Enhancements**: Implement new ML algorithms or improve existing models

### 🔄 Contribution Workflow

#### 1. Fork and Clone the Repository
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Side-Effects-Drug-Analytics.git
cd Side-Effects-Drug-Analytics

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_AUTHOR/Side-Effects-Drug-Analytics.git
```

#### 2. Create a Feature Branch
```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# For bug fixes
git checkout -b bugfix/issue-description

# For documentation
git checkout -b docs/improvement-description
```

#### 3. Set Up Development Environment
```bash
# Create development environment
conda create -n drug-adr-dev python=3.11
conda activate drug-adr-dev

# Install development dependencies
pip install -r requirements-dev.txt
# or install core packages with development tools
pip install pandas numpy matplotlib seaborn scikit-learn shap plotly
pip install jupyter jupyterlab nbstripout pre-commit
```

#### 4. Make Your Changes
- Write clean, documented code following our standards
- Add tests for new functionality
- Update documentation as needed
- Ensure all existing tests pass

#### 5. Test Your Changes
```bash
# Run the complete notebook to ensure no errors
jupyter nbconvert --execute Side_Effects_Drug_Analytics.ipynb

# Run any additional tests
python -m pytest tests/  # if tests exist
```

#### 6. Commit and Push
```bash
# Stage your changes
git add -A

# Commit with descriptive message
git commit -m "feat: add new visualization for side effect trends

- Added interactive plotly chart for temporal analysis
- Included confidence intervals in trend visualization
- Updated documentation with usage examples

Closes #123"

# Push to your fork
git push origin feature/your-feature-name
```

#### 7. Create Pull Request
- Go to the original repository on GitHub
- Click "New Pull Request"
- Provide detailed description of changes
- Reference any related issues
- Submit for review

### 📋 Code Standards

#### Python Code Style
- **Follow PEP 8**: Use consistent indentation (4 spaces), naming conventions
- **Type Hints**: Add type annotations to all functions
- **Docstrings**: Use Google-style docstrings for all functions and classes
- **Line Length**: Maximum 88 characters (Black formatter standard)

```python
# Good example
def calculate_side_effect_frequency(df: pd.DataFrame, 
                                  drug_name: str,
                                  min_reports: int = 10) -> pd.Series:
    """Calculate frequency of side effects for a specific drug.
    
    Args:
        df: DataFrame containing drug review data
        drug_name: Name of the drug to analyze
        min_reports: Minimum number of reports required for inclusion
        
    Returns:
        Series with side effect frequencies, sorted by count
        
    Raises:
        ValueError: If drug_name not found in dataset
    """
    # Implementation here
    pass
```

#### Jupyter Notebook Guidelines
- **Clear Structure**: Use markdown cells to explain each section
- **Sequential Execution**: Ensure cells can be run in order without errors
- **Output Control**: Clear unnecessary output before committing
- **Comments**: Add comments to complex code cells
- **Visualizations**: Include appropriate titles, labels, and legends

#### Documentation Standards
- **README Updates**: Keep README.md current with new features
- **Inline Comments**: Explain complex logic and assumptions
- **Example Usage**: Provide working examples for new functions
- **API Documentation**: Update docstrings when modifying functions

### 🧪 Testing Requirements

#### For Code Contributions
- **Unit Tests**: Write tests for new functions using pytest
- **Integration Tests**: Test interaction between components
- **Data Tests**: Validate data loading and preprocessing
- **Model Tests**: Test ML model training and evaluation

```python
# Example test structure
def test_load_drug_data():
    """Test drug data loading functionality."""
    df = load_drug_data('webmd.csv')
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'Reviews' in df.columns
    assert 'Satisfaction' in df.columns

def test_model_performance():
    """Test ML model meets minimum performance criteria."""
    model = build_ml_pipeline()
    metrics = evaluate_model(model, X_test, y_test)
    
    assert metrics['accuracy'] >= 0.7  # Minimum 70% accuracy
    assert metrics['f1_score'] >= 0.65  # Minimum F1 score
```

#### For Notebook Contributions
- **Execution Test**: Ensure notebook runs without errors
- **Reproducibility**: Use fixed random seeds for reproducible results
- **Performance**: Verify reasonable execution time (< 10 minutes)
- **Output Quality**: Check that visualizations and results are meaningful

### 📊 Data and Model Standards

#### Data Quality
- **Validation**: Check for missing values, outliers, and data types
- **Documentation**: Document data sources and preprocessing steps
- **Privacy**: Ensure no personally identifiable information is exposed
- **Bias Assessment**: Evaluate potential biases in data and models

#### Model Requirements
- **Reproducibility**: Set random seeds and document hyperparameters
- **Performance**: Meet minimum accuracy/precision thresholds
- **Interpretability**: Include explainability analysis for ML models
- **Fairness**: Assess model performance across demographic groups

### 📝 Pull Request Template

When creating a pull request, please include:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement
- [ ] Other (please specify):

## Changes Made
- List specific changes made
- Include any new dependencies added
- Note breaking changes

## Testing
- [ ] Unit tests pass
- [ ] Notebook executes without errors
- [ ] New functionality tested
- [ ] Documentation updated

## Screenshots/Output
Include relevant visualizations or output if applicable.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

### 🎉 Recognition

Contributors will be:
- **Acknowledged** in the project README
- **Credited** in release notes for significant contributions
- **Invited** to collaborate on future project developments

### 📞 Getting Help

If you need help contributing:
1. Check existing [Issues](https://github.com/your-repo/issues) for similar questions
2. Create a new issue with the `help-wanted` label
3. Join our community discussions
4. Contact maintainers directly for sensitive issues

## Citation
If you use or build upon this work, please cite:
- Shankar Bhatt, “Side Effects Drug Analytics,” December 2025.
- Data source: WebMD reviews (verify and cite the original dataset provider as applicable).
BibTeX:
```bibtex
@misc{bhatt2025sideeffects,
  title  = {Side Effects Drug Analytics},
  author = {Shankar Bhatt},
  year   = {2025},
  note   = {Comprehensive ADR analytics using WebMD reviews, ML/DL, XAI},
}
```

 ### 📞 Getting Help

If you need help contributing:
1. Check existing [Issues](https://github.com/your-repo/issues) for similar questions
2. Create a new issue with the `help-wanted` label
3. Join our community discussions
4. Contact maintainers directly for sensitive issues

## License

### 📄 MIT License

```
MIT License

Copyright (c) 2025 Shankar Bhatt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### ⚠️ Important Licensing Notes

**Dataset Licensing**: This project uses the WebMD drug reviews dataset. Users must:
- Verify and comply with WebMD's terms of service and data usage policies
- Respect any additional licensing requirements from the original data provider
- Use the data for educational and research purposes only
- Not attempt to re-identify individuals from the dataset

**Commercial Use**: While the code is licensed under MIT, commercial use of the analysis results may be subject to additional restrictions based on the dataset licensing terms.

**Attribution**: When using this project, please cite both the code repository and the original data source appropriately.

## Acknowledgments

### 🙏 Contributors

**Primary Author**: Shankar Bhatt • December 2025

**Special Thanks To**:
- The healthcare and pharmacovigilance community for domain expertise
- Open-source contributors who have provided feedback and improvements
- Researchers in explainable AI and healthcare analytics for methodological insights

### 📊 Data Sources

- **WebMD Drug Reviews Dataset**: The primary dataset used in this analysis ([Kaggle - WebMD Drug Reviews Dataset](https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset))
- **NLTK and TextBlob**: Natural language processing libraries
- **Scikit-learn**: Machine learning algorithms and evaluation metrics
- **SHAP**: Explainable AI library for model interpretability

### 🛠️ Tools and Libraries

This project builds upon the excellent work of numerous open-source projects:

**Core Analytics Stack**:
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [NumPy](https://numpy.org/) - Numerical computing
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) - Data visualization
- [Scikit-learn](https://scikit-learn.org/) - Machine learning

**Specialized Libraries**:
- [SHAP](https://shap.readthedocs.io/) - Model explainability
- [Plotly](https://plotly.com/) and [Dash](https://dash.plotly.com/) - Interactive visualizations
- [NLTK](https://www.nltk.org/) and [TextBlob](https://textblob.readthedocs.io/) - Natural language processing

**Development Environment**:
- [Jupyter](https://jupyter.org/) - Interactive computing environment
- [Python](https://www.python.org/) - Programming language

### 📚 Academic References

If you use this project in academic research, please consider citing relevant methodological papers:

```bibtex
@misc{bhatt2025sideeffects,
  title={Side Effects Drug Analytics: Explainable AI for Healthcare Insights},
  author={Bhatt, Shankar},
  year={2025},
  howpublished={\url{https://github.com/your-repo/Side-Effects-Drug-Analytics}},
  note={Accessed: YYYY-MM-DD}
}

@article{lundberg2017unified,
  title={A unified approach to interpreting model predictions},
  author={Lundberg, Scott M and Lee, Su-In},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

### 🏥 Healthcare Disclaimer

This project is intended for **educational and research purposes only**. The analysis results should not be used for:
- Direct medical decision-making
- Substituting professional medical advice
- Commercial healthcare applications without proper validation
- Diagnosing or treating medical conditions

Always consult qualified healthcare professionals for medical decisions and advice.

### 🌟 Support the Project

If you find this project useful:
- ⭐ Give it a star on GitHub
- 🔄 Share it with colleagues and friends
- 🐛 Report issues and suggest improvements
- 🤝 Contribute to the codebase
- 📢 Cite it in your research and publications

Your support helps maintain and improve this open-source project for the benefit of the healthcare analytics community.
