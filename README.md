# InvestorRiskAnalysis
# Investor Risk Analysis

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://investorriskanalysis-7ky6wethbz3jrnqlz8ddra.streamlit.app/)

## Project Overview
This project predicts investor risk profiles based on user features and assigns users to behavioral clusters. It leverages a Random Forest model for classification and provides SHAP explanations for model interpretability. Clustering analysis offers insight into different user segments and portfolio behaviors.

## Features
- Age, income, experience years, risk survey score  
- Transaction metrics: total transactions, average amount, monthly frequency, transactions in last 30 days  
- Portfolio distribution: percentages of crypto, stocks, ETFs, bonds, forex, other  
- Volatility measures: mean and std for 30- and 60-day windows  

## Functionality
- **Risk Prediction:** Users can input their information and get a predicted risk level (low, medium, high).  
- **Cluster Assignment:** Users are assigned to one of 3 behavioral clusters.  
- **SHAP Explanations:** Visual explanation of top features affecting model predictions.  
- **Cluster Insights:** PCA visualization, distribution charts, and violin plots to understand segment behavior.  

## How to Run
1. Clone this repository:
```bash
git clone https://github.com/Melekikiz/InvestorRiskAnalysis.git
pip install -r requirements.txt
streamlit run app/streamlit_app.py


app/                # Streamlit app and utility scripts
data/               # Raw data files
intermediate/       # Preprocessed features and clustered data
models/             # Saved model, scaler, label encoder
notebooks/          # EDA, modeling, SHAP analysis
reports/            # Visualizations for SHAP and clustering
scripts/            # Data preparation and model training scripts
README.md           # Project overview and instructions
requirements.txt    # Python dependencies
