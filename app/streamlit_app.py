# app/streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import utils

st.set_page_config(page_title="Investor Risk Profile Demo", layout="wide")

# -------------------------------
# Load Models and Cluster Profiles
# -------------------------------
model, scaler, le, FEATURE_COLUMNS = utils.load_model_objects()
cluster_profiles = utils.load_cluster_profiles()

# -------------------------------
# Sidebar Input Form
# -------------------------------
st.sidebar.header("User Input Form")

def user_input_form():
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.sidebar.number_input("Income ($)", min_value=1000, max_value=1_000_000, value=50000)
    experience_years = st.sidebar.number_input("Experience Years", min_value=0, max_value=50, value=5)
    risk_survey_score = st.sidebar.number_input("Risk Survey Score", min_value=0, max_value=100, value=50)
    total_transactions = st.sidebar.number_input("Total Transactions", min_value=0, max_value=1000, value=10)
    avg_amount = st.sidebar.number_input("Average Transaction Amount", min_value=0.0, value=200.0)
    active_days = st.sidebar.number_input("Active Days", min_value=0, max_value=5000, value=365)
    monthly_freq = st.sidebar.number_input("Monthly Frequency", min_value=0, max_value=100, value=1)
    tx_count_30d = st.sidebar.number_input("Transactions Last 30 Days", min_value=0, max_value=500, value=20)
    pct_crypto = st.sidebar.slider("Percentage Crypto", 0.0, 1.0, 0.2)
    pct_etf = st.sidebar.slider("Percentage ETF", 0.0, 1.0, 0.1)
    pct_forex = st.sidebar.slider("Percentage Forex", 0.0, 1.0, 0.1)
    pct_other = st.sidebar.slider("Percentage Other", 0.0, 1.0, 0.3)
    pct_stocks = st.sidebar.slider("Percentage Stocks", 0.0, 1.0, 0.3)
    pct_bond = st.sidebar.slider("Percentage Bond", 0.0, 1.0, 0.1)
    volatility_mean_30 = st.sidebar.number_input("Volatility Mean 30", min_value=0.0, value=0.05)
    volatility_mean_60 = st.sidebar.number_input("Volatility Mean 60", min_value=0.0, value=0.06)
    volatility_std_30 = st.sidebar.number_input("Volatility Std 30", min_value=0.0, value=0.02)

    data = {
        "age": age,
        "income": income,
        "experience_years": experience_years,
        "risk_survey_score": risk_survey_score,
        "total_transactions": total_transactions,
        "avg_amount": avg_amount,
        "active_days": active_days,
        "monthly_freq": monthly_freq,
        "tx_count_30d": tx_count_30d,
        "pct_crypto": pct_crypto,
        "pct_etf": pct_etf,
        "pct_forex": pct_forex,
        "pct_other": pct_other,
        "pct_stocks": pct_stocks,
        "pct_bond": pct_bond,
        "volatility_mean_30": volatility_mean_30,
        "volatility_mean_60": volatility_mean_60,
        "volatility_std_30": volatility_std_30
    }
    return pd.DataFrame([data])

input_df = user_input_form()

# -------------------------------
# Main Panel
# -------------------------------
st.title("Investor Risk Profile Demo")
st.write("This demo predicts the investor risk profile based on input features and assigns the user to a cluster.")

# Predict risk
risk_label = utils.predict_risk(input_df, model, scaler, le, FEATURE_COLUMNS)
st.subheader("Predicted Risk Level")
st.success(f"{risk_label}")

# Assign cluster
cluster_label = utils.assign_cluster(input_df, cluster_profiles)
st.subheader("Assigned Cluster")
st.info(f"Cluster {cluster_label}")

# SHAP explanation
st.subheader("SHAP Local Explanation")
shap_df, expected_value = utils.shap_local_explanation(input_df, model, scaler, FEATURE_COLUMNS)

# Top 5 SHAP features
top_features = shap_df.T.abs().mean(axis=1).sort_values(ascending=False).head(5).index
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=shap_df[top_features].iloc[0].values, y=top_features, palette="viridis", ax=ax)
ax.set_xlabel("SHAP Value")
ax.set_title("Top 5 Features Contribution")
st.pyplot(fig)

# Cluster profile bar chart
st.subheader("Cluster Portfolio Profile")
cluster_data = cluster_profiles.loc[cluster_label]
fig2 = px.bar(
    x=cluster_data.index,
    y=cluster_data.values,
    labels={"x": "Feature", "y": "Average Value"},
    title=f"Cluster {cluster_label} Average Feature Values"
)
st.plotly_chart(fig2, use_container_width=True)

# Commentary
st.subheader("Commentary")
st.write(f"User is assigned to Cluster {cluster_label} with predicted risk level '{risk_label}'.")
st.write("Top features affecting this prediction are shown above via SHAP values.")
st.write("Cluster profile allows comparison against average behavior in this cluster.")
