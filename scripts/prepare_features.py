import os
import sqlite3
from datetime import timedelta
import pandas as pd
import numpy as np

DATA_DIR= "data"
INTERMEDIATE_DIR = "intermediate"
SQLITE_DB = "data/investor_data.db"
SAVE_SQLITE = True
VERBOSE = True

os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

# Read CSV Files
if VERBOSE:
    print("Reading CSVs from:", DATA_DIR)

users_path = os.path.join(DATA_DIR, "users.csv")
transactions_path = os.path.join(DATA_DIR, "transactions.csv")
market_path = os.path.join(DATA_DIR, "market_data.csv")
risk_path = os.path.join(DATA_DIR, "risk_survey.csv")

users = pd.read_csv(users_path)
transactions = pd.read_csv(transactions_path)
market = pd.read_csv(market_path)
risk = pd.read_csv(risk_path)

if VERBOSE:
    print("Shapes => users:", users.shape, "transactions:", transactions.shape, 
          "market:", market.shape, "risk:", risk.shape)

# Simple Clean up
for col in ["date"]:
    if col in transactions.columns:
        transactions["date"] = pd.to_datetime(transactions["date"])
if "date" in market.columns:
    market["date"] = pd.to_datetime(market["date"])

if "amount" in transactions.columns:
    transactions["amount"] = pd.to_numeric(transactions["amount"], errors="coerce")
if "price" in market.columns:
    market["price"] = pd.to_numeric(market["price"], errors="coerce")

# Minor column name fixes
if "ecperience_years" in users.columns:
    users = users.rename(columns={"ecperience_years": "experience_years"})

# Missing values
if VERBOSE:
    print("\nMissing values summary:")
    print("Users:\n", users.isna().sum())
    print("Transactions:\n", transactions.isna().sum())
    print("Market:\n", market.isna().sum())
    print("Risk:\n", risk.isna().sum())

# Optionally save to SQLite
if SAVE_SQLITE:
    if VERBOSE:
        print("\nWriting CSVs to SQLite DB:", SQLITE_DB)
    conn = sqlite3.connect(SQLITE_DB)
    users.to_sql("users", conn, index=False, if_exists="replace")
    transactions.to_sql("transactions", conn, index=False, if_exists="replace")
    market.to_sql("market_data", conn, index=False, if_exists="replace")
    risk.to_sql("risk_survey", conn, index=False, if_exists="replace")

    if VERBOSE:
        q = "SELECT user_id, COUNT(*) as cnt FROM transactions GROUP BY user_id ORDER BY cnt DESC LIMIT 5;"
        print("\nTop 5 users by tx count (sample SQL):")
        print(pd.read_sql(q, conn))

# Basic summaries
tx_per_user = transactions.groupby("user_id").size().rename("total_transactions").reset_index()

# Monthly transaction frequency
transactions["year_month"] = transactions["date"].dt.to_period("M").astype(str)
monthly_counts = (
    transactions.groupby(["user_id", "year_month"])
    .size()
    .rename("monthly_tx_count")
    .reset_index()
)

# Feature Engineering
if VERBOSE:
    print("\nStarting Feature Engineering...")

# Demographics
user_feats = users.copy()

if "total_score" in risk.columns:
    risk = risk.rename(columns={"total_score": "risk_survey_score"})
user_feats = user_feats.merge(risk[["user_id", "risk_survey_score"]], on="user_id", how="left")

# Total transactions
user_feats = user_feats.merge(tx_per_user, on="user_id", how="left")
user_feats["total_transactions"] = user_feats["total_transactions"].fillna(0).astype(int)

# Average transaction amount
avg_amount = transactions.groupby("user_id")["amount"].mean().rename("avg_amount").reset_index()
user_feats = user_feats.merge(avg_amount, on="user_id", how="left")
user_feats["avg_amount"] = user_feats["avg_amount"].fillna(0.0)

# First & last transaction dates, active days
first_last = transactions.groupby("user_id")["date"].agg(["min", "max"]).reset_index().rename(
    columns={"min": "first_tx_date", "max": "last_tx_date"})
user_feats = user_feats.merge(first_last, on="user_id", how="left")
user_feats["first_tx_date"] = pd.to_datetime(user_feats["first_tx_date"])
user_feats["last_tx_date"] = pd.to_datetime(user_feats["last_tx_date"])
user_feats["active_days"] = (user_feats["last_tx_date"] - user_feats["first_tx_date"]).dt.days.fillna(0).astype(int)

# Monthly frequency last 12 months
global_last_date = transactions["date"].max()
one_year_ago = global_last_date - pd.DateOffset(months=12)
tx_last_12m = transactions[transactions["date"] >= one_year_ago]
monthly_12m = tx_last_12m.groupby(["user_id", tx_last_12m["date"].dt.to_period("M")]).size().reset_index(name="cnt")
monthly_avg = monthly_12m.groupby("user_id")["cnt"].mean().rename("monthly_freq").reset_index()
user_feats = user_feats.merge(monthly_avg, on="user_id", how="left")
user_feats["monthly_freq"] = user_feats["monthly_freq"].fillna(0.0)

# Transactions last 30 days
thirty_days_ago = global_last_date - pd.Timedelta(days=30)
tx_30d = transactions[transactions["date"] >= thirty_days_ago].groupby("user_id").size().rename("tx_count_30d").reset_index()
user_feats = user_feats.merge(tx_30d, on="user_id", how="left")
user_feats["tx_count_30d"] = user_feats["tx_count_30d"].fillna(0).astype(int)

# Asset-based percentages
crypto_list = ["BTC", "ETH"]
stock_list = ["AAPL", "TSLA"]
etf_list = ["S&P500"]
bond_list = []
forex_list = ["EURUSD"]

def asset_category(a):
    if a in crypto_list:
        return "crypto"
    if a in stock_list:
        return "stock"
    if a in etf_list:
        return "etf"
    if a in bond_list:
        return "bond"
    if a in forex_list:
        return "forex"
    return "other"

transactions["asset_category"] = transactions["asset"].apply(asset_category)

# --- FIXED BLOCK ---
asset_counts = (
    transactions.groupby(["user_id", "asset_category"])
    .size()
    .rename("cnt")
    .reset_index()
)

user_totals = (
    asset_counts.groupby("user_id")["cnt"]
    .sum()
    .rename("total_cnt")
    .reset_index()
)

asset_pct = asset_counts.merge(user_totals, on="user_id", how="left")
asset_pct["pct"] = asset_pct["cnt"] / asset_pct["total_cnt"]

asset_pct = (
    asset_pct.pivot(index="user_id", columns="asset_category", values="pct")
    .fillna(0)
    .reset_index()
)

user_feats = user_feats.merge(asset_pct, on="user_id", how="left")

# Fill missing pct columns
for col in ["crypto", "stock", "etf", "bond", "forex", "other"]:
    if col not in user_feats.columns:
        user_feats[col] = 0.0

user_feats = user_feats.rename(columns={
    "crypto": "pct_crypto",
    "stock": "pct_stocks",
    "etf": "pct_etf",
    "bond": "pct_bond",
    "forex": "pct_forex",
    "other": "pct_other"
})

# Volatility
market_sorted = market.sort_values(["asset", "date"]).copy()
market_sorted["return"] = market_sorted.groupby("asset")["price"].pct_change()
market_sorted["vol_30"] = market_sorted.groupby("asset")["return"].rolling(window=30, min_periods=5).std().reset_index(level=0, drop=True)
market_sorted["vol_60"] = market_sorted.groupby("asset")["return"].rolling(window=60, min_periods=10).std().reset_index(level=0, drop=True)

user_asset_counts = transactions.groupby(["user_id", "asset"]).size().rename("count").reset_index()

asset_vol_recent = (
    market_sorted.groupby("asset")[["vol_30", "vol_60"]]
    .mean()
    .reset_index()
    .rename(columns={"vol_30": "asset_vol_30_mean", "vol_60": "asset_vol_60_mean"})
)

user_asset = user_asset_counts.merge(asset_vol_recent, on="asset", how="left")

user_total = (
    user_asset.groupby("user_id")["count"]
    .sum()
    .reset_index()
    .rename(columns={"count": "total_count"})
)

user_vol = (
    user_asset.assign(
        weighted_v30=lambda df: df["count"] * df["asset_vol_30_mean"],
        weighted_v60=lambda df: df["count"] * df["asset_vol_60_mean"]
    )
    .groupby("user_id")
    .agg(
        volatility_mean_30=("weighted_v30", "sum"),
        volatility_mean_60=("weighted_v60", "sum")
    )
    .reset_index()
)

user_vol = user_vol.merge(user_total, on="user_id", how="left")

user_vol["volatility_mean_30"] = user_vol["volatility_mean_30"] / user_vol["total_count"]
user_vol["volatility_mean_60"] = user_vol["volatility_mean_60"] / user_vol["total_count"]

user_vol = user_vol[["user_id", "volatility_mean_30", "volatility_mean_60"]]

user_feats = user_feats.merge(user_vol, on="user_id", how="left")
user_feats["volatility_mean_30"] = user_feats["volatility_mean_30"].fillna(0.0)
user_feats["volatility_mean_60"] = user_feats["volatility_mean_60"].fillna(0.0)

user_asset_vol_std = user_asset.groupby("user_id")["asset_vol_30_mean"].std().rename("volatility_std_30").reset_index()
user_feats = user_feats.merge(user_asset_vol_std, on="user_id", how="left")
user_feats["volatility_std_30"] = user_feats["volatility_std_30"].fillna(0.0)

# Cleanup & type corrections
numeric_cols = ["income", "risk_survey_score", "total_transactions", "avg_amount",
                "monthly_freq", "tx_count_30d",
                "pct_crypto", "pct_stocks", "pct_etf", "pct_bond", "pct_forex",
                "volatility_mean_30", "volatility_mean_60", "volatility_std_30"]

for c in numeric_cols:
    if c in user_feats.columns:
        user_feats[c] = pd.to_numeric(user_feats[c], errors="coerce").fillna(0.0)

if "first_tx_date" in user_feats.columns:
    user_feats["first_tx_date"] = pd.to_datetime(user_feats["first_tx_date"]).dt.date
if "last_tx_date" in user_feats.columns:
    user_feats["last_tx_date"] = pd.to_datetime(user_feats["last_tx_date"]).dt.date

if "risk_survey_score" in user_feats.columns:
    bins = [0, 10, 17, 30]
    labels = ["low", "medium", "high"]
    user_feats["risk_level"] = pd.cut(user_feats["risk_survey_score"], bins=bins, labels=labels, include_lowest=True)

# Save CSV
out_path = os.path.join(INTERMEDIATE_DIR, "user_features.csv")
user_feats.to_csv(out_path, index=False)
if VERBOSE:
    print("\nSaved user features to:", out_path)
    print("user_features shape:", user_feats.shape)
    print(user_feats.head().T)

if VERBOSE:
    print("\nQuick summary stats:")
    print(user_feats[["total_transactions", "avg_amount", "monthly_freq", "tx_count_30d"]].describe())

# Close SQLite connection
if SAVE_SQLITE:
    conn.close()
