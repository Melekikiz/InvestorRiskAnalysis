import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import os

#USERS DATA
N_USERS = 2000

np.random.seed(42)

ages = np.random.normal(loc = 38, scale = 10, size = N_USERS).astype(int)
ages = np.clip(ages, 18, 70)

income_distribution = (
    np.random.lognormal(mean=10.5, sigma=0.6, size=N_USERS) / 100
).astype(int)

experience_years = np.clip(
    (ages - 18)*np.random.uniform(0.2, 0.8, size=N_USERS),0,None
).astype(int)

occupations = [
    "Software Engineer", "Teacher", "Doctor", "Sales", "Nurse", "Finance Analyst",
    "Civil Engineer", "Manager", "Freelancer", "Student"
]

users = pd.DataFrame({
    "user_id": range(1, N_USERS + 1),
    "age":ages,
    "income": income_distribution,
    "experience_years": experience_years
})

#Market Data
assets =["AAPL", "BTC", "ETH", "TSLA", "GLD", "S&P500", "EURUSD"]
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31)
dates = pd.date_range(start_date, end_date)

market_records=[]

for asset in assets:
    price = 100 + np.random.rand() * 20
    for date in dates:
        price += np.random.normal(0, 1.2) #daily volatility
        price = max(price, 1)
        market_records.append([date, asset, round(price, 2)])

market_data = pd.DataFrame(market_records, columns=["date", "asset", "price"])

#Risk Survey
risk_survey = pd.DataFrame({
    "user_id": users["user_id"],
    "q1":np.random.randint(1, 6, N_USERS),
    "q2":np.random.randint(1, 6, N_USERS),
    "q3": np.random.randint(1, 6, N_USERS),
    "q4": np.random.randint(1, 6, N_USERS),
    "q5": np.random.randint(1, 6, N_USERS)
})

risk_survey["total_score"]= risk_survey[["q1","q2", "q3", "q4", "q5"]].sum(axis=1)

risk_survey["total_score"] += (users["age"] < 30).astype(int) * np.random.randint(2, 4, N_USERS)
risk_survey["total_score"] -= (users["age"] > 55).astype(int) * np.random.randint(1, 3, N_USERS)
risk_survey["total_score"] += (users["income"] > 80000).astype(int) * np.random.randint(0, 3, N_USERS)

risk_survey["total_score"] = risk_survey["total_score"].clip(5, 25)

#Transactions Data
transaction_types = ["buy", "sell"]
transactions = []

for i in range(N_USERS):
    user = users.iloc[i]
    risk = risk_survey.iloc[i]["total_score"]

    num_tx = np.random.poisson(lam=5 + risk/5 + user["income"]/50000)

    for _ in range(num_tx):
        asset = np.random.choice(assets, p=[0.15,0.2,0.2,0.1,0.1,0.2,0.05])
        amount = round(abs(np.random.normal(500 + risk*20, 200)), 2)

        tx_date = start_date + timedelta(days = np.random.randint(0, (end_date - start_date).days))

        transactions.append([
            user["user_id"],
            random.choice(transaction_types),
            tx_date,
            amount,
            asset
        ])

transactions_df = pd.DataFrame(transactions, columns=[
    "user_id", "transaction_type", "date", "amount", "asset"
])

#Save Data
os.makedirs("data", exist_ok=True)

users.to_csv("data/users.csv", index=False)
market_data.to_csv("data/market_data.csv", index=False)
risk_survey.to_csv("data/risk_survey.csv", index=False)
transactions_df.to_csv("data/transactions.csv", index=False)

print("Synthetic data generated and saved in 'data' directory.")