import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "intermediate", "user_features_clustered.csv")

MODEL_PATH = "../models/risk_model.pkl"
SCALER_PATH = "../models/scaler.pkl"
LE_PATH = "../models/label_encoder.pkl"
TARGET = "risk_level"
DROP_COLS = ["first_tx_date", "last_tx_date"]

def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLS)
    df['pct_other'] = df['pct_other'].fillna(df['pct_other'].median())
    return df

def preprocess(df, target_col):
    X = df.drop(columns=[target_col, "user_id"])
    y = df[target_col]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le

def split_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, y_train, X_train_scaled):
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train_scaled, y_train)
    
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    
    return lr, rf

def evaluate_model(model, X_test, y_test, model_name="Model"):
    pred = model.predict(X_test)
    print(f"{model_name} Accuracy:", accuracy_score(y_test, pred))
    print(f"{model_name} Classification Report:\n", classification_report(y_test, pred))
    print(f"{model_name} Confusion Matrix:\n", confusion_matrix(y_test, pred))
    return pred

def save_artifacts(model, scaler, le, model_path, scaler_path, le_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    X, y_encoded, le = preprocess(df, TARGET)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_scale(X, y_encoded)
    
    lr, rf = train_models(X_train, y_train, X_train_scaled)
    
    print("\n--- Logistic Regression Evaluation ---")
    evaluate_model(lr, X_test_scaled, y_test, "Logistic Regression")
    
    print("\n--- Random Forest Evaluation ---")
    evaluate_model(rf, X_test, y_test, "Random Forest")
    
    save_artifacts(rf, scaler, le, MODEL_PATH, SCALER_PATH, LE_PATH)
    print("\nModels and preprocessing objects saved successfully!")