"""
Cyber4Africa MVP - AI Fraud Detection Model Trainer
Generates synthetic financial transaction data and trains a fraud detection model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

np.random.seed(42)

def generate_synthetic_data(n_samples=10000):
    """Generate realistic synthetic financial transaction data."""
    n_legit = int(n_samples * 0.97)
    n_fraud = n_samples - n_legit

    # Legitimate transactions
    legit = pd.DataFrame({
        'amount': np.random.lognormal(mean=4.5, sigma=1.2, size=n_legit),
        'hour': np.random.choice(range(8, 22), size=n_legit),  # business hours
        'day_of_week': np.random.randint(0, 5, size=n_legit),  # weekdays
        'num_transactions_today': np.random.poisson(lam=5, size=n_legit),
        'avg_transaction_amount': np.random.lognormal(mean=4.2, sigma=0.8, size=n_legit),
        'account_age_days': np.random.randint(30, 3650, size=n_legit),
        'failed_attempts': np.random.choice([0, 1], size=n_legit, p=[0.95, 0.05]),
        'is_international': np.random.choice([0, 1], size=n_legit, p=[0.9, 0.1]),
        'device_change': np.random.choice([0, 1], size=n_legit, p=[0.92, 0.08]),
        'location_change': np.random.choice([0, 1], size=n_legit, p=[0.85, 0.15]),
        'Class': 0
    })

    # Fraudulent transactions
    fraud = pd.DataFrame({
        'amount': np.random.lognormal(mean=6.5, sigma=1.5, size=n_fraud),  # higher amounts
        'hour': np.random.choice(list(range(0, 6)) + list(range(22, 24)), size=n_fraud),  # odd hours
        'day_of_week': np.random.randint(0, 7, size=n_fraud),
        'num_transactions_today': np.random.poisson(lam=15, size=n_fraud),  # many transactions
        'avg_transaction_amount': np.random.lognormal(mean=3.0, sigma=1.5, size=n_fraud),
        'account_age_days': np.random.randint(1, 60, size=n_fraud),  # new accounts
        'failed_attempts': np.random.choice([0, 1, 2], size=n_fraud, p=[0.3, 0.4, 0.3]),
        'is_international': np.random.choice([0, 1], size=n_fraud, p=[0.4, 0.6]),
        'device_change': np.random.choice([0, 1], size=n_fraud, p=[0.3, 0.7]),
        'location_change': np.random.choice([0, 1], size=n_fraud, p=[0.2, 0.8]),
        'Class': 1
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1).reset_index(drop=True)
    return df

def train_and_save_model():
    print("🔄 Generating synthetic transaction data...")
    df = generate_synthetic_data(10000)

    features = ['amount', 'hour', 'day_of_week', 'num_transactions_today',
                'avg_transaction_amount', 'account_age_days', 'failed_attempts',
                'is_international', 'device_change', 'location_change']

    X = df[features]
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("🤖 Training Random Forest fraud detection model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("\n📊 Model Performance:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/fraud_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(features, 'model/features.pkl')

    # Save sample transactions for demo
    sample = df.sample(50)
    sample.to_csv('model/sample_transactions.csv', index=False)

    print("\n✅ Model saved to model/fraud_model.pkl")
    print("✅ Sample transactions saved to model/sample_transactions.csv")
    return model, scaler, features

if __name__ == '__main__':
    train_and_save_model()
