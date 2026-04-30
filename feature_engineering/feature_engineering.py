import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

#load preprocessed data splits
X_train = pd.read_csv("preprocessing/train_set.csv")
X_test = pd.read_csv("preprocessing/test_set.csv")
X_Valid = pd.read_csv("preprocessing/valid_set.csv")

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    safe_balance = np.clip(df["AccountBalance"].to_numpy(), 1.0, None)
    df["amount_balance_ratio"] = df["TransactionAmount"] / safe_balance
    df["night_transaction"] = (df["transaction_hour"] <= 5).astype(int)
    df["high_login_attempts"] = (df["LoginAttempts"] > 3).astype(int)
    safe_duration = np.clip(df["TransactionDuration"].to_numpy(), 1.0, None)
    df["quick_transactions"] = df["TransactionAmount"] / safe_duration
    
    df = df.drop(columns=["transaction_hour"])
    return df

X_train = add_features(X_train)
X_test = add_features(X_test)
X_Valid = add_features(X_Valid)

print("Features after feature engineering:", X_train.shape[1])

#saving the data for isolation forest
os.makedirs("feature_engineering", exist_ok=True)
X_train.to_csv("feature_engineering/X_train_isolation.csv", index=False)
X_test.to_csv("feature_engineering/X_test_isolation.csv", index=False)
X_Valid.to_csv("feature_engineering/X_valid_isolation.csv", index=False)

# Autoencoder will be trained on all features, including the engineered ones
columns = ["TransactionAmount", "CustomerAge", "TransactionDuration", "AccountBalance", "LoginAttempts",
           "days_since_last_transaction", "amount_balance_ratio", "quick_transactions", "Location"]

autoen_scaler = MinMaxScaler()
X_train_autoen = X_train.copy()
X_test_autoen = X_test.copy()
X_Valid_autoen = X_Valid.copy()

X_train_autoen[columns] = autoen_scaler.fit_transform(X_train_autoen[columns])
X_test_autoen[columns] = autoen_scaler.transform(X_test_autoen[columns])
X_Valid_autoen[columns] = autoen_scaler.transform(X_Valid_autoen[columns])

cyclical_columns = ["hour_sin", "hour_cos", "week_sin", "week_cos", "month_sin", "month_cos"]
for cols in cyclical_columns:
    X_train_autoen[cols] = (X_train_autoen[cols]+1)/2
    X_test_autoen[cols] = (X_test_autoen[cols]+1)/2
    X_Valid_autoen[cols] = (X_Valid_autoen[cols]+1)/2
    
#Save autoencoder data
os.makedirs("saved_scalers", exist_ok=True)
joblib.dump(autoen_scaler, "saved_scalers/autoen_minmax_scaler.pkl")

X_train_autoen.to_csv ("feature_engineering/X_train_autoen.csv", index=False)
X_test_autoen.to_csv ("feature_engineering/X_test_autoen.csv", index=False)
X_Valid_autoen.to_csv ("feature_engineering/X_Valid_autoen.csv", index=False)

print("FEATURE ENGINEERING COMPLETED")
print("Isolation Forest train set shape:", X_train.shape)
print("Autoencoder train set shape:", X_train_autoen.shape)
print("\nAutoencoder train ranges:")
print(X_train_autoen.agg(["min", "max"]).T.to_string())

#next folder - models/training