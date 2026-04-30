#This file follows 7 steps to complete preprocessing, 
#feature engineering is separated to a different file

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

#1: Load dataset from data folder
data = pd.read_csv("data/bank_transactions_data_2.csv")

#2. Parsing TransactionDate
data["TransactionDate"] = pd.to_datetime(data["TransactionDate"], dayfirst=True, errors="coerce")

#3: Create new feature - days since last transaction for each accountID
data = data.sort_values(["AccountID", "TransactionDate"]).reset_index(drop=True)
data["days_since_last_transaction"] = data.groupby("AccountID")["TransactionDate"].diff().dt.total_seconds() / (24 * 3600)
data["days_since_last_transaction"] = data["days_since_last_transaction"].fillna(0)

#4: Cyclical encoding of time features 
hour = data["TransactionDate"].dt.hour
week = data["TransactionDate"].dt.dayofweek
month = data["TransactionDate"].dt.month

data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
data["week_sin"] = np.sin(2 * np.pi * week / 7)
data["week_cos"] = np.cos(2 * np.pi * week / 7)
data["month_sin"] = np.sin(2 * np.pi * month / 12)
data["month_cos"] = np.cos(2 * np.pi * month / 12)
data["transaction_hour"] = hour

drop_columns = ["TransactionID","TransactionDate", "AccountID", "DeviceID", "IP Address", "MerchantID", "PreviousTransactionDate"]
data.drop(columns=drop_columns, inplace=True)

#5: Encode categorical columns
data["TransactionType"] = (data["TransactionType"] == "Debit").astype(int)
data = pd.get_dummies(data, columns = ["Channel", "CustomerOccupation"], drop_first=False)
bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype(int)

#6: Train/Test/Valid Split - 70% training 15% testing 15% validation
X_train, X_temp = train_test_split (data, test_size=0.3, random_state=42)
X_valid, X_test = train_test_split (X_temp, test_size=0.5, random_state=42)

#Frequency encoding of Location
location_frequency = X_train["Location"].value_counts(normalize=True)
X_train["Location"] = X_train["Location"].map(location_frequency)
X_valid["Location"] = X_valid["Location"].map(location_frequency).fillna(0)
X_test["Location"] = X_test["Location"].map(location_frequency).fillna(0)

#7: Save the preprocessed data for later use 
os.makedirs("preprocessing", exist_ok=True)
os.makedirs("saved_scalers", exist_ok=True)
X_train.to_csv("preprocessing/train_set.csv", index=False)
X_test.to_csv("preprocessing/test_set.csv", index=False)
X_valid.to_csv("preprocessing/valid_set.csv", index=False)
joblib.dump(location_frequency, "saved_scalers/location_frequency.pkl")

print("Preprocessing completed")
print("Train shape:", X_train.shape)
print("Test shape: ", X_test.shape)
print("Validation shape:", X_valid.shape)
missing = X_train.isnull().sum()
print("\nMissing values in train set:\n", missing[missing > 0] if missing.sum() > 0 else "(none)")
print("\nfirst 3 rows:\n", X_train.head(3).to_string())