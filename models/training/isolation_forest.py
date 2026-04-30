import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

# Load isolation forest data
X_train = pd.read_csv("feature_engineering/X_train_isolation.csv")
X_test = pd.read_csv("feature_engineering/X_test_isolation.csv")
X_Valid = pd.read_csv("feature_engineering/X_valid_isolation.csv")

# Training Starts - defining parameters
n_estimators = 200
contamination = 0.05
random_state = 42

isolation_forest = IsolationForest(
    n_estimators=n_estimators,
    contamination=contamination,
    random_state=random_state,
    n_jobs=-1)
isolation_forest.fit(X_train)
print("Training completed")

# Scoring function to get anomaly scores 
def score(df):
    raw = isolation_forest.score_samples(df) #higher value = norm transaction
    predictions = isolation_forest.predict(df)
    return -raw, (predictions == -1).astype(int) #higher value = more anomalous 

train_score, train_flags = score(X_train)
test_score, test_flags = score(X_test)
valid_score, valid_flags = score(X_Valid)

print("\nFlagged ANOMALY SCORES:")
print(f"Train set: {train_flags.sum(): >4} / {len(train_flags)} ({train_flags.mean(): .1%})")
print(f"Test set:  {test_flags.sum(): >4} / {len(test_flags)} ({test_flags.mean(): .1%})")
print(f"Valid set: {valid_flags.sum(): >4} / {len(valid_flags)} ({valid_flags.mean(): .1%})")

# Save model and results
os.makedirs("models/training", exist_ok=True)
os.makedirs("models/saved_models", exist_ok=True)
joblib.dump(isolation_forest, "models/saved_models/isolation_forest.pkl")

results = pd.concat([
    pd.DataFrame({"split": "train", "anomaly_score": train_score, "flag": train_flags}),
    pd.DataFrame({"split": "test", "anomaly_score": test_score, "flag": test_flags}),
    pd.DataFrame({"split": "valid", "anomaly_score": valid_score, "flag": valid_flags})], ignore_index=True)
results.to_csv("models/training/isolation_forest_results.csv", index=False)

test_show = X_test.copy().reset_index(drop=True)
test_show["anomaly_score"] = test_score
inspect_columns = ["TransactionAmount", "TransactionDuration", "AccountBalance", "LoginAttempts",
                "amount_balance_ratio", "anomaly_score", "high_login_attempts"]

print("\nTop 10 anomalies in test set with highest anomaly scores:")
print(test_show.nlargest(10, "anomaly_score")[inspect_columns].round(3).to_string())

import matplotlib.pyplot as plt

plt.hist(train_score, bins=50)
plt.title("Anomaly Score Distribution")
plt.show()