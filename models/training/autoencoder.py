#This file loads the preprocessed data, creates autoencoder architecture, trains autoencoder model,
#calculates reconstruction error, flags anomalies and saves the results. 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

HIDDEN_DIM = 16
LATENT_DIM = 8
EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 15
THRESHOLD_PERCENTILE = 95

# Load the preprocessed + engineered data
X_train = pd.read_csv("feature_engineering/X_train_autoen.csv")
X_test = pd.read_csv("feature_engineering/X_test_autoen.csv")
X_Valid = pd.read_csv("feature_engineering/X_valid_autoen.csv")

INPUT_DIM = X_train.shape[1]

for name, df in [("train", X_train), ("valid", X_Valid), ("test", X_test)]:
    bad = df.select_dtypes(include='object').columns.tolist()
    if bad:
        print(f"  {name} has non-numeric columns: {bad}")
        for col in bad:
            print(f"    {col} sample: {df[col].unique()[:5]}")

X_train = X_train.apply(pd.to_numeric, errors='raise').astype('float32')
X_Valid = X_Valid.apply(pd.to_numeric, errors='raise').astype('float32')
X_test  = X_test.apply(pd.to_numeric, errors='raise').astype('float32')

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
X_Valid_tensor = torch.tensor(X_Valid.values, dtype=torch.float32)

#Create DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor),batch_size=32,
                          shuffle=True, generator=torch.Generator().manual_seed(SEED))

#Creating an Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print("Training...")
print(f" architecture: {INPUT_DIM} -> {HIDDEN_DIM} -> {LATENT_DIM} -> {HIDDEN_DIM} -> {INPUT_DIM}")
n_parameters = sum(p.numel() for p in model.parameters())
print(f" parameters: {n_parameters} samples: {X_train_tensor.shape[0]}")

best_val_loss = float("inf")
best_state = None
patience_counter = 0
history = {"train": [], "val": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for batch_data in train_loader:
        batch = batch_data[0]
        optimizer.zero_grad()
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.size(0)
    epoch_loss /= len(X_train_tensor)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_Valid_tensor), X_Valid_tensor).item()

    history["train"].append(epoch_loss)
    history["val"].append(val_loss)

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 10 == 0 or epoch == 1:
        marker = "*" if patience_counter == 0 and epoch > 1 else ""
        print(f" epoch {epoch:>3}: train: {epoch_loss:.6f} val: {val_loss:.6f} {marker}")

    if patience_counter >= PATIENCE:
        print(f" early stopping at epoch {epoch} (best val = {best_val_loss:.6f})")
        break

model.load_state_dict(best_state)

def reconstruction_error(X_tensor):
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)
    return ((X_tensor - reconstructed) ** 2).mean(dim=1).numpy()   

train_errors = reconstruction_error(X_train_tensor)
test_errors = reconstruction_error(X_test_tensor)
valid_errors = reconstruction_error(X_Valid_tensor)

threshold = float(np.percentile(valid_errors, THRESHOLD_PERCENTILE))
print(f"\nReconstruction Error Threshold (P{THRESHOLD_PERCENTILE})): {threshold:.6f}")

train_flag = (train_errors > threshold).astype(int)
test_flag = (test_errors > threshold).astype(int)
valid_flag = (valid_errors > threshold).astype(int)


#Save model and results
os.makedirs("models/saved_models", exist_ok=True)
os.makedirs("models/training", exist_ok=True)

torch.save(model.state_dict(), "models/saved_models/autoencoder.pth")
joblib.dump({
    "input_dim": INPUT_DIM,
    "hidden_dim": HIDDEN_DIM,
    "latent_dim": LATENT_DIM,
    "threshold": threshold,
    "feature_columns": list(X_train.columns)}, "models/saved_models/autoencoder_metadata.pkl")

results = pd.concat([
    pd.DataFrame({
        "split": "train","reconstruction_error": train_errors, "flag": train_flag}),
    pd.DataFrame({
        "split": "test", "reconstruction_error": test_errors, "flag": test_flag}),
    pd.DataFrame({
        "split": "valid", "reconstruction_error": valid_errors, "flag": valid_flag})
], ignore_index=True)
results.to_csv("models/training/autoencoder_results.csv", index=False)

pd.DataFrame(history).to_csv("models/training/autoencoder_history.csv", index=False)

X_test_raw = pd.read_csv("feature_engineering/X_test_isolation.csv").reset_index(drop=True)

X_test_raw["reconstruction_error"] = test_errors
inspect_columns = ["TransactionAmount", "AccountBalance", "LoginAttempts", "TransactionDuration",
                   "amount_balance_ratio", "high_login_attempts", "reconstruction_error"]
print ("\nTop 10 anomalies of Test Set with highest Reconstruction Errors:")
print(X_test_raw.nlargest(10, "reconstruction_error")[inspect_columns].round(3).to_string())