import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt


# =========================================================
# CONFIG (IMPORTANT: ONE SOURCE OF TRUTH)
# =========================================================
TARGET_COL = "y_b"
N_STEPS = 24


# =========================================================
# 1. SEQUENCE CREATION
# =========================================================
def prepare_lstm_data(df, target_col, n_steps):
    df = df.copy()

    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')

    data = df[target_col].values

    X, y = [], []

    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y


# =========================================================
# 2. MODEL
# =========================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()


# =========================================================
# 3. TRAINING
# =========================================================
def train_model(model, loader, criterion, optimizer, device, epochs=2):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.6f}")


# =========================================================
# 4. TEST PREDICTION (WITH LABELS)
# =========================================================
def predict_with_labels(model, df, scaler, target_col, n_steps, device):
    df = df.copy()

    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')

    df[target_col] = scaler.transform(df[[target_col]])

    X, y = prepare_lstm_data(df, target_col, n_steps)

    X = torch.tensor(X, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(X).cpu().numpy()

    preds = scaler.inverse_transform(preds.reshape(-1, 1))
    y = scaler.inverse_transform(y.reshape(-1, 1))

    return preds, y


# =========================================================
# 5. INFERENCE (NO LABELS)
# =========================================================
def predict_only(model, df, scaler, target_col, n_steps, device):
    df = df.copy()

    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')

    # ⚠️ must use SAME column as training
    if target_col not in df.columns:
        raise ValueError(
            f"Missing column '{target_col}' in inference data."
        )

    values = scaler.transform(df[[target_col]])

    temp_df = pd.DataFrame(values, columns=[target_col])

    X, _ = prepare_lstm_data(temp_df, target_col, n_steps)

    X = torch.tensor(X, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(X).cpu().numpy()

    return scaler.inverse_transform(preds.reshape(-1, 1))


# =========================================================
# 6. PLOTTING
# =========================================================
def plot_results(true, preds, title="Prediction"):
    plt.figure(figsize=(12, 5))
    plt.plot(true, label="True")
    plt.plot(preds, label="Predicted")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(title.replace(" ", "_").lower() + ".png")
    plt.close()


# =========================================================
# 7. MAIN
# =========================================================
if __name__ == "__main__":

    # -------------------------
    # LOAD TRAIN DATA
    # -------------------------
    df = pd.read_csv("data/level_1_train.csv", sep=";")

    # -------------------------
    # SCALE (fit ONLY on train)
    # -------------------------
    scaler = MinMaxScaler()
    df[TARGET_COL] = scaler.fit_transform(df[[TARGET_COL]])

    # -------------------------
    # SEQUENCES
    # -------------------------
    X, y = prepare_lstm_data(df, TARGET_COL, N_STEPS)

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,
        shuffle=True
    )

    # -------------------------
    # MODEL
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMModel().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -------------------------
    # TRAIN
    # -------------------------
    train_model(model, train_loader, criterion, optimizer, device, epochs=2)

    # -------------------------
    # TEST
    # -------------------------
    preds, true = predict_with_labels(
        model, df, scaler, TARGET_COL, N_STEPS, device
    )

    print("\n--- TEST RESULTS ---")
    print("RMSE:", np.sqrt(mean_squared_error(true, preds)))
    print("R2:", r2_score(true, preds))

    plot_results(true, preds, "Test Prediction")

    # -------------------------
    # INFERENCE DATA
    # -------------------------
    new_df = pd.read_csv("data/level_1_test_submission.csv", sep=";")

    preds_new = predict_only(
        model, new_df, scaler, TARGET_COL, N_STEPS, device
    )

    print("\nNew data prediction done.")

    plot_results(preds_new[:100], preds_new[:100], "Forecast Preview")