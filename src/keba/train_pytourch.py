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
# CONFIG
# =========================================================
TARGET_COL = "y_b"
N_STEPS = 24


# =========================================================
# 1. SEQUENCE CREATION (SAFE VERSION)
# =========================================================
def prepare_lstm_data(df, target_col, n_steps):
    df = df.copy()

    # only sort if time exists
    if 'ds' in df.columns:
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
def train_model(model, loader, criterion, optimizer, device, epochs=5):
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
# 4. FORECASTING (REAL FIX — NO LABELS NEEDED)
# =========================================================
def forecast_future(model, df, scaler, target_col, n_steps, device, steps_ahead=24):
    df = df.copy()

    # scale
    values = scaler.transform(df[[target_col]])

    # last known window
    window = values[-n_steps:].reshape(1, n_steps, 1)

    input_seq = torch.tensor(window, dtype=torch.float32).to(device)

    preds = []

    model.eval()

    for _ in range(steps_ahead):
        with torch.no_grad():
            pred = model(input_seq).cpu().numpy()[0]

        preds.append(pred)

        # slide window
        next_window = np.append(input_seq.cpu().numpy()[0][1:], [[pred]], axis=0)
        input_seq = torch.tensor(next_window.reshape(1, n_steps, 1), dtype=torch.float32).to(device)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1))


# =========================================================
# 5. PLOTTING
# =========================================================
def plot_series(values, title="Forecast"):
    plt.figure(figsize=(12, 5))
    plt.plot(values, label="Prediction")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(title.replace(" ", "_").lower() + ".png")
    plt.close()


# =========================================================
# 6. MAIN PIPELINE
# =========================================================
if __name__ == "__main__":

    # -------------------------
    # LOAD TRAIN DATA
    # -------------------------
    df = pd.read_csv("data/level_1_train.csv", sep=";")

    # -------------------------
    # SCALE
    # -------------------------
    scaler = MinMaxScaler()
    df[TARGET_COL] = scaler.fit_transform(df[[TARGET_COL]])

    # -------------------------
    # DATASET
    # -------------------------
    X, y = prepare_lstm_data(df, TARGET_COL, N_STEPS)

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

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
    train_model(model, train_loader, criterion, optimizer, device, epochs=5)

    # -------------------------
    # FORECAST (THIS FIXES YOUR ERROR COMPLETELY)
    # -------------------------
    new_df = pd.read_csv("data/level_1_test_submission.csv", sep=";")

    preds_new = forecast_future(
        model=model,
        df=df,                      # use TRAIN history, NOT submission file
        scaler=scaler,
        target_col=TARGET_COL,
        n_steps=N_STEPS,
        device=device,
        steps_ahead=24
    )

    print("\nForecast completed.")

    plot_series(preds_new, "Future Forecast")