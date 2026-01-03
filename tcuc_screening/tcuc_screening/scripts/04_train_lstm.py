#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Dataset
# -----------------------------

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()   # (S, 192, F)
        self.Y = torch.from_numpy(Y).float()   # (S, 24, 120)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]          # (S, 192, 73)
    Yhist = data["Yhist"]  # (S, 192, 120)
    XY = data["XY"]        # (S, 192, 193)
    Yhat = data["Yhat"]    # (S, 24, 120)
    splits = data["splits"]  # [train_S, val_S, test_S]
    return X, Yhist, XY, Yhat, splits


# -----------------------------
# Paper-faithful Model
# -----------------------------

class PaperLSTM(nn.Module):
    """
    Matches paper Fig. 1:

      LSTM-1: 1000 units
      LSTM-2: 500 units
      FC-1 : 3000 ReLU
      FC-2 : 1000 ReLU
      FC-3 : 3000 ReLU
      O/P  : 2880 Sigmoid  (L*24, with L=120)

    Output reshaped to (24, 120).
    """

    def __init__(self, input_dim: int):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=1000,
            num_layers=1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=1000,
            hidden_size=500,
            num_layers=1,
            batch_first=True,
        )

        self.fc1 = nn.Linear(500, 3000)
        self.fc2 = nn.Linear(3000, 1000)
        self.fc3 = nn.Linear(1000, 3000)
        self.out = nn.Linear(3000, 24 * 120)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, 192, F)
        o1, _ = self.lstm1(x)          # (B, 192, 1000)
        o2, _ = self.lstm2(o1)         # (B, 192, 500)
        h = o2[:, -1, :]               # last time step -> (B, 500)

        h = self.relu(self.fc1(h))     # (B, 3000)
        h = self.relu(self.fc2(h))     # (B, 1000)
        h = self.relu(self.fc3(h))     # (B, 3000)

        y = self.sigmoid(self.out(h))  # (B, 2880) in [0,1]
        return y.view(-1, 24, 120)


# -----------------------------
# Metrics / Baselines
# -----------------------------

@torch.no_grad()
def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()


@torch.no_grad()
def rmse_by_horizon(y_pred: torch.Tensor, y_true: torch.Tensor) -> list[float]:
    return [
        torch.sqrt(torch.mean((y_pred[:, h, :] - y_true[:, h, :]) ** 2)).item()
        for h in range(y_true.shape[1])
    ]


@torch.no_grad()
def persistence_baseline(X_input: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Predict next 24h line loadings as last observed loading repeated.
    Only defined for modes with Y in input.
    """
    if mode == "Y":
        last = X_input[:, -1, :]          # (B,120)
    elif mode == "XY":
        last = X_input[:, -1, 73:]        # (B,120)
    else:
        raise ValueError("Persistence baseline requires mode Y or XY")

    return last.unsqueeze(1).repeat(1, 24, 1)


# -----------------------------
# Train / Eval
# -----------------------------

def run_epoch(model, loader, optimizer, device, train: bool):
    mse = nn.MSELoss()
    model.train() if train else model.eval()

    total = 0.0
    n = 0

    for Xb, Yb in loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)

        if train:
            optimizer.zero_grad()

        pred = model(Xb)
        loss = mse(pred, Yb)

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total += loss.item() * Xb.size(0)
        n += Xb.size(0)

    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for Xb, Yb in loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        preds.append(model(Xb))
        trues.append(Yb)
    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(trues, dim=0)
    return {
        "rmse": rmse(y_pred, y_true),
        "rmse_by_horizon": rmse_by_horizon(y_pred, y_true),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str,
                    default="datasets/rts96_lstm_v1/dataset_windows.npz")
    ap.add_argument("--mode", type=str,
                    choices=["X", "Y", "XY"], required=True)

    # Training hyperparameters (keep reasonable defaults; not “tuning” unless you sweep)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=0)

    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    npz_path = root / args.dataset

    X, Yhist, XY, Yhat, splits = load_npz(npz_path)
    train_S, val_S, test_S = [int(x) for x in splits.tolist()]

    if args.mode == "X":
        Xin = X
    elif args.mode == "Y":
        Xin = Yhist
    else:
        Xin = XY

    S = Xin.shape[0]
    assert S == train_S + val_S + test_S

    # Chronological splits
    X_train, Y_train = Xin[:train_S], Yhat[:train_S]
    X_val, Y_val = Xin[train_S:train_S + val_S], Yhat[train_S:train_S + val_S]
    X_test, Y_test = Xin[train_S + val_S:], Yhat[train_S + val_S:]

    train_loader = DataLoader(WindowDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(WindowDataset(X_val, Y_val), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(WindowDataset(X_test, Y_test), batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = Xin.shape[2]
    model = PaperLSTM(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = root / "models" / args.mode / "paper_arch"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.json"

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        tr_mse = run_epoch(model, train_loader, optimizer, device, train=True)
        va_mse = run_epoch(model, val_loader, optimizer, device, train=False)
        print(
            f"Epoch {epoch:02d} | train_mse={tr_mse:.6f} | val_mse={va_mse:.6f}")

        if va_mse < best_val - 1e-6:
            best_val = va_mse
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(),
                       "mode": args.mode, "input_dim": input_dim}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (patience={args.patience}).")
                break

    # Load best and evaluate
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    persistence_metrics = None
    if args.mode in ("Y", "XY"):
        all_preds, all_trues = [], []
        for Xb, Yb in test_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            all_preds.append(persistence_baseline(Xb, args.mode))
            all_trues.append(Yb)
        p_pred = torch.cat(all_preds, dim=0)
        p_true = torch.cat(all_trues, dim=0)
        persistence_metrics = {
            "rmse": rmse(p_pred, p_true),
            "rmse_by_horizon": rmse_by_horizon(p_pred, p_true),
        }

    results = {
        "mode": args.mode,
        "device": str(device),
        "splits": {"train": train_S, "val": val_S, "test": test_S},
        "arch": "paper_exact",
        "training": {"batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr, "patience": args.patience},
        "best_val_mse": best_val,
        "val": val_metrics,
        "test": test_metrics,
        "persistence_test": persistence_metrics,
    }

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nRESULTS SAVED")
    print("Best model:", best_path)
    print("Metrics:", metrics_path)
    print("Test RMSE:", test_metrics["rmse"])
    if persistence_metrics is not None:
        print("Persistence Test RMSE:", persistence_metrics["rmse"])


if __name__ == "__main__":
    main()
