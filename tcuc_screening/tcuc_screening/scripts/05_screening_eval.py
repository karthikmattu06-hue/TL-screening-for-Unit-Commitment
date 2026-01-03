#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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
    Yhat = data["Yhat"]    # (S, 24, 120) ground truth future loadings
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

        self.lstm1 = nn.LSTM(input_size=input_dim,
                             hidden_size=1000, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=1000, hidden_size=500,
                             num_layers=1, batch_first=True)

        self.fc1 = nn.Linear(500, 3000)
        self.fc2 = nn.Linear(3000, 1000)
        self.fc3 = nn.Linear(1000, 3000)
        self.out = nn.Linear(3000, 24 * 120)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        o1, _ = self.lstm1(x)          # (B, 192, 1000)
        o2, _ = self.lstm2(o1)         # (B, 192, 500)
        h = o2[:, -1, :]               # (B, 500)

        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))

        y = self.sigmoid(self.out(h))  # (B, 2880) in [0,1]
        return y.view(-1, 24, 120)


# -----------------------------
# Inference + Metrics
# -----------------------------

@torch.no_grad()
def infer_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y_true: (N, 24, 120) float
      y_pred: (N, 24, 120) float
    """
    model.eval()
    y_trues = []
    y_preds = []

    for Xb, Yb in loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        pred = model(Xb)
        y_trues.append(Yb.cpu().numpy())
        y_preds.append(pred.cpu().numpy())

    y_true = np.concatenate(y_trues, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    return y_true, y_pred


def confusion_counts(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> dict:
    """
    Inputs: boolean arrays of same shape
    Returns TP, FP, TN, FN as ints.
    """
    tp = int(np.logical_and(y_pred_bin, y_true_bin).sum())
    fp = int(np.logical_and(y_pred_bin, np.logical_not(y_true_bin)).sum())
    tn = int(np.logical_and(np.logical_not(y_pred_bin),
             np.logical_not(y_true_bin)).sum())
    fn = int(np.logical_and(np.logical_not(y_pred_bin), y_true_bin).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else float("nan")


def metrics_at_threshold(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> dict:
    """
    y_true, y_pred: (N, 24, 120) floats in [0,1]
    Evaluate across ALL (sample, hour, line) pairs.
    """
    true_bin = (y_true >= tau)
    pred_bin = (y_pred >= tau)

    counts = confusion_counts(true_bin, pred_bin)
    TP, FP, TN, FN = counts["TP"], counts["FP"], counts["TN"], counts["FN"]
    total = TP + FP + TN + FN

    # fraction truly positive
    prevalence = safe_div(TP + FN, total)
    accuracy = safe_div(TP + TN, total)
    recall = safe_div(TP, TP + FN)                     # TPR
    fnr = safe_div(FN, TP + FN)                        # 1 - recall
    precision = safe_div(TP, TP + FP)                  # PPV
    kept_rate = safe_div(TP + FP, total)               # predicted positives
    reduction = 1.0 - kept_rate

    return {
        "threshold": float(tau),
        "prevalence": float(prevalence),
        "accuracy": float(accuracy),
        "recall": float(recall),
        "fnr": float(fnr),
        "precision": float(precision),
        "kept_rate": float(kept_rate),
        "reduction": float(reduction),
        **counts,
        "total_pairs": int(total),
    }


def load_model_checkpoint(model_path: Path, input_dim: int, device: torch.device) -> nn.Module:
    ckpt = torch.load(model_path, map_location=device)
    model = PaperLSTM(input_dim=input_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def print_accuracy_table(rows: list[dict]):
    """
    Print Table II style: threshold vs accuracy for X,Y,XY.
    """
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="threshold", columns="mode",
                     values="accuracy").sort_index()
    print("\nTABLE II REPLICATION (Accuracy)")
    print(pivot.to_string(float_format=lambda x: f"{x*100:6.3f}%"))


def print_companion_table(rows: list[dict]):
    """
    Print threshold vs key metrics per mode.
    """
    df = pd.DataFrame(rows).sort_values(
        ["mode", "threshold"]).reset_index(drop=True)

    # Keep a compact set for console readability
    keep_cols = ["mode", "threshold", "prevalence", "accuracy",
                 "recall", "precision", "fnr", "kept_rate", "reduction"]
    df2 = df[keep_cols].copy()

    def pct(x): return f"{100*x:6.3f}%"
    for c in ["prevalence", "accuracy", "recall", "precision", "fnr", "kept_rate", "reduction"]:
        df2[c] = df2[c].apply(lambda v: pct(
            v) if np.isfinite(v) else "   nan ")

    print("\nCOMPANION METRICS (all (sample,hour,line) pairs)")
    print(df2.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str,
                    default="datasets/rts96_lstm_v1/dataset_windows.npz")
    ap.add_argument("--arch", type=str, default="paper_arch",
                    help="Subfolder under models/<mode>/")
    ap.add_argument("--modes", type=str, default="X,Y,XY",
                    help="Comma-separated: X,Y,XY")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_json", type=str,
                    default="results/screening_eval_paper_arch.json")
    ap.add_argument("--out_csv", type=str,
                    default="results/screening_eval_paper_arch.csv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    npz_path = root / args.dataset

    X, Yhist, XY, Yhat, splits = load_npz(npz_path)
    train_S, val_S, test_S = [int(x) for x in splits.tolist()]

    # Test slice
    start = train_S + val_S
    end = start + test_S
    Y_test = Yhat[start:end]  # ground truth

    device = torch.device("cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"))
    pin = (device.type == "cuda")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    thresholds = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]  # 0.1..0.9

    all_rows = []

    (root / "results").mkdir(parents=True, exist_ok=True)

    for mode in modes:
        if mode == "X":
            Xin = X[start:end]
        elif mode == "Y":
            Xin = Yhist[start:end]
        elif mode == "XY":
            Xin = XY[start:end]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        input_dim = Xin.shape[2]
        ds = WindowDataset(Xin, Y_test)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

        model_path = root / "models" / mode / args.arch / "best.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

        model = load_model_checkpoint(
            model_path, input_dim=input_dim, device=device)
        y_true, y_pred = infer_predictions(model, loader, device)

        # Evaluate thresholds
        for tau in thresholds:
            met = metrics_at_threshold(y_true=y_true, y_pred=y_pred, tau=tau)
            met["mode"] = mode
            all_rows.append(met)

        print(
            f"\nDONE: mode={mode} | device={device} | test_samples={len(ds)}")

    # Print tables
    print_accuracy_table(all_rows)
    print_companion_table(all_rows)

    # Save outputs
    out_json = root / args.out_json
    out_csv = root / args.out_csv
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w") as f:
        json.dump(all_rows, f, indent=2)

    pd.DataFrame(all_rows).to_csv(out_csv, index=False)

    print("\nSAVED")
    print(" JSON:", out_json)
    print(" CSV :", out_csv)


if __name__ == "__main__":
    main()
