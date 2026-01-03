#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tcuc_screening.models.predictors import PredictorSpec, build_model

# tqdm (safe fallback if not installed)
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()  # (S, 192, F)
        self.Y = torch.from_numpy(Y).float()  # (S, 24, L)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    Yhist = data["Yhist"]
    XY = data["XY"]
    Yhat = data["Yhat"]
    splits = data["splits"]
    return X, Yhist, XY, Yhat, splits


def _iter_loader(loader, desc: str):
    """tqdm wrapper that degrades gracefully if tqdm is unavailable."""
    if tqdm is None:
        return loader
    return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)


def run_epoch(model, loader, optimizer, device, train: bool, *, epoch: int):
    mse = nn.MSELoss()
    model.train() if train else model.eval()

    total = 0.0
    n = 0

    phase = "train" if train else "val"
    it = _iter_loader(loader, desc=f"epoch {epoch:02d} | {phase}")

    for Xb, Yb in it:
        Xb = Xb.to(device, non_blocking=True)
        Yb = Yb.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(Xb)
        loss = mse(pred, Yb)

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = int(Xb.size(0))
        total += float(loss.item()) * bs
        n += bs

        if tqdm is not None and hasattr(it, "set_postfix"):
            it.set_postfix(loss=float(loss.item()))

    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Streaming evaluation to avoid torch.cat over the full dataset
    and to avoid large reductions on MPS that can yield inf/NaN.
    We compute SSE on CPU in float32 for numerical stability.
    """
    model.eval()

    sse = 0.0
    count = 0

    H = None
    sse_h = None
    count_h = None

    for Xb, Yb in loader:
        Xb = Xb.to(device, non_blocking=True)
        Yb = Yb.to(device, non_blocking=True)

        pred = model(Xb)

        # stable reductions on CPU
        pred_cpu = pred.detach().float().cpu()
        y_cpu = Yb.detach().float().cpu()
        diff = pred_cpu - y_cpu

        sse += float((diff * diff).sum().item())
        count += diff.numel()

        if H is None:
            H = int(y_cpu.shape[1])
            sse_h = [0.0] * H
            count_h = [0] * H

        # horizon-wise SSE (kept simple and robust)
        for h in range(H):
            dh = diff[:, h, :]
            sse_h[h] += float((dh * dh).sum().item())
            count_h[h] += dh.numel()

    rmse_all = (sse / max(count, 1)) ** 0.5
    rmse_h = [(sse_h[h] / max(count_h[h], 1)) **
              0.5 for h in range(H)]  # type: ignore[index]

    return {"rmse": rmse_all, "rmse_by_horizon": rmse_h}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default="datasets/rts96_lstm_v1/dataset_windows.npz",
    )
    ap.add_argument("--mode", type=str,
                    choices=["X", "Y", "XY"], required=True)
    ap.add_argument(
        "--model",
        type=str,
        choices=["lstm", "transformer", "gcn_lstm"],
        required=True,
    )

    # arch naming:
    #   lstm: paper_arch
    #   transformer: tfm_d256_h8_l4_ff512 (optionally add _mean)
    #   gcn_lstm: gcnlstm_h500 (or any string; used as folder name)
    ap.add_argument("--arch", type=str, default=None)

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

    S = int(Xin.shape[0])
    assert S == train_S + val_S + test_S

    # chronological splits
    X_train, Y_train = Xin[:train_S], Yhat[:train_S]
    X_val, Y_val = Xin[train_S: train_S +
                       val_S], Yhat[train_S: train_S + val_S]
    X_test, Y_test = Xin[train_S + val_S:], Yhat[train_S + val_S:]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # pin_memory only helps on CUDA; also keep num_workers modest unless tuned
    pin = bool(device.type == "cuda")

    train_loader = DataLoader(
        WindowDataset(X_train, Y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=bool(args.num_workers > 0),
    )
    val_loader = DataLoader(
        WindowDataset(X_val, Y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=bool(args.num_workers > 0),
    )
    test_loader = DataLoader(
        WindowDataset(X_test, Y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=bool(args.num_workers > 0),
    )

    input_dim = int(Xin.shape[2])

    # default arch
    arch = args.arch
    if arch is None:
        if args.model == "lstm":
            arch = "paper_arch"
        elif args.model == "transformer":
            arch = "tfm_d256_h8_l4_ff512"
        else:
            arch = "gcnlstm_h500"

    # For gcn_lstm we must provide repo_root so adjacency can be loaded.
    repo_root = str(root) if args.model == "gcn_lstm" else None

    spec = PredictorSpec(
        model=args.model,  # type: ignore[arg-type]
        input_dim=input_dim,
        arch=arch,
        repo_root=repo_root,
        L=int(Yhat.shape[2]),
        horizon=int(Yhat.shape[1]),
    )

    model = build_model(spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = root / "models" / args.mode / arch
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.json"

    print(
        f"START | mode={args.mode} model={args.model} arch={arch} "
        f"device={device} bs={args.batch_size} workers={args.num_workers}",
        flush=True,
    )

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_mse = run_epoch(model, train_loader, optimizer,
                           device, train=True, epoch=epoch)
        t1 = time.time()
        va_mse = run_epoch(model, val_loader, optimizer,
                           device, train=False, epoch=epoch)
        t2 = time.time()

        print(
            f"Epoch {epoch:02d} | train_mse={tr_mse:.6f} ({t1-t0:.1f}s) | "
            f"val_mse={va_mse:.6f} ({t2-t1:.1f}s)",
            flush=True,
        )

        if va_mse < best_val - 1e-6:
            best_val = va_mse
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mode": args.mode,
                    "model": args.model,
                    "arch": arch,
                    "input_dim": input_dim,
                    "spec": {
                        "model": args.model,
                        "arch": arch,
                        "input_dim": input_dim,
                        "repo_root": repo_root,
                        "L": int(Yhat.shape[2]),
                        "horizon": int(Yhat.shape[1]),
                    },
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"Early stopping (patience={args.patience}).", flush=True)
                break

    # Evaluate best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    results = {
        "mode": args.mode,
        "model": args.model,
        "arch": arch,
        "device": str(device),
        "splits": {"train": train_S, "val": val_S, "test": test_S},
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            "num_workers": args.num_workers,
        },
        "best_val_mse": best_val,
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nRESULTS SAVED", flush=True)
    print("Best model:", best_path, flush=True)
    print("Metrics:", metrics_path, flush=True)
    print("Test RMSE:", test_metrics["rmse"], flush=True)


if __name__ == "__main__":
    main()
