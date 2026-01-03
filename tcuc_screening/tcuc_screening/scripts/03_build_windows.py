#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit mean/std over (time, features) array x of shape (T, F)."""
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu, sigma


def zscore_apply(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x - mu) / sigma


def build_windows(arr: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    arr: (T, F)
    Returns:
      hist: (S, lookback, F)   using t0..t0+lookback-1
      fut:  (S, horizon,  F)   using t0+lookback..t0+lookback+horizon-1
    """
    T, F = arr.shape
    S = T - lookback - horizon + 1
    if S <= 0:
        raise ValueError(
            f"Not enough data: T={T}, lookback={lookback}, horizon={horizon}")

    hist = np.zeros((S, lookback, F), dtype=np.float32)
    fut = np.zeros((S, horizon, F), dtype=np.float32)

    for s in range(S):
        t0 = s
        hist[s] = arr[t0: t0 + lookback, :]
        fut[s] = arr[t0 + lookback: t0 + lookback + horizon, :]

    return hist, fut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data_processed/rts96_full",
                    help="Directory that contains demands.parquet and loadings.parquet")
    ap.add_argument("--out_dir", type=str, default="datasets/rts96_lstm_v1",
                    help="Where to write windowed dataset")
    ap.add_argument("--lookback", type=int, default=192)
    ap.add_argument("--horizon", type=int, default=24)

    # Split control (chronological)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    # test_frac implied = 1 - train - val

    ap.add_argument("--visual_check", action="store_true",
                    help="Print one sample summary (shapes + min/max)")

    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    processed_dir = root / args.processed_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    demands_path = processed_dir / "demands.parquet"
    loadings_path = processed_dir / "loadings.parquet"
    if not demands_path.exists() or not loadings_path.exists():
        raise FileNotFoundError(
            f"Missing demands/loadings parquet in {processed_dir}")

    dem_df = pd.read_parquet(demands_path)
    load_df = pd.read_parquet(loadings_path)

    # Identify columns
    demand_cols = [c for c in dem_df.columns if c.startswith("d_")]
    line_cols = [c for c in load_df.columns if c.startswith("l_")]

    if len(demand_cols) != 73:
        print(f"WARNING: expected 73 demand columns, got {len(demand_cols)}")
    if len(line_cols) != 120:
        print(f"WARNING: expected 120 line columns, got {len(line_cols)}")

    # Align by time index t (defensive)
    if "t" in dem_df.columns and "t" in load_df.columns:
        merged = dem_df[[
            "t"] + demand_cols].merge(load_df[["t"] + line_cols], on="t", how="inner")
        merged = merged.sort_values("t").reset_index(drop=True)
        X_raw = merged[demand_cols].to_numpy(dtype=np.float32)   # (T, 73)
        Y_raw = merged[line_cols].to_numpy(dtype=np.float32)     # (T, 120)
        t_index = merged["t"].to_numpy()
    else:
        # assume already aligned row-wise
        T = min(len(dem_df), len(load_df))
        X_raw = dem_df[demand_cols].iloc[:T].to_numpy(dtype=np.float32)
        Y_raw = load_df[line_cols].iloc[:T].to_numpy(dtype=np.float32)
        t_index = np.arange(T)

    T = X_raw.shape[0]
    lookback = args.lookback
    horizon = args.horizon
    S = T - lookback - horizon + 1
    if S <= 0:
        raise ValueError(
            f"Not enough hours T={T} for lookback={lookback}, horizon={horizon}")

    # Split by samples (not hours) to preserve window integrity
    train_S = int(S * args.train_frac)
    val_S = int(S * args.val_frac)
    test_S = S - train_S - val_S
    if train_S <= 0 or val_S <= 0 or test_S <= 0:
        raise ValueError(
            f"Bad split: S={S}, train={train_S}, val={val_S}, test={test_S}")

    # Fit demand normalization on TRAIN portion only.
    # Training windows cover hours [0 .. train_S + lookback - 1]
    train_hours_end = train_S + lookback  # exclusive
    mu, sigma = zscore_fit(X_raw[:train_hours_end, :])
    X_norm = zscore_apply(X_raw, mu, sigma).astype(np.float32)

    # Build windows
    # we only need hist for X
    X_hist, _ = build_windows(X_norm, lookback, horizon)
    Y_hist, Y_fut = build_windows(Y_raw, lookback, horizon)

    # Compose XY
    XY_hist = np.concatenate([X_hist, Y_hist], axis=2)  # (S, lookback, 193)

    # Target is future loadings (Ŷ)
    Y_hat = Y_fut  # (S, horizon, 120)

    # Checks
    def _assert_finite(name, a):
        if not np.isfinite(a).all():
            bad = np.where(~np.isfinite(a))
            raise RuntimeError(
                f"{name} contains non-finite values at indices like {tuple(x[0] for x in bad)}")

    _assert_finite("X_hist", X_hist)
    _assert_finite("Y_hist", Y_hist)
    _assert_finite("XY_hist", XY_hist)
    _assert_finite("Y_hat", Y_hat)

    assert X_hist.shape == (S, lookback, X_raw.shape[1])
    assert Y_hist.shape == (S, lookback, Y_raw.shape[1])
    assert XY_hist.shape == (S, lookback, X_raw.shape[1] + Y_raw.shape[1])
    assert Y_hat.shape == (S, horizon, Y_raw.shape[1])

    # Save as NPZ (simple, fast, framework-agnostic)
    np.savez_compressed(
        out_dir / "dataset_windows.npz",
        X=X_hist,
        Yhist=Y_hist,
        XY=XY_hist,
        Yhat=Y_hat,
        t_index=t_index,
        demand_cols=np.array(demand_cols),
        line_cols=np.array(line_cols),
        mu=mu.astype(np.float32),
        sigma=sigma.astype(np.float32),
        splits=np.array([train_S, val_S, test_S], dtype=np.int64),
    )

    meta = {
        "processed_dir": str(processed_dir),
        "T_hours": int(T),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "S_samples": int(S),
        "features_X": int(X_raw.shape[1]),
        "features_Y": int(Y_raw.shape[1]),
        "features_XY": int(X_raw.shape[1] + Y_raw.shape[1]),
        "splits_samples": {"train": int(train_S), "val": int(val_S), "test": int(test_S)},
        "normalization": {"X": "zscore_fit_on_train_hours_only", "Y": "none"},
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("OK")
    print("Saved:", out_dir / "dataset_windows.npz")
    print("Shapes:")
    print("  X   :", X_hist.shape)
    print("  Yhist:", Y_hist.shape)
    print("  XY  :", XY_hist.shape)
    print("  Yhat:", Y_hat.shape)
    print("Splits (samples):", train_S, val_S, test_S)

    if args.visual_check:
        s = np.random.randint(0, S)
        print("\nVISUAL CHECK (one random sample)")
        print(" sample idx:", s)
        print(" X[min,max]:", float(X_hist[s].min()), float(X_hist[s].max()))
        print(" Yhist[min,max]:", float(
            Y_hist[s].min()), float(Y_hist[s].max()))
        print(" Yhat[min,max]:", float(Y_hat[s].min()), float(Y_hat[s].max()))
        print(" Yhat first-hour max loading:", float(Y_hat[s, 0, :].max()))


if __name__ == "__main__":
    main()
