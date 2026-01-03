#!/usr/bin/env python3
# scripts/06_run_uc_eval.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tcuc_screening.models.predictors import load_model, predict_day
from tcuc_screening.uc_eval.constraint_generation import (
    active_lines_from_predicted_loadings,
)
from tcuc_screening.uc_eval.solve_full import solve_full_tcuc_24h
from tcuc_screening.uc_eval.solve_screened import (
    solve_screened_tcuc_24h,
    solve_screened_with_repair_tcuc_24h,
)
from tcuc_screening.uc_eval.metrics import compute_uc_eval_metrics



def load_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    Yhist = data["Yhist"]
    XY = data["XY"]
    Yhat = data["Yhat"]
    splits = data["splits"]
    meta = data["meta"].item() if "meta" in data.files else {}
    return X, Yhist, XY, Yhat, splits, meta


def build_ptdf_from_susceptance(
    buses_df: pd.DataFrame,
    branches_df: pd.DataFrame,
    slack_bus_id: int | None = None,
):
    bus_ids = buses_df["bus_id"].astype(int).tolist()
    N = len(bus_ids)
    bus_index = {b: i for i, b in enumerate(bus_ids)}
    if slack_bus_id is None:
        slack_bus_id = bus_ids[0]
    slack = bus_index[int(slack_bus_id)]

    from_bus = branches_df["from_bus"].astype(int).to_numpy()
    to_bus = branches_df["to_bus"].astype(int).to_numpy()
    b_line = branches_df["b"].astype(float).to_numpy()
    L = len(branches_df)

    B = np.zeros((N, N), dtype=float)
    for fb, tb, bb in zip(from_bus, to_bus, b_line):
        i = bus_index[int(fb)]
        j = bus_index[int(tb)]
        B[i, i] += bb
        B[j, j] += bb
        B[i, j] -= bb
        B[j, i] -= bb

    keep = [i for i in range(N) if i != slack]
    Bred = B[np.ix_(keep, keep)]

    PTDF = np.zeros((L, N), dtype=float)
    for ell in range(L):
        i = bus_index[int(from_bus[ell])]
        j = bus_index[int(to_bus[ell])]
        bb = float(b_line[ell])
        h = np.zeros(N, dtype=float)
        h[i] = bb
        h[j] = -bb

        y = np.linalg.solve(Bred, h[keep])
        PTDF[ell, keep] = y
        PTDF[ell, slack] = -PTDF[ell, keep].sum()

    return PTDF, bus_ids, slack_bus_id



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="datasets/rts96_lstm_v1/dataset_windows.npz")
    ap.add_argument("--mode", type=str, default="X", choices=["X", "Y", "XY"])
    ap.add_argument("--model", type=str, default="lstm", choices=["lstm","transformer"])
    ap.add_argument("--arch", type=str, default="paper_arch")
    ap.add_argument("--thresholds", type=str, default="0.5,0.7,0.9")
    ap.add_argument("--num_days", type=int, default=25, help="How many test samples (days) to evaluate")
    ap.add_argument("--solver", type=str, default="highs")
    ap.add_argument("--time_limit", type=float, default=None)
    ap.add_argument(
        "--processed_case",
        type=str,
        default="rts96_full",
        help="Folder under tcuc_screening/data_processed that contains demands.parquet etc.",
    )

    # Repair options
    ap.add_argument(
        "--repair",
        action="store_true",
        help="Iteratively add back violated removed lines and re-solve screened UC.",
    )
    ap.add_argument("--max_repair_rounds", type=int, default=5)
    ap.add_argument("--repair_tol", type=float, default=1e-9)

    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    # Load dataset windows
    X, Yhist, XY, Ytrue, splits, meta = load_npz(root / args.dataset)
    train_S, val_S, test_S = [int(x) for x in splits.tolist()]
    test_start = train_S + val_S

    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]

    # Load standardized power data
    raw_dir = root / "data_raw" / "oasys_ieee96"
    buses = pd.read_csv(raw_dir / "buses.csv")
    branches = pd.read_csv(raw_dir / "branches.csv")
    gens = pd.read_csv(raw_dir / "generators.csv")

    # Demands for ALL hours (needed to build 24h horizon demands)
    demands_parq = root / "data_processed" / args.processed_case / "demands.parquet"
    dem_all = pd.read_parquet(demands_parq)
    demand_cols = [c for c in dem_all.columns if c.startswith("d_")]
    demands_all_TN = dem_all[demand_cols].to_numpy(dtype=float)  # (T_total, N)

    # PTDF and arrays
    PTDF, bus_ids, slack = build_ptdf_from_susceptance(buses, branches)
    bus_index = {b: i for i, b in enumerate(bus_ids)}

    gens_bus = gens["bus_id"].astype(int).map(bus_index).to_numpy()
    p_min = gens["p_min"].astype(float).to_numpy()
    p_max = gens["p_max"].astype(float).to_numpy()
    cost_lin = gens["cost_lin"].astype(float).to_numpy()

    f_max = branches["f_max"].astype(float).to_numpy()

    # Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # Load model
    if args.mode == "X":
        input_dim = X.shape[2]
    elif args.mode == "Y":
        input_dim = Yhist.shape[2]
    else:
        input_dim = XY.shape[2]

    ckpt_path = root / "models" / args.mode / args.arch / "best.pt"
    model = load_model(
        str(ckpt_path),
        model=args.model,
        input_dim=input_dim,
        device=device,
        arch=args.arch,
        L=120,
        horizon=24,
    )

    lookback = 192
    horizon = 24

    # Evaluate first N test samples
    Ndays = min(args.num_days, test_S)
    rows: list[dict] = []

    for k in range(Ndays):
        sample_idx = test_start + k

        if args.mode == "X":
            x_in = X[sample_idx]
        elif args.mode == "Y":
            x_in = Yhist[sample_idx]
        else:
            x_in = XY[sample_idx]

        y_pred_24L = predict_day(model, x_in, device=device)

        # Build 24h demand block from raw demands series
        hour0 = sample_idx + lookback
        demands_24N = demands_all_TN[hour0 : hour0 + horizon, :]
        if demands_24N.shape[0] != horizon:
            break

        # Solve FULL once per day (independent of threshold)
        full = solve_full_tcuc_24h(
            demands_TN=demands_24N,
            gens_bus=gens_bus,
            p_min=p_min,
            p_max=p_max,
            cost_lin=cost_lin,
            PTDF=PTDF,
            f_max=f_max,
            solver_name=args.solver,
            time_limit_sec=args.time_limit,
        )

        for tau in thresholds:
            active_lines = active_lines_from_predicted_loadings(
                y_pred_24L, threshold=tau, policy="any_hour"
            )

            # Default repair cols (always logged so CSV schema is stable)
            repair_cols = {
                "repair_enabled": bool(args.repair),
                "repair_rounds": 0,
                "repair_added_lines": 0,
                "repair_final_active_lines": int(len(active_lines)),
            }

            if args.repair:
                screened, rinfo = solve_screened_with_repair_tcuc_24h(
                    demands_TN=demands_24N,
                    gens_bus=gens_bus,
                    p_min=p_min,
                    p_max=p_max,
                    cost_lin=cost_lin,
                    PTDF=PTDF,
                    f_max=f_max,
                    active_lines=active_lines,
                    solver_name=args.solver,
                    time_limit_sec=args.time_limit,
                    tol=args.repair_tol,
                    max_rounds=args.max_repair_rounds,
                )

                final_active = np.asarray(
                    getattr(rinfo, "final_active_lines", active_lines), dtype=int
                )
                active_for_metrics = final_active

                repair_cols.update(
                    {
                        "repair_rounds": int(
                            getattr(rinfo, "rounds", getattr(rinfo, "n_rounds", 0))
                        ),
                        "repair_added_lines": int(
                            getattr(rinfo, "added_lines", getattr(rinfo, "n_added", 0))
                        ),
                        "repair_final_active_lines": int(final_active.size),
                    }
                )
            else:
                screened = solve_screened_tcuc_24h(
                    demands_TN=demands_24N,
                    gens_bus=gens_bus,
                    p_min=p_min,
                    p_max=p_max,
                    cost_lin=cost_lin,
                    PTDF=PTDF,
                    f_max=f_max,
                    active_lines=active_lines,
                    solver_name=args.solver,
                    time_limit_sec=args.time_limit,
                )
                active_for_metrics = np.asarray(active_lines, dtype=int)

            met = compute_uc_eval_metrics(
                full=full,
                screened=screened,
                f_max=f_max,
                active_lines=active_for_metrics,
            )

            rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "hour0": int(hour0),
                    "threshold": float(tau),
                    "mode": args.mode,
                    **repair_cols,
                    **met.__dict__,
                }
            )

        print(f"Done day {k+1}/{Ndays} (sample_idx={sample_idx})")

    out_dir = root / "results" / "screened_tcuc"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    out_csv = out_dir / f"uc_eval_{args.mode}.csv"
    df.to_csv(out_csv, index=False)

    print("\nSAVED:", out_csv)

    # Summary
    summary_cols = [
        "constraint_reduction",
        "speedup",
        "objective_gap_rel",
        "max_violation_ratio_removed",
        "violation_rate_removed",
        "repair_rounds",
        "repair_added_lines",
        "repair_final_active_lines",
    ]
    present = [c for c in summary_cols if c in df.columns]
    if len(df) and present:
        print(df.groupby("threshold")[present].mean(numeric_only=True))


if __name__ == "__main__":
    main()
