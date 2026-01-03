#!/usr/bin/env python3
"""
02_generate_labels.py  (PAPER-FAITHFUL, SINGLE-HOUR MIP) + SAFETY/RESUME

Paper constraints per hour:
(1) min sum_g c_g p_g
(2) sum_g p_g = sum_n D_n
(3) x_g Pmin_g <= p_g <= x_g Pmax_g, x_g in {0,1}
(4) -fmax_l <= f_l <= fmax_l
(5) f_l = sum_n alpha_{l,n} ( sum_{g in n} p_g - D_n )

Where alpha is PTDF/GSF mapping bus injections to line flows.

Inputs (standardized):
- data_raw/oasys_ieee96/buses.csv
- data_raw/oasys_ieee96/branches.csv   (line_id, from_bus, to_bus, b, f_max)
- data_raw/oasys_ieee96/generators.csv (gen_id, bus_id, p_min, p_max, cost_lin)
- data_raw/oasys_ieee96/demands.csv    (t, d_1..d_73)

Outputs:
- demands.parquet
- flows.parquet
- loadings.parquet
- objective.parquet
- metadata.json

Safety additions:
- termination condition checks
- finite checks (p/x/inj/flows)
- periodic checkpointing
- resume support
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo


# ---------------------------
# PTDF (alpha_{l,n}) builder
# ---------------------------

def build_ptdf_from_susceptance(
    buses_df: pd.DataFrame,
    branches_df: pd.DataFrame,
    slack_bus_id: int | None = None
):
    """
    Build DC PTDF alpha_{l,n} using line susceptances b.
    No explicit inverse; uses np.linalg.solve for stability.

    Returns:
      PTDF: (L, N)
      bus_ids: list[int]
      slack_bus_id: int
    """
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

    # Build nodal susceptance matrix B
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

        h_red = h[keep]
        y = np.linalg.solve(Bred, h_red)

        PTDF[ell, keep] = y
        PTDF[ell, slack] = -PTDF[ell, keep].sum()

    return PTDF, bus_ids, slack_bus_id


# ---------------------------
# Single-hour MIP solver
# ---------------------------

def solve_hour_mip(
    demands_bus_mw: np.ndarray,      # (N,)
    gens_bus: np.ndarray,            # (G,) bus index per generator
    p_min: np.ndarray,
    p_max: np.ndarray,
    cost_lin: np.ndarray,
    PTDF: np.ndarray,                # (L,N)
    f_max: np.ndarray,               # (L,)
    solver_name: str = "highs",
    time_limit_sec: int | None = None,
):
    """
    Returns:
      p (G,), x (G,), flows (L,), objective (float)
    """
    N = demands_bus_mw.shape[0]
    G = cost_lin.shape[0]
    L = f_max.shape[0]

    m = pyo.ConcreteModel()
    m.G = pyo.RangeSet(0, G - 1)
    m.L = pyo.RangeSet(0, L - 1)

    m.p = pyo.Var(m.G, domain=pyo.NonNegativeReals)
    m.x = pyo.Var(m.G, domain=pyo.Binary)

    m.obj = pyo.Objective(
        expr=sum(float(cost_lin[g]) * m.p[g] for g in m.G), sense=pyo.minimize)

    # x-linked bounds
    m.p_lb = pyo.Constraint(
        m.G, rule=lambda m, g: m.p[g] >= float(p_min[g]) * m.x[g])
    m.p_ub = pyo.Constraint(
        m.G, rule=lambda m, g: m.p[g] <= float(p_max[g]) * m.x[g])

    # Balance
    total_demand = float(demands_bus_mw.sum())
    m.balance = pyo.Constraint(expr=sum(m.p[g] for g in m.G) == total_demand)

    # Map generators to buses
    gen_at_bus = [[] for _ in range(N)]
    for g in range(G):
        gen_at_bus[int(gens_bus[g])].append(g)

    def flow_expr(l):
        expr = 0.0
        for n in range(N):
            gen_sum = sum(m.p[g]
                          for g in gen_at_bus[n]) if gen_at_bus[n] else 0.0
            inj_n = gen_sum - float(demands_bus_mw[n])
            expr += float(PTDF[l, n]) * inj_n
        return expr

    m.flow_pos = pyo.Constraint(
        m.L, rule=lambda m, l: flow_expr(l) <= float(f_max[l]))
    m.flow_neg = pyo.Constraint(
        m.L, rule=lambda m, l: flow_expr(l) >= -float(f_max[l]))

    solver = pyo.SolverFactory(solver_name)
    if time_limit_sec is not None:
        try:
            solver.options["time_limit"] = float(time_limit_sec)
        except Exception:
            pass

    res = solver.solve(m, tee=False)

    # ---- SAFETY: ensure solution is usable ----
    tc = getattr(res.solver, "termination_condition", None)
    ok = {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
        pyo.TerminationCondition.locallyOptimal,
    }
    if tc not in ok:
        raise RuntimeError(f"Solver unusable termination_condition={tc}")

    p = np.array([pyo.value(m.p[g]) for g in range(G)], dtype=float)
    x = np.array([pyo.value(m.x[g]) for g in range(G)], dtype=float)

    if (not np.isfinite(p).all()) or (not np.isfinite(x).all()):
        raise RuntimeError("Non-finite p or x returned by solver (nan/inf).")

    # numeric flows
    inj = np.zeros(N, dtype=float)
    for n in range(N):
        inj[n] = p[gens_bus == n].sum() - demands_bus_mw[n]

    if not np.isfinite(inj).all():
        raise RuntimeError("Non-finite injection vector (nan/inf).")

    flows = PTDF.dot(inj)

    if not np.isfinite(flows).all():
        raise RuntimeError(
            "Non-finite flows after PTDF multiply. "
            f"max|inj|={float(np.max(np.abs(inj)))} max|PTDF|={float(np.max(np.abs(PTDF)))}"
        )

    obj = float(pyo.value(m.obj))
    if not np.isfinite(obj):
        raise RuntimeError("Non-finite objective (nan/inf).")

    return p, x, flows, obj


# ---------------------------
# Checkpoint helpers
# ---------------------------

def save_checkpoint(out_dir: Path, t_done: int, flows_out: np.ndarray, loadings_out: np.ndarray, obj_out: np.ndarray, idx: pd.DataFrame, line_ids: list):
    ckpt_dir = out_dir / "_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays up to t_done (exclusive index t_done means length t_done)
    flows_df = pd.concat([idx.iloc[:t_done].copy(),
                          pd.DataFrame(flows_out[:t_done], columns=[f"l_{lid}" for lid in line_ids])], axis=1)
    loadings_df = pd.concat([idx.iloc[:t_done].copy(),
                             pd.DataFrame(loadings_out[:t_done], columns=[f"l_{lid}" for lid in line_ids])], axis=1)
    obj_df = pd.concat([idx.iloc[:t_done].copy(),
                        pd.DataFrame({"objective": obj_out[:t_done]})], axis=1)

    flows_df.to_parquet(ckpt_dir / "flows.partial.parquet", index=False)
    loadings_df.to_parquet(ckpt_dir / "loadings.partial.parquet", index=False)
    obj_df.to_parquet(ckpt_dir / "objective.partial.parquet", index=False)

    with open(ckpt_dir / "progress.json", "w") as f:
        json.dump({"t_done": int(t_done)}, f, indent=2)


def try_resume(out_dir: Path, T: int, L: int, line_ids: list):
    ckpt_dir = out_dir / "_checkpoints"
    prog = ckpt_dir / "progress.json"
    if not prog.exists():
        return 0, None, None, None

    with open(prog, "r") as f:
        t_done = int(json.load(f).get("t_done", 0))

    flows_p = ckpt_dir / "flows.partial.parquet"
    load_p = ckpt_dir / "loadings.partial.parquet"
    obj_p = ckpt_dir / "objective.partial.parquet"
    if not (flows_p.exists() and load_p.exists() and obj_p.exists()):
        return 0, None, None, None

    flows_df = pd.read_parquet(flows_p)
    load_df = pd.read_parquet(load_p)
    obj_df = pd.read_parquet(obj_p)

    # allocate full arrays
    flows_out = np.zeros((T, L), dtype=float)
    loadings_out = np.zeros((T, L), dtype=float)
    obj_out = np.zeros((T,), dtype=float)

    # fill
    flows_out[:t_done, :] = flows_df[[
        f"l_{lid}" for lid in line_ids]].to_numpy(dtype=float)
    loadings_out[:t_done, :] = load_df[[
        f"l_{lid}" for lid in line_ids]].to_numpy(dtype=float)
    obj_out[:t_done] = obj_df["objective"].to_numpy(dtype=float)

    return t_done, flows_out, loadings_out, obj_out


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24,
                    help="Use 8640 for paper horizon.")
    ap.add_argument("--out", type=str,
                    default="data_processed/rts96_v1", help="Output directory")
    ap.add_argument("--time_limit", type=int, default=None,
                    help="Optional time limit per hour (sec)")
    ap.add_argument("--checkpoint_every", type=int, default=200,
                    help="Checkpoint frequency in hours.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last checkpoint if present.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data_raw" / "oasys_ieee96"
    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load standardized CSVs
    buses = pd.read_csv(raw_dir / "buses.csv")
    branches = pd.read_csv(raw_dir / "branches.csv")
    gens = pd.read_csv(raw_dir / "generators.csv")
    dem_all = pd.read_csv(raw_dir / "demands.csv")

    dem = dem_all.iloc[:args.hours].copy()
    demand_cols = [c for c in dem.columns if c.startswith("d_")]

    # Quick adequacy sanity
    total_pmax = float(gens["p_max"].astype(float).sum())
    peak_demand = float(dem[demand_cols].sum(axis=1).max())
    print("Total generation Pmax:", total_pmax)
    print("Peak total demand:", peak_demand)

    # PTDF
    PTDF, bus_ids, slack_bus_id = build_ptdf_from_susceptance(
        buses, branches, slack_bus_id=None)
    print("PTDF finite?", np.isfinite(PTDF).all())
    print("PTDF max abs:", float(np.max(np.abs(PTDF))))

    bus_index = {b: i for i, b in enumerate(bus_ids)}

    gens_bus = gens["bus_id"].astype(int).map(bus_index).to_numpy()
    p_min = gens["p_min"].astype(float).to_numpy()
    p_max = gens["p_max"].astype(float).to_numpy()
    cost_lin = gens["cost_lin"].astype(float).to_numpy()

    f_max = branches["f_max"].astype(float).to_numpy()
    line_ids = branches["line_id"].tolist()

    T = len(dem)
    L = len(line_ids)

    # Allocate output arrays
    idx = dem[["t"]].copy() if "t" in dem.columns else pd.DataFrame(
        {"t": range(T)})

    start_t = 0
    flows_out = np.zeros((T, L), dtype=float)
    loadings_out = np.zeros((T, L), dtype=float)
    obj_out = np.zeros((T,), dtype=float)

    if args.resume:
        rt, f0, l0, o0 = try_resume(out_dir, T, L, line_ids)
        if f0 is not None:
            start_t = rt
            flows_out = f0
            loadings_out = l0
            obj_out = o0
            print(f"Resuming from checkpoint: start_t={start_t}")

    t0 = time.time()
    for t in range(start_t, T):
        demands_bus = dem.loc[t, demand_cols].to_numpy(dtype=float)

        try:
            p, x, flows, obj = solve_hour_mip(
                demands_bus_mw=demands_bus,
                gens_bus=gens_bus,
                p_min=p_min,
                p_max=p_max,
                cost_lin=cost_lin,
                PTDF=PTDF,
                f_max=f_max,
                solver_name="highs",
                time_limit_sec=args.time_limit,
            )
        except Exception:
            print(
                f"\nFAILED at hour t={t}, total demand={float(demands_bus.sum()):.3f} MW")
            raise

        flows_out[t, :] = flows
        loadings_out[t, :] = np.abs(flows) / f_max
        obj_out[t] = obj

        if (t + 1) % 6 == 0:
            elapsed = time.time() - t0
            print(f"Solved {t+1}/{T} hours. Elapsed: {elapsed:.1f}s")

        if args.checkpoint_every and (t + 1) % args.checkpoint_every == 0:
            save_checkpoint(out_dir, t + 1, flows_out,
                            loadings_out, obj_out, idx, line_ids)
            print(f"Checkpoint saved at t={t+1}")

    # Final write
    flows_df = pd.concat([idx, pd.DataFrame(flows_out, columns=[
                         f"l_{lid}" for lid in line_ids])], axis=1)
    loadings_df = pd.concat([idx, pd.DataFrame(loadings_out, columns=[
                            f"l_{lid}" for lid in line_ids])], axis=1)
    obj_df = pd.concat([idx, pd.DataFrame({"objective": obj_out})], axis=1)
    demands_df = dem[[
        "t"] + demand_cols].copy() if "t" in dem.columns else dem[demand_cols].copy()

    demands_df.to_parquet(out_dir / "demands.parquet", index=False)
    flows_df.to_parquet(out_dir / "flows.parquet", index=False)
    loadings_df.to_parquet(out_dir / "loadings.parquet", index=False)
    obj_df.to_parquet(out_dir / "objective.parquet", index=False)

    meta = {
        "case": "rts96_v1",
        "hours_used": int(T),
        "N_buses": int(len(bus_ids)),
        "L_lines": int(L),
        "G_gens": int(len(cost_lin)),
        "slack_bus_id": int(slack_bus_id),
        "solver": "highs",
        "model": "single_hour_MIP_with_binary_commitment",
        "PTDF_max_abs": float(np.max(np.abs(PTDF))),
        "checkpoint_every": int(args.checkpoint_every),
        "resumable": True,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    mx = float(loadings_out.max())
    p95 = float(np.quantile(loadings_out, 0.95))
    print("\nDONE")
    print(f"Sanity: max loading={mx:.3f}, 95th percentile={p95:.3f}")
    print("Wrote outputs to:", out_dir)


if __name__ == "__main__":
    main()
