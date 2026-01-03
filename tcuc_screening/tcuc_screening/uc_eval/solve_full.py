# tcuc_screening/uc_eval/solve_full.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import time

import numpy as np
import pyomo.environ as pyo


@dataclass
class SolveResult:
    ok: bool
    status: str
    objective: float
    p_TG: np.ndarray          # (T, G)
    x_TG: np.ndarray          # (T, G)
    flows_TL: np.ndarray      # (T, L)  flows on ALL lines
    solve_time_sec: float


def build_tcuc_model_24h(
    demands_TN: np.ndarray,       # (T,N)
    gens_bus: np.ndarray,         # (G,) bus index per gen
    p_min: np.ndarray,            # (G,)
    p_max: np.ndarray,            # (G,)
    cost_lin: np.ndarray,         # (G,)
    PTDF: np.ndarray,             # (L,N)
    f_max: np.ndarray,            # (L,)
    active_lines: Optional[Sequence[int]] = None,
):
    """
    24h TC-UC MILP (paper-faithful simplified version: commitment + DC line limits).
    If active_lines is provided, enforce line constraints only for those lines.
    """
    demands_TN = np.asarray(demands_TN, dtype=float)
    PTDF = np.asarray(PTDF, dtype=float)
    f_max = np.asarray(f_max, dtype=float)
    gens_bus = np.asarray(gens_bus, dtype=int)

    T, N = demands_TN.shape
    L = int(f_max.shape[0])
    G = int(cost_lin.shape[0])

    if active_lines is None:
        active_lines = list(range(L))
    active_lines = list(map(int, active_lines))

    # generator indices per bus
    gen_at_bus = [[] for _ in range(N)]
    for g in range(G):
        gen_at_bus[int(gens_bus[g])].append(g)

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T - 1)
    m.G = pyo.RangeSet(0, G - 1)
    m.LA = pyo.RangeSet(0, len(active_lines) - 1)

    m.p = pyo.Var(m.T, m.G, domain=pyo.NonNegativeReals)
    m.x = pyo.Var(m.T, m.G, domain=pyo.Binary)

    m.obj = pyo.Objective(
        expr=sum(float(cost_lin[g]) * m.p[t, g] for t in m.T for g in m.G),
        sense=pyo.minimize,
    )

    m.p_lb = pyo.Constraint(
        m.T, m.G, rule=lambda m, t, g: m.p[t, g] >= float(p_min[g]) * m.x[t, g]
    )
    m.p_ub = pyo.Constraint(
        m.T, m.G, rule=lambda m, t, g: m.p[t, g] <= float(p_max[g]) * m.x[t, g]
    )

    def balance_rule(m, t):
        return sum(m.p[t, g] for g in m.G) == float(demands_TN[int(t), :].sum())

    m.balance = pyo.Constraint(m.T, rule=balance_rule)

    def flow_expr(t: int, l: int):
        expr = 0.0
        for n in range(N):
            gen_sum = sum(m.p[t, g]
                          for g in gen_at_bus[n]) if gen_at_bus[n] else 0.0
            inj = gen_sum - float(demands_TN[t, n])
            expr += float(PTDF[l, n]) * inj
        return expr

    def flow_pos_rule(m, t, la):
        l = active_lines[int(la)]
        return flow_expr(int(t), int(l)) <= float(f_max[l])

    def flow_neg_rule(m, t, la):
        l = active_lines[int(la)]
        return flow_expr(int(t), int(l)) >= -float(f_max[l])

    m.flow_pos = pyo.Constraint(m.T, m.LA, rule=flow_pos_rule)
    m.flow_neg = pyo.Constraint(m.T, m.LA, rule=flow_neg_rule)

    return m


def compute_flows_all_lines(
    demands_TN: np.ndarray,   # (T,N)
    p_TG: np.ndarray,         # (T,G)
    gens_bus: np.ndarray,     # (G,)
    PTDF: np.ndarray,         # (L,N)
):
    demands_TN = np.asarray(demands_TN, dtype=np.float64)
    p_TG = np.asarray(p_TG, dtype=np.float64)
    gens_bus = np.asarray(gens_bus, dtype=int)
    PTDF = np.asarray(PTDF, dtype=np.float64)

    T, N = demands_TN.shape
    L = PTDF.shape[0]
    flows_TL = np.zeros((T, L), dtype=np.float64)

    for t in range(T):
        inj = np.zeros(N, dtype=np.float64)
        for n in range(N):
            inj[n] = p_TG[t, gens_bus == n].sum() - demands_TN[t, n]

        if not np.isfinite(inj).all():
            raise FloatingPointError(f"Non-finite injection at t={t}")

        # BLAS-free: f = PTDF @ inj
        f = np.sum(PTDF * inj[None, :], axis=1)

        if not np.isfinite(f).all():
            bad = np.where(~np.isfinite(f))[0][:5]
            raise FloatingPointError(
                f"Non-finite flow at t={t}, bad lines={bad.tolist()}, "
                f"max|PTDF|={float(np.max(np.abs(PTDF)))} max|inj|={float(np.max(np.abs(inj)))}"
            )

        flows_TL[t, :] = f

    return flows_TL


def solve_full_tcuc_24h(
    demands_TN: np.ndarray,   # (24,N)
    gens_bus: np.ndarray,
    p_min: np.ndarray,
    p_max: np.ndarray,
    cost_lin: np.ndarray,
    PTDF: np.ndarray,
    f_max: np.ndarray,
    solver_name: str = "highs",
    time_limit_sec: float | None = None,
) -> SolveResult:
    t0 = time.time()

    m = build_tcuc_model_24h(
        demands_TN=demands_TN,
        gens_bus=gens_bus,
        p_min=p_min,
        p_max=p_max,
        cost_lin=cost_lin,
        PTDF=PTDF,
        f_max=f_max,
        active_lines=None,
    )

    solver = pyo.SolverFactory(solver_name)
    if time_limit_sec is not None:
        # solver options vary; this is “best effort”
        for key in ("time_limit", "timelimit", "tmlim"):
            try:
                solver.options[key] = float(time_limit_sec)
                break
            except Exception:
                pass

    res = solver.solve(m, tee=False)
    elapsed = time.time() - t0
    status = str(getattr(res.solver, "termination_condition", "unknown"))

    T = int(demands_TN.shape[0])
    G = int(cost_lin.shape[0])

    p_TG = np.zeros((T, G), dtype=float)
    x_TG = np.zeros((T, G), dtype=float)
    for t in range(T):
        for g in range(G):
            p_TG[t, g] = float(pyo.value(m.p[t, g]))
            x_TG[t, g] = float(pyo.value(m.x[t, g]))

    flows_TL = compute_flows_all_lines(demands_TN, p_TG, gens_bus, PTDF)

    ok = status.lower() in ("optimal", "feasible", "locallyoptimal", "locally_optimal")

    return SolveResult(
        ok=bool(ok),
        status=status,
        objective=float(pyo.value(m.obj)),
        p_TG=p_TG,
        x_TG=x_TG,
        flows_TL=flows_TL,
        solve_time_sec=float(elapsed),
    )
