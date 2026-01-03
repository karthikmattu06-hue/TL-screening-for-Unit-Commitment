from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pyomo.environ as pyo

from .solve_full import SolveResult, build_tcuc_model_24h, compute_flows_all_lines


def _is_status_ok(status: str) -> bool:
    s = (status or "").lower()
    # Pyomo often prints: "optimal", "feasible", "locallyOptimal", etc.
    return s in ("optimal", "feasible", "locallyoptimal", "locally_optimal")


def solve_screened_tcuc_24h(
    demands_TN: np.ndarray,
    gens_bus: np.ndarray,
    p_min: np.ndarray,
    p_max: np.ndarray,
    cost_lin: np.ndarray,
    PTDF: np.ndarray,
    f_max: np.ndarray,
    active_lines: Sequence[int],  # indices of lines whose limits are enforced
    *,
    solver_name: str = "highs",
    time_limit_sec: Optional[float] = None,
) -> SolveResult:
    active_lines = list(map(int, np.asarray(active_lines).ravel().tolist()))

    t0 = time.time()
    m = build_tcuc_model_24h(
        demands_TN=demands_TN,
        gens_bus=gens_bus,
        p_min=p_min,
        p_max=p_max,
        cost_lin=cost_lin,
        PTDF=PTDF,
        f_max=f_max,
        active_lines=active_lines,
    )

    solver = pyo.SolverFactory(solver_name)
    if time_limit_sec is not None:
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

    # Always compute flows on ALL lines for violation checks on removed constraints
    flows_TL = compute_flows_all_lines(demands_TN, p_TG, gens_bus, PTDF)

    ok = _is_status_ok(status)

    return SolveResult(
        ok=bool(ok),
        status=status,
        objective=float(pyo.value(m.obj)),
        p_TG=p_TG,
        x_TG=x_TG,
        flows_TL=flows_TL,
        solve_time_sec=float(elapsed),
    )


def find_violated_removed_lines(
    flows_TL: np.ndarray,           # (T,L) flows on ALL lines
    f_max: np.ndarray,              # (L,)
    active_lines: Sequence[int],    # currently-enforced line indices
    *,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Given a SCREENED solution's flows, find which REMOVED lines violate limits.

    Returns:
      violated_lines: (K,) int array of global line indices to add back.
    """
    flows_TL = np.asarray(flows_TL, dtype=float)
    f_max = np.asarray(f_max, dtype=float)
    L = int(f_max.shape[0])

    active_lines = np.asarray(list(map(int, active_lines)), dtype=int)
    active_mask = np.zeros(L, dtype=bool)
    if active_lines.size > 0:
        active_mask[active_lines] = True
    removed_mask = ~active_mask

    if not removed_mask.any():
        return np.array([], dtype=int)

    denom = f_max[removed_mask]
    if (denom <= 0).any() or (not np.isfinite(denom).all()):
        raise ValueError("Invalid f_max on removed lines (must be finite and > 0).")

    ratios = np.abs(flows_TL[:, removed_mask]) / denom[None, :]
    # treat non-finite as violation to be conservative
    finite = np.isfinite(ratios)
    if not finite.all():
        ratios = np.where(finite, ratios, np.inf)

    viol_any = (ratios > (1.0 + tol)).any(axis=0)   # per removed line
    removed_indices = np.where(removed_mask)[0]
    violated_lines = removed_indices[viol_any].astype(int)
    return violated_lines


@dataclass
class RepairInfo:
    rounds_used: int
    initial_active_count: int
    final_active_count: int
    added_lines_total: int
    final_active_lines: np.ndarray
    violated_lines_last_round: np.ndarray


def solve_screened_with_repair_tcuc_24h(
    demands_TN: np.ndarray,
    gens_bus: np.ndarray,
    p_min: np.ndarray,
    p_max: np.ndarray,
    cost_lin: np.ndarray,
    PTDF: np.ndarray,
    f_max: np.ndarray,
    active_lines: Sequence[int],
    *,
    solver_name: str = "highs",
    time_limit_sec: Optional[float] = None,
    tol: float = 1e-9,
    max_rounds: int = 5,
) -> Tuple[SolveResult, RepairInfo]:
    """
    Screening + feasibility repair loop:
      - Solve screened UC with given active_lines
      - Check removed-line violations using resulting flows
      - Add violated removed lines back into active set and re-solve
      - Repeat until no violations or max_rounds reached

    Returns:
      (result, repair_info)
    """
    L = int(np.asarray(f_max).shape[0])

    active = np.asarray(list(map(int, np.asarray(active_lines).ravel().tolist())), dtype=int)
    active = np.unique(active[(active >= 0) & (active < L)])  # sanitize
    initial_active_count = int(active.size)

    added_total = 0
    violated_last = np.array([], dtype=int)
    result: SolveResult | None = None

    rounds = 0
    while True:
        rounds += 1
        result = solve_screened_tcuc_24h(
            demands_TN=demands_TN,
            gens_bus=gens_bus,
            p_min=p_min,
            p_max=p_max,
            cost_lin=cost_lin,
            PTDF=PTDF,
            f_max=f_max,
            active_lines=active,
            solver_name=solver_name,
            time_limit_sec=time_limit_sec,
        )

        # If solver itself didn't return a usable solution, stop early.
        if not result.ok:
            violated_last = np.array([], dtype=int)
            break

        violated = find_violated_removed_lines(
            flows_TL=result.flows_TL,
            f_max=f_max,
            active_lines=active,
            tol=tol,
        )

        violated_last = violated

        if violated.size == 0:
            break

        new_active = np.unique(np.concatenate([active, violated]))
        added_total += int(new_active.size - active.size)
        active = new_active

        if rounds >= max_rounds:
            break

    assert result is not None
    info = RepairInfo(
        rounds_used=int(rounds),
        initial_active_count=int(initial_active_count),
        final_active_count=int(active.size),
        added_lines_total=int(added_total),
        final_active_lines=active.astype(int),
        violated_lines_last_round=violated_last.astype(int),
    )
    return result, info
