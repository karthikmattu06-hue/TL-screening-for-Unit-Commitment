from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LineGraph:
    A_hat: np.ndarray  # (L, L) normalized adjacency with self-loops


def build_line_adjacency_from_branches(branches_df: pd.DataFrame) -> np.ndarray:
    """
    Line graph adjacency: lines are connected if they share an endpoint bus.
    branches_df must have: from_bus, to_bus (bus ids).
    Returns A (L,L) binary adjacency WITHOUT normalization/self-loops.
    """
    fb = branches_df["from_bus"].astype(int).to_numpy()
    tb = branches_df["to_bus"].astype(int).to_numpy()
    L = len(branches_df)

    # Map bus -> list of incident line indices
    bus_to_lines: dict[int, list[int]] = {}
    for ell in range(L):
        bus_to_lines.setdefault(int(fb[ell]), []).append(ell)
        bus_to_lines.setdefault(int(tb[ell]), []).append(ell)

    A = np.zeros((L, L), dtype=np.float64)
    for lines in bus_to_lines.values():
        # fully connect incident lines
        for i in lines:
            for j in lines:
                if i != j:
                    A[i, j] = 1.0
    return A


def normalize_adj_with_self_loops(A: np.ndarray) -> np.ndarray:
    """
    GCN normalization: A_hat = D^{-1/2} (A + I) D^{-1/2}.

    This version is numerically robust:
    - enforces finite A
    - clamps/repairs bad degrees (<=0, NaN, Inf)
    - avoids diag-matmul overflow by using elementwise scaling
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square (got {A.shape})")

    L = A.shape[0]

    # Ensure adjacency is finite
    if not np.isfinite(A).all():
        bad = np.argwhere(~np.isfinite(A))
        raise FloatingPointError(
            f"Adjacency A contains non-finite values at {bad[:5].tolist()} (showing up to 5)."
        )

    # Add self-loops
    A_tilde = A + np.eye(L, dtype=np.float64)

    # Degree
    deg = A_tilde.sum(axis=1)

    # Repair bad degrees: NaN/Inf/<=0 -> 1.0 (equivalent to isolated node with only self-loop)
    bad_deg = (~np.isfinite(deg)) | (deg <= 0.0)
    if np.any(bad_deg):
        deg = deg.copy()
        deg[bad_deg] = 1.0

    d_inv_sqrt = 1.0 / np.sqrt(deg)  # (L,)

    # Elementwise normalization: D^{-1/2} A_tilde D^{-1/2}
    A_hat = (A_tilde * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]

    # Final safety check
    if not np.isfinite(A_hat).all():
        bad = np.argwhere(~np.isfinite(A_hat))
        raise FloatingPointError(
            f"A_hat contains non-finite values at {bad[:5].tolist()} (showing up to 5)."
        )

    return A_hat


def _resolve_oasys_branches_path(repo_root: Path) -> Path:
    """
    Supports both layouts:
      A) <repo_root>/data_raw/oasys_ieee96/branches.csv
      B) <repo_root>/tcuc_screening/data_raw/oasys_ieee96/branches.csv
    """
    repo_root = Path(repo_root).resolve()

    cand_a = repo_root / "data_raw" / "oasys_ieee96" / "branches.csv"
    if cand_a.exists():
        return cand_a

    cand_b = repo_root / "tcuc_screening" / \
        "data_raw" / "oasys_ieee96" / "branches.csv"
    if cand_b.exists():
        return cand_b

    raise FileNotFoundError(
        "branches.csv not found. Tried:\n"
        f" - {cand_a}\n"
        f" - {cand_b}\n"
        f"repo_root={repo_root}"
    )


def load_rts96_line_graph(repo_root: Path) -> LineGraph:
    """
    Loads RTS-96 branches.csv and builds the normalized line-graph adjacency.
    """
    branches_path = _resolve_oasys_branches_path(repo_root)
    branches = pd.read_csv(branches_path)
    A = build_line_adjacency_from_branches(branches)
    A_hat = normalize_adj_with_self_loops(A)
    return LineGraph(A_hat=A_hat)
