#!/usr/bin/env python3
"""
01_fetch_raw_data.py

Purpose:
1) Ensure the OASYS IEEE RTS-96 data repo exists locally (data_ieee96/)
2) Copy the required raw CSVs into data_raw/oasys_ieee96/
3) Standardize them into:
   - buses.csv
   - branches.csv
   - generators.csv
   - demands.csv

This keeps *all* raw-data responsibilities in one place.
"""

from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path
import pandas as pd


REPO_URL = "https://github.com/groupoasys/data_ieee96.git"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_REPO_DIR = PROJECT_ROOT / "data_ieee96"
RAW_DIR = PROJECT_ROOT / "data_raw" / "oasys_ieee96"

REQUIRED_RAW_FILES = [
    "lines_ieee96.csv",
    "thermal_ieee96.csv",
    "load_ieee96.csv",
]


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_repo():
    if RAW_REPO_DIR.exists():
        print(f"Found existing repo at: {RAW_REPO_DIR}")
        return
    print(f"Cloning OASYS RTS-96 repo into: {RAW_REPO_DIR}")
    run(["git", "clone", REPO_URL, str(RAW_REPO_DIR)])


def copy_raw_files():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname in REQUIRED_RAW_FILES:
        src = RAW_REPO_DIR / fname
        if not src.exists():
            raise FileNotFoundError(
                f"Missing {fname} in {RAW_REPO_DIR}. Found: {list(RAW_REPO_DIR.iterdir())[:10]}")
        dst = RAW_DIR / fname
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")


def standardize_csvs():
    """
    Converts:
      lines_ieee96.csv    -> branches.csv  (with susceptance b and f_max)
      thermal_ieee96.csv  -> generators.csv
      load_ieee96.csv     -> demands.csv (wide, with d_<bus_id> columns + hour index)
      + buses.csv
    """
    lines = pd.read_csv(RAW_DIR / "lines_ieee96.csv")
    gens = pd.read_csv(RAW_DIR / "thermal_ieee96.csv")
    load = pd.read_csv(RAW_DIR / "load_ieee96.csv")

    # 1) buses.csv - infer from load columns '1'..'73'
    bus_ids = sorted([int(c) for c in load.columns])
    buses = pd.DataFrame({"bus_id": bus_ids})
    buses.to_csv(RAW_DIR / "buses.csv", index=False)

    # 2) branches.csv (store susceptance as 'b' to build PTDF later)
    branches = pd.DataFrame({
        "line_id":  lines["# line"],
        "from_bus": lines["from bus"].astype(int),
        "to_bus":   lines["to bus"].astype(int),
        "b":        lines["Suscep (MW)"].astype(float),
        "f_max":    lines["Pmax (MW)"].astype(float),
    })
    branches.to_csv(RAW_DIR / "branches.csv", index=False)

    # 3) generators.csv
    generators = pd.DataFrame({
        "gen_id":   gens["# gen"],
        "bus_id":   gens["# bus"].astype(int),
        "p_min":    gens["Pmin (MW)"].astype(float),
        "p_max":    gens["Pmax (MW)"].astype(float),
        "cost_lin": gens["cost (€/Mwh)"].astype(float),
        # keep ramps for later, not used in simplified model
        "ramp_down": gens["RampDO (MW)"].astype(float),
        "ramp_up":   gens["RampUP (MW)"].astype(float),
    })
    generators.to_csv(RAW_DIR / "generators.csv", index=False)

    # 4) demands.csv (wide)
    dem = load.copy()
    dem.columns = [f"d_{c}" for c in dem.columns]  # d_1..d_73
    dem.insert(0, "t", range(len(dem)))            # explicit hour index
    dem.to_csv(RAW_DIR / "demands.csv", index=False)

    print("Standardized files written to:", RAW_DIR)
    print(" - buses.csv")
    print(" - branches.csv")
    print(" - generators.csv")
    print(" - demands.csv")
    print("Demands hours:", len(dem))


def main():
    ensure_repo()
    copy_raw_files()
    standardize_csvs()
    print("\nStep 01 complete.")


if __name__ == "__main__":
    main()
