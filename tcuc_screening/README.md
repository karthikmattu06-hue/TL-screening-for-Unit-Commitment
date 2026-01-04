# TCUC Screening

Transmission-Constrained Unit Commitment (TCUC) screening experiments on IEEE RTS-96 style grid data. This repository contains reproducible pipelines and model baselines (LSTM and a basic Transformer) to learn screening signals that can reduce optimization burden in TCUC/SCUC workflows. A GCN-LSTM implementation is included but the full experimental sweep has not yet been executed due to compute constraints.

---

## Repository Status

- **Implemented**
  - LSTM baseline (paper-faithful implementation)
  - Basic Transformer baseline
  - UC evaluation pipeline and orchestration scripts
  - Config-driven experimentation (`configs/`)
- **In Progress**
  - GCN-LSTM implementation (full experiments pending due to limited research compute)
- **Intentionally Not Versioned**
  - Large datasets and generated artifacts (see **Data**)

---

## Project Goals

1. Develop a reproducible framework for screening in transmission-constrained unit commitment.
2. Learn screening signals that reduce downstream UC/MIP computational burden.
3. Compare learning baselines under consistent preprocessing and splits.
4. Provide a transparent public research codebase suitable for extension.

---

## Repository Structure

tcuc_screening/
├── tcuc_screening/
│   ├── models/
│   ├── screening/
│   ├── uc_eval/
│   └── utils/
├── configs/
│   ├── rts96.yaml
│   └── experiment.yaml
├── data_raw/
├── data_processed/
├── datasets/
├── results/
├── strategy/
├── run_uc_eval_all.sh
├── init_repo.py
└── README.md

---

## Data

This repository does not version datasets or generated artifacts.

### External Dataset: OASYS IEEE-96

Source: https://github.com/groupoasys/data_ieee96

Clone into:

tcuc_screening/tcuc_screening/data_ieee96

---

## Environment Setup

Python 3.10+ recommended.

Create environment:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

---

## Models

- **LSTM**: paper-faithful baseline
- **Transformer**: basic baseline
- **GCN-LSTM**: code present, experiments pending

---

## License

See LICENSE file.

---

## Citation

@misc{tcuc_screening,
  title = {TCUC Screening},
  author = {Karthik Mattu and contributors},
  year = {2026},
  howpublished = {GitHub repository}
}
