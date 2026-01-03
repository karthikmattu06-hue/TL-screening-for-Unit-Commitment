from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from tcuc_screening.models.gcn_lstm import GCNLSTMRegressor
from tcuc_screening.models.graph_utils import load_rts96_line_graph


# -----------------------------
# Paper-faithful LSTM (existing)
# -----------------------------
class PaperLSTM(nn.Module):
    def __init__(self, input_dim: int, L: int = 120, horizon: int = 24):
        super().__init__()
        self.L = int(L)
        self.horizon = int(horizon)

        self.lstm1 = nn.LSTM(input_size=input_dim,
                             hidden_size=1000, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=1000, hidden_size=500,
                             num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(500, 3000)
        self.fc2 = nn.Linear(3000, 1000)
        self.fc3 = nn.Linear(1000, 3000)
        self.out = nn.Linear(3000, self.horizon * self.L)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o1, _ = self.lstm1(x)
        o2, _ = self.lstm2(o1)
        h = o2[:, -1, :]
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        y = self.sigmoid(self.out(h))
        return y.view(-1, self.horizon, self.L)


# -----------------------------
# Lightweight Transformer regressor
# -----------------------------
class TransformerRegressor(nn.Module):
    """
    Encoder-only Transformer: (B, 192, F) -> (B, 24, L) in [0,1]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        seq_len: int = 192,
        L: int = 120,
        horizon: int = 24,
        pool: Literal["last", "mean"] = "last",
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.L = int(L)
        self.horizon = int(horizon)
        self.pool = pool

        self.in_proj = nn.Linear(input_dim, d_model)

        # learned positional embedding
        self.pos = nn.Parameter(torch.zeros(1, self.seq_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.out = nn.Linear(1024, self.horizon * self.L)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x) + self.pos[:, : x.shape[1], :]
        h = self.encoder(h)

        if self.pool == "mean":
            z = h.mean(dim=1)
        else:
            z = h[:, -1, :]

        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        z = self.relu(self.fc3(z))
        y = self.sigmoid(self.out(z))
        return y.view(-1, self.horizon, self.L)


# -----------------------------
# Spec + factory
# -----------------------------
@dataclass(frozen=True)
class PredictorSpec:
    model: Literal["lstm", "transformer", "gcn_lstm"]
    input_dim: int
    arch: str

    # Needed only for gcn_lstm (to load RTS-96 line graph adjacency)
    repo_root: str | None = None

    L: int = 120
    horizon: int = 24

    # transformer params
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    pool: Literal["last", "mean"] = "last"

    # gcn_lstm params (defaults aligned to your intended diagram; tweak later if needed)
    gcn_hidden: int = 500
    lstm_hidden: int = 500
    # for XY: use last 120 dims (line loading history)
    use_last_120: bool = False


def build_model(spec: PredictorSpec) -> nn.Module:
    if spec.model == "lstm":
        return PaperLSTM(input_dim=spec.input_dim, L=spec.L, horizon=spec.horizon)

    elif spec.model == "transformer":
        return TransformerRegressor(
            input_dim=spec.input_dim,
            d_model=spec.d_model,
            nhead=spec.nhead,
            num_layers=spec.num_layers,
            dim_feedforward=spec.dim_feedforward,
            dropout=spec.dropout,
            seq_len=192,
            L=spec.L,
            horizon=spec.horizon,
            pool=spec.pool,
        )

    elif spec.model == "gcn_lstm":
        if spec.repo_root is None:
            raise ValueError(
                "GCN-LSTM requires spec.repo_root (repo root path) to load RTS-96 line graph.")
        lg = load_rts96_line_graph(Path(spec.repo_root))

        return GCNLSTMRegressor(
            input_dim=spec.input_dim,
            num_lines=spec.L,         # should be 120 for RTS-96
            A_hat=lg.A_hat,           # expected shape (L, L)
            gcn_hidden=spec.gcn_hidden,
            lstm_hidden=spec.lstm_hidden,
            out_horizon=spec.horizon,
            use_last_120=spec.use_last_120,
        )

    raise ValueError(f"Unknown model: {spec.model}")


# -----------------------------
# Loading utilities
# -----------------------------
def load_model(
    ckpt_path: str,
    *,
    model: Literal["lstm", "transformer", "gcn_lstm"],
    input_dim: int,
    device: torch.device,
    arch: str,
    repo_root: str | None = None,
    L: int = 120,
    horizon: int = 24,
    mode: Literal["X", "Y", "XY"] | None = None,
) -> nn.Module:
    """
    Loads a saved model checkpoint into the appropriate architecture.

    - transformer arch parsing stays as you had it
    - gcn_lstm uses repo_root to load adjacency
    """
    # defaults
    d_model, nhead, num_layers, ff, pool = 256, 8, 4, 512, "last"

    if model == "transformer" and arch.startswith("tfm_"):
        parts = arch.split("_")
        for p in parts:
            if p.startswith("d"):
                d_model = int(p[1:])
            elif p.startswith("h"):
                nhead = int(p[1:])
            elif p.startswith("l"):
                num_layers = int(p[1:])
            elif p.startswith("ff"):
                ff = int(p[2:])
            elif p in ("mean", "last"):
                pool = p

    # For XY, we often want the model to consume only last-120 of XY sequence
    use_last_120 = bool(mode == "XY") if mode is not None else False

    spec = PredictorSpec(
        model=model,
        input_dim=input_dim,
        arch=arch,
        repo_root=repo_root,
        L=L,
        horizon=horizon,

        # transformer params (ignored if not transformer)
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=ff,
        pool=pool,  # type: ignore[arg-type]

        # gcn_lstm params (ignored if not gcn_lstm)
        use_last_120=use_last_120,
    )

    m = build_model(spec).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(
        ckpt, dict) and "model_state" in ckpt else ckpt
    m.load_state_dict(state)
    m.eval()
    return m


@torch.no_grad()
def predict_day(model: nn.Module, x_192F: np.ndarray, device: torch.device) -> np.ndarray:
    xb = torch.from_numpy(x_192F[None, :, :]).float().to(device)
    y = model(xb).cpu().numpy()[0]
    return y
