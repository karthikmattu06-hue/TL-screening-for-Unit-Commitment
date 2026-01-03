from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GraphConv(nn.Module):
    """
    Simple GCN layer: H = A_hat X W
    Shapes:
      X: (B, L, F_in)
      A_hat: (L, L)  (pre-normalized, includes self loops)
      W: (F_in, F_out)
      H: (B, L, F_out)
    """

    def __init__(self, f_in: int, f_out: int, A_hat: np.ndarray):
        super().__init__()
        self.W = nn.Linear(f_in, f_out, bias=False)
        A_hat = np.asarray(A_hat, dtype=np.float32)
        self.register_buffer("A_hat", torch.from_numpy(A_hat))  # (L,L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F_in)
        xw = self.W(x)  # (B, L, F_out)
        # Apply A_hat over the node dimension L (middle axis), not over features:
        # (L, L) @ (B, L, F_out) -> (B, L, F_out)
        # A_hat: (L,L), xw: (B,L,F_out)  => out: (B,L,F_out)
        return torch.einsum("ij,bjk->bik", self.A_hat, xw)


class GCNLSTMRegressor(nn.Module):
    """
    Paper-like pipeline (adapted to your dataset format):
      input:  (B, 192, F)
      output: (B, 24, 120)

    We use the last 120 features as line loadings if mode == XY,
    and all features if mode == Y (where F==120).
    """

    def __init__(
        self,
        input_dim: int,
        num_lines: int,
        A_hat: np.ndarray,
        gcn_hidden: int = 500,
        lstm_hidden: int = 500,
        out_horizon: int = 24,
        use_last_120: bool = False,
    ):
        super().__init__()
        self.num_lines = int(num_lines)
        self.out_horizon = int(out_horizon)
        self.use_last_120 = bool(use_last_120)

        # Two GCN layers like the paper diagram
        self.gcn1 = GraphConv(1, gcn_hidden, A_hat=A_hat)
        self.gcn2 = GraphConv(gcn_hidden, gcn_hidden, A_hat=A_hat)
        self.relu = nn.ReLU()

        # LSTM over time for each line; we’ll run it as (B*L, T, F)
        self.lstm = nn.LSTM(
            input_size=gcn_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Map last hidden state -> 24-step forecast (per line)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 3000),
            nn.ReLU(),
            nn.Linear(3000, out_horizon),
            nn.Sigmoid(),
        )

        # sanity
        if use_last_120:
            if input_dim < num_lines:
                raise ValueError(
                    f"input_dim={input_dim} < num_lines={num_lines} but use_last_120=True")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,192,F)
        B, T, F = x.shape

        if self.use_last_120:
            yhist = x[:, :, -self.num_lines:]  # (B,192,120)
        else:
            yhist = x  # expect (B,192,120) for mode Y

        # reorder to per-time-step node features: (B,T,L)
        yhist = yhist[..., : self.num_lines]
        yhist_BTL = yhist

        # GCN expects (B,L,Fin) for each time step; loop time (192 is small, loop is acceptable)
        gcn_out = []
        for t in range(T):
            xt = yhist_BTL[:, t, :].unsqueeze(-1)  # (B,L,1)
            ht = self.relu(self.gcn1(xt))
            ht = self.relu(self.gcn2(ht))          # (B,L,500)
            gcn_out.append(ht)
        H = torch.stack(gcn_out, dim=1)  # (B,T,L,500)

        # LSTM per line: reshape to (B*L, T, 500)
        H = H.permute(0, 2, 1, 3).contiguous()  # (B,L,T,500)
        H2 = H.view(B * self.num_lines, T, -1)

        o, _ = self.lstm(H2)          # (B*L, T, lstm_hidden)
        last = o[:, -1, :]            # (B*L, lstm_hidden)
        y_line = self.head(last)      # (B*L, 24)
        y_line = y_line.view(B, self.num_lines, self.out_horizon)  # (B,L,24)

        # return in your standard shape (B,24,120)
        return y_line.permute(0, 2, 1).contiguous()
