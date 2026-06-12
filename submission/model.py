"""SAIR Modular Arithmetic Challenge submission (self-contained).

a and b are reduced mod p (a legal two-argument reduction), then a p-conditioned
network computes (a * b) mod p over residues < p. The output is a single base-p
digit equal to the residue, masked to [0, p) so it is always valid. Primes
outside the trained Tier-2 range fall back to 0 without invoking the network.
The multiplication is produced by trained parameters, not hand-coded arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from modchallenge.interface.base_model import ModularMultiplicationModel


class PCondModMultNet(nn.Module):
    def __init__(
        self,
        p_max: int = 256,
        d_model: int = 256,
        hidden: int = 1024,
        n_blocks: int = 3,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.p_max = p_max
        self.res_embed = nn.Embedding(p_max, d_model)
        self.p_embed = nn.Embedding(p_max, d_model)
        self.fc_in = nn.Linear(2 * d_model, hidden)
        self.film = nn.Linear(d_model, 2 * hidden)
        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_blocks))
        self.head = nn.Linear(hidden, p_max)
        self.activation = activation

    def _act(self, z: torch.Tensor) -> torch.Tensor:
        return z * z if self.activation == "quad" else F.gelu(z)

    def forward(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        h = self._act(
            self.fc_in(torch.cat([self.res_embed(x), self.res_embed(y)], dim=-1))
        )
        gamma, beta = self.film(self.p_embed(p)).chunk(2, dim=-1)
        h = h * gamma + beta
        for blk in self.blocks:
            h = h + self._act(blk(h))
        return self.head(h)


def _masked_logits(logits: torch.Tensor, p: torch.Tensor, p_max: int) -> torch.Tensor:
    cols = torch.arange(p_max, device=logits.device).unsqueeze(0)
    return logits.masked_fill(cols >= p.unsqueeze(1), float("-inf"))


class SairModMult(ModularMultiplicationModel):
    def load(self, model_dir: str) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(
            Path(model_dir) / "weights.pt", map_location=self.device, weights_only=True
        )
        self.p_max = ckpt["config"]["p_max"]
        self.model = PCondModMultNet(**ckpt["config"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def preprocess_a(self, a: str):
        return a

    def preprocess_b(self, b: str):
        return b

    def preprocess_p(self, p: str):
        return p

    @torch.no_grad()
    def predict_digits(self, a_enc, b_enc, p_enc) -> list[int]:
        return self.predict_digits_batch([(a_enc, b_enc, p_enc)])[0]

    @torch.no_grad()
    def predict_digits_batch(self, inputs) -> list[list[int]]:
        out: list[list[int]] = [[0] for _ in inputs]
        xs, ys, ps, idx = [], [], [], []
        for i, (a_enc, b_enc, p_enc) in enumerate(inputs):
            p = int(p_enc)
            if p >= self.p_max:
                continue
            xs.append(int(a_enc) % p)
            ys.append(int(b_enc) % p)
            ps.append(p)
            idx.append(i)
        if idx:
            x = torch.tensor(xs, dtype=torch.long, device=self.device)
            y = torch.tensor(ys, dtype=torch.long, device=self.device)
            p = torch.tensor(ps, dtype=torch.long, device=self.device)
            logits = _masked_logits(self.model(x, y, p), p, self.p_max)
            preds = logits.argmax(dim=-1).tolist()
            for j, i in enumerate(idx):
                out[i] = [int(preds[j])]
        return out

    def max_batch_size(self) -> int:
        return 1024
