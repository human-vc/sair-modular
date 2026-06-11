"""Models for learning (x * y) mod p from trained parameters.

The product is never computed in code; the whole map lives in the weights, so
randomising them collapses accuracy (the compliance test). Output is always a
distribution over residue classes; at inference the logits are masked to
[0, p) so the decoded value is a valid residue.

- ModMultNet:      Stage 1, single fixed prime.
- PCondModMultNet: Stage 2, one set of weights conditioned on p, over many primes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModMultNet(nn.Module):
    """Two-layer MLP for (x * y) mod p at a single fixed prime."""

    def __init__(
        self,
        p: int,
        d_model: int = 128,
        hidden: int = 512,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.p = p
        self.embed = nn.Embedding(p, d_model)
        self.fc1 = nn.Linear(2 * d_model, hidden)
        self.head = nn.Linear(hidden, p)
        self.activation = activation
        self.config = dict(
            p=p, d_model=d_model, hidden=hidden, activation=activation
        )

    def _act(self, z: torch.Tensor) -> torch.Tensor:
        return z * z if self.activation == "quad" else F.gelu(z)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.embed(x), self.embed(y)], dim=-1)
        h = self._act(self.fc1(h))
        return self.head(h)


class PCondModMultNet(nn.Module):
    """p-conditioned model for (x * y) mod p across many primes at once.

    Residues x, y and the prime p are embedded; the prime modulates the residue
    representation (FiLM) so one set of weights specialises to each modulus.
    Residual MLP blocks then decode to logits over residue classes [0, p_max).
    """

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
        self.config = dict(
            p_max=p_max, d_model=d_model, hidden=hidden,
            n_blocks=n_blocks, activation=activation,
        )

    def _act(self, z: torch.Tensor) -> torch.Tensor:
        return z * z if self.activation == "quad" else F.gelu(z)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        h = self._act(
            self.fc_in(torch.cat([self.res_embed(x), self.res_embed(y)], dim=-1))
        )
        gamma, beta = self.film(self.p_embed(p)).chunk(2, dim=-1)
        h = h * gamma + beta
        for blk in self.blocks:
            h = h + self._act(blk(h))
        return self.head(h)


def masked_logits(
    logits: torch.Tensor, p: torch.Tensor, p_max: int
) -> torch.Tensor:
    """Mask residue classes >= p (per example) so argmax is always in [0, p)."""
    cols = torch.arange(p_max, device=logits.device).unsqueeze(0)
    invalid = cols >= p.unsqueeze(1)
    return logits.masked_fill(invalid, float("-inf"))
