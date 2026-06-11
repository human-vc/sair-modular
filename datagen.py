"""Synthetic data for modular multiplication.

Tables are every (x, y) pair with label (x * y) mod p. This is ground-truth
label generation for training; the model never sees it as an algorithm. The
Tier-2 prime range mirrors config.TIERS in the official harness: [2^4, 2^8).
"""

from __future__ import annotations

import torch


def full_table(p: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.arange(p).repeat_interleave(p)
    ys = torch.arange(p).repeat(p)
    labels = (xs * ys) % p
    return xs, ys, labels


def split_indices(
    n: int, train_frac: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    k = int(n * train_frac)
    return perm[:k], perm[k:]


def _sieve(limit: int) -> list[int]:
    is_p = bytearray([1]) * limit
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i * i :: i] = bytearray(len(is_p[i * i :: i]))
    return [i for i in range(2, limit) if is_p[i]]


def tier2_primes() -> list[int]:
    """The ~48 primes in [16, 256) = [2^4, 2^8), the Tier-2 range."""
    return [p for p in _sieve(256) if 16 <= p < 256]


def build_tier2_dataset(
    primes: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full enumerated (x, y, p) -> (x*y) mod p table over the given primes."""
    xs, ys, ps, labels = [], [], [], []
    for p in primes:
        x = torch.arange(p).repeat_interleave(p)
        y = torch.arange(p).repeat(p)
        xs.append(x)
        ys.append(y)
        ps.append(torch.full((p * p,), p, dtype=torch.long))
        labels.append((x * y) % p)
    if not primes:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty, empty
    return torch.cat(xs), torch.cat(ys), torch.cat(ps), torch.cat(labels)
