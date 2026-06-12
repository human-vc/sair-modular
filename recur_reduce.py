"""Novel Tier-3 route: learn the ALGORITHM, not the function.

A recurrent net performs base-B long division of n by p, one digit at a time.
Each step is a small bounded primitive -- estimate the quotient digit q = (running
remainder) // p in [0,B) -- which (unlike `n mod p` directly) is low-frequency and
plausibly learnable. The recurrence chains these into an exact reduction; per-step
quotient supervision induces the long-division circuit. Compliant: the loop is
scaffolding, the quotient/remainder tracking lives in trained weights.

Crux question: does it generalize to HELD-OUT primes? If yes, bolt onto learned
multiplication (DFST/Neural-GPU prove that half) and Tier 3 is alive.

  python3 -u recur_reduce.py
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = 16
N_DIG = 8          # n = x*y < p^2 < 2^32 = 16^8
P_DIG = 4          # p < 2^16 = 16^4
A_DIG = 4          # answer < p


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def digits(n: torch.Tensor, width: int, base=BASE) -> torch.Tensor:
    """MSB-first base-`base` digits of n -> (B, width)."""
    out = []
    for i in range(width - 1, -1, -1):
        out.append((n // (base ** i)) % base)
    return torch.stack(out, dim=1)


def longdiv_labels(n, p, n_dig=N_DIG, base=BASE):
    """Vectorized base-B long division: returns (quotient_digits (B,n_dig), remainder (B,))."""
    r = torch.zeros_like(n)
    qs = []
    for i in range(n_dig):
        d = (n // (base ** (n_dig - 1 - i))) % base
        r = r * base + d
        qi = r // p
        r = r % p
        qs.append(qi)
    return torch.stack(qs, dim=1), r


class RecurReducer(nn.Module):
    def __init__(self, base=BASE, d=192, hidden=512):
        super().__init__()
        self.base = base
        self.n_emb = nn.Embedding(base, d)
        self.n_pos = nn.Embedding(N_DIG, d)
        self.p_emb = nn.Embedding(base, d)
        self.p_pos = nn.Embedding(P_DIG, d)
        self.p_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.cell = nn.GRUCell(2 * d, hidden)
        self.q_head = nn.Linear(hidden, base)                 # per-step quotient digit (aux)
        self.a_head = nn.Linear(hidden, A_DIG * base)         # final answer digits

    def encode_p(self, p_digs):
        z = self.p_emb(p_digs) + self.p_pos(torch.arange(P_DIG, device=p_digs.device))
        return self.p_mlp(z.mean(1))

    def forward(self, n_digs, p_digs):
        pv = self.encode_p(p_digs)
        h = torch.zeros(n_digs.size(0), self.cell.hidden_size, device=n_digs.device)
        qlog = []
        for t in range(N_DIG):
            x = torch.cat([self.n_emb(n_digs[:, t]) + self.n_pos.weight[t], pv], dim=-1)
            h = self.cell(x, h)
            qlog.append(self.q_head(h))
        q_logits = torch.stack(qlog, dim=1)                   # (B, N_DIG, base)
        a_logits = self.a_head(h).view(-1, A_DIG, self.base)  # (B, A_DIG, base)
        return q_logits, a_logits


def _sieve(lo, hi):
    s = bytearray([1]) * hi
    s[0] = s[1] = 0
    for i in range(2, int(hi**0.5) + 1):
        if s[i]:
            s[i * i :: i] = bytearray(len(s[i * i :: i]))
    return [i for i in range(lo, hi) if s[i]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--q_aux", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1000)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = device()

    primes = _sieve(512, 65536)
    g = torch.Generator().manual_seed(args.seed)
    primes = [primes[i] for i in torch.randperm(len(primes), generator=g).tolist()]
    train_p = torch.tensor(primes[:-200], device=dev)
    held_p = torch.tensor(primes[-200:], device=dev)
    print(f"device={dev} | train primes {len(train_p)} | held-out primes {len(held_p)}", flush=True)

    def make(pset, n, gen):
        p = pset[torch.randint(0, len(pset), (n,), generator=gen, device=dev)]
        x = (torch.rand(n, generator=gen, device=dev) * p.float()).long()
        y = (torch.rand(n, generator=gen, device=dev) * p.float()).long()
        prod = x * y
        qd, rem = longdiv_labels(prod, p)
        return digits(prod, N_DIG), digits(p, P_DIG), qd, rem

    model = RecurReducer().to(dev)
    print(f"params={sum(t.numel() for t in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)
    bg = torch.Generator(device=dev).manual_seed(args.seed)
    ge = torch.Generator(device=dev).manual_seed(99)
    pw = (BASE ** torch.arange(A_DIG - 1, -1, -1, device=dev))  # place values, MSB-first

    @torch.no_grad()
    def evalp(pset):
        model.eval()
        nd, pd, qd, rem = make(pset, 8192, ge)
        _, al = model(nd, pd)
        pred = (al.argmax(-1) * pw).sum(1)
        return (pred == rem).float().mean().item()

    best = 0.0
    for s in range(1, args.steps + 1):
        model.train()
        nd, pd, qd, rem = make(train_p, args.batch, bg)
        ad = digits(rem, A_DIG)
        ql, al = model(nd, pd)
        loss = F.cross_entropy(al.reshape(-1, BASE), ad.reshape(-1)) + \
            args.q_aux * F.cross_entropy(ql.reshape(-1, BASE), qd.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % args.eval_every == 0 or s == 1:
            tr, he = evalp(train_p), evalp(held_p)
            best = max(best, he)
            print(f"step {s:6d} | loss {loss.item():.4f} | train-primes {tr:.4f} | HELD-OUT-primes {he:.4f}", flush=True)
    verdict = "PASS (learned recurrent reduction generalizes!)" if best > 0.5 else ("PARTIAL" if best > 0.05 else "FAIL")
    print(f">> recur-reduce best held-out-prime={best:.4f} -> {verdict}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
