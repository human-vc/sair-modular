"""Computational verification of the top Tier-3 novelty bets.

Runs three decisive experiments and prints hard PASS/FAIL signals:

  EXP1  opt-stack       Does Grokfast + Omnigrok small-init + two-set repetition
                        break the p=8191 wall? (optimization vs fundamental)
  EXP2  dlog-learnable  Can a net LEARN dlog(x) for a 16-bit prime and generalize
                        to held-out x? (learned dlog path vs baked-table-only)
  EXP3  rns-reduce      Does fixed-moduli RNS + a LEARNED p-conditioned reduction
                        generalize to HELD-OUT primes? (cleanest compliant bet)

Run:  python3 -u verify_tier3.py            # all three
      python3 -u verify_tier3.py --only 3   # one
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def factorize(n: int) -> set[int]:
    f, d = set(), 2
    while d * d <= n:
        while n % d == 0:
            f.add(d)
            n //= d
        d += 1
    if n > 1:
        f.add(n)
    return f


def primitive_root(p: int) -> int:
    fs = factorize(p - 1)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in fs):
            return g
    return -1


def dlog_table(p: int, g: int) -> torch.Tensor:
    t = torch.zeros(p, dtype=torch.long)
    e = 1
    for k in range(p - 1):
        t[e] = k
        e = (e * g) % p
    return t


def grokfast(model, ema: dict, alpha=0.98, lamb=2.0) -> None:
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if n in ema:
            ema[n].mul_(alpha).add_(p.grad.detach(), alpha=1 - alpha)
        else:
            ema[n] = p.grad.detach().clone()
        p.grad.add_(ema[n], alpha=lamb)


def shrink_init(model, factor=0.4) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(factor)


def p_features(p: torch.Tensor, n_freq=32, p_max=65536) -> torch.Tensor:
    pf = p.float().unsqueeze(-1)
    ks = torch.arange(1, n_freq + 1, device=p.device).float()
    ang = 2 * torch.pi * pf * ks / p_max
    return torch.cat([pf / p_max, torch.sin(ang), torch.cos(ang)], dim=-1)


# --------------------------------------------------------------------------- #
# EXP1 — optimization stack on a single 13-bit prime
# --------------------------------------------------------------------------- #
class MultNet(nn.Module):
    def __init__(self, p, d=256, h=2048):
        super().__init__()
        self.emb = nn.Embedding(p, d)
        self.fc1 = nn.Linear(2 * d, h)
        self.fc2 = nn.Linear(h, h)
        self.head = nn.Linear(h, p)

    def forward(self, x, y):
        z = torch.cat([self.emb(x), self.emb(y)], -1)
        z = F.gelu(self.fc1(z))
        z = z + F.gelu(self.fc2(z))
        return self.head(z)


def exp1(dev, p=8191, steps=25000, batch=16384, seed=0):
    print(f"\n===== EXP1 opt-stack | p={p} (Grokfast + small-init + two-set) =====", flush=True)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    pool = torch.randint(0, p * p, (20_000_000,), generator=g, dtype=torch.long)
    rep = pool[:2_000_000]                       # two-set: small repeated subset
    held = torch.unique(torch.randint(0, p * p, (40000,), generator=torch.Generator().manual_seed(7), dtype=torch.long))[:20000]
    pool = pool[~torch.isin(pool, held)]
    rep = rep[~torch.isin(rep, held)]
    pool, rep = pool.to(dev), rep.to(dev)
    tx, ty = (held // p).to(dev), (held % p).to(dev)
    tl = ((held // p) * (held % p) % p).to(dev)

    model = MultNet(p).to(dev)
    shrink_init(model, 0.4)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-4)
    ema, bg = {}, torch.Generator(device=dev).manual_seed(seed)
    best = 0.0
    for s in range(1, steps + 1):
        model.train()
        src = rep if torch.rand(1, generator=bg, device=dev).item() < 0.85 else pool
        idx = torch.randint(0, len(src), (batch,), generator=bg, device=dev)
        c = src[idx]
        x, y, lab = (c // p).to(dev), (c % p).to(dev), ((c // p) * (c % p) % p).to(dev)
        loss = F.cross_entropy(model(x, y), lab)
        opt.zero_grad()
        loss.backward()
        grokfast(model, ema)
        opt.step()
        sched.step()
        if s % 1000 == 0 or s == 1:
            model.eval()
            with torch.no_grad():
                he = (model(tx, ty).argmax(-1) == tl).float().mean().item()
                tr = (model(x, y).argmax(-1) == lab).float().mean().item()
            best = max(best, he)
            print(f"  step {s:6d} | loss {loss.item():.4f} | train {tr:.4f} | held-out {he:.4f}", flush=True)
    verdict = "PASS (wall was optimization!)" if best > 0.5 else ("PARTIAL" if best > 0.05 else "FAIL (fundamental wall)")
    print(f"  >> EXP1 best held-out={best:.4f} -> {verdict}", flush=True)


# --------------------------------------------------------------------------- #
# EXP2 — is the discrete-log map learnable from a sample?
# --------------------------------------------------------------------------- #
class DlogNet(nn.Module):
    def __init__(self, p, d=256, h=2048):
        super().__init__()
        self.emb = nn.Embedding(p, d)
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, h)
        self.head = nn.Linear(h, p - 1)

    def forward(self, x):
        z = F.gelu(self.fc1(self.emb(x)))
        z = z + F.gelu(self.fc2(z))
        return self.head(z)


def exp2(dev, p=65521, steps=8000, batch=8192, seed=0):
    print(f"\n===== EXP2 dlog-learnable | p={p} (can a net learn dlog and generalize?) =====", flush=True)
    torch.manual_seed(seed)
    gr = primitive_root(p)
    tab = dlog_table(p, gr).to(dev)
    xs = torch.arange(1, p)
    perm = xs[torch.randperm(len(xs), generator=torch.Generator().manual_seed(seed))]
    k = len(perm) // 2
    tr_x, te_x = perm[:k].to(dev), perm[k:].to(dev)            # disjoint by construction
    print(f"  primitive root g={gr} | train {len(tr_x)} | held-out {len(te_x)} (chance={1/(p-1):.2e})", flush=True)

    model = DlogNet(p).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    bg = torch.Generator(device=dev).manual_seed(seed)
    best = 0.0
    for s in range(1, steps + 1):
        model.train()
        b = tr_x[torch.randint(0, len(tr_x), (batch,), generator=bg, device=dev)]
        loss = F.cross_entropy(model(b), tab[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 1000 == 0 or s == 1:
            model.eval()
            with torch.no_grad():
                tr = (model(b).argmax(-1) == tab[b]).float().mean().item()
                he = (model(te_x).argmax(-1) == tab[te_x]).float().mean().item()
            best = max(best, he)
            print(f"  step {s:6d} | loss {loss.item():.4f} | train {tr:.4f} | held-out {he:.4f}", flush=True)
    verdict = "LEARNABLE (dlog path open!)" if best > 0.1 else "UNLEARNABLE (dlog must be baked -> rules ruling decides)"
    print(f"  >> EXP2 best held-out={best:.4f} -> {verdict}", flush=True)


# --------------------------------------------------------------------------- #
# EXP3 — RNS decomposition + LEARNED p-conditioned reduction, cross-prime
# --------------------------------------------------------------------------- #
Q = [251, 241, 239, 233, 229]   # fixed coprime moduli, product 7.7e11 > 65535^2


class RnsReduce(nn.Module):
    def __init__(self, p_max=65536, d=96, h=1024, n_freq=32):
        super().__init__()
        self.p_max = p_max
        self.embs = nn.ModuleList(nn.Embedding(q, d) for q in Q)
        pf_dim = 1 + 2 * n_freq
        self.fc1 = nn.Linear(len(Q) * d + pf_dim, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.head = nn.Linear(h, p_max)

    def forward(self, res, p):
        e = torch.cat([emb(res[:, i]) for i, emb in enumerate(self.embs)], -1)
        z = torch.cat([e, p_features(p, p_max=self.p_max)], -1)
        z = F.gelu(self.fc1(z))
        z = z + F.gelu(self.fc2(z))
        z = z + F.gelu(self.fc3(z))
        logits = self.head(z)
        cols = torch.arange(self.p_max, device=logits.device).unsqueeze(0)
        return logits.masked_fill(cols >= p.unsqueeze(1), float("-inf"))


def _sieve(lo, hi):
    s = bytearray([1]) * hi
    s[0] = s[1] = 0
    for i in range(2, int(hi**0.5) + 1):
        if s[i]:
            s[i * i :: i] = bytearray(len(s[i * i :: i]))
    return [i for i in range(lo, hi) if s[i]]


def exp3(dev, steps=25000, batch=8192, seed=0):
    print(f"\n===== EXP3 rns-reduce | cross-prime RNS+learned reduction =====", flush=True)
    torch.manual_seed(seed)
    primes = _sieve(512, 65536)
    rng = torch.Generator().manual_seed(seed)
    primes = [primes[i] for i in torch.randperm(len(primes), generator=rng).tolist()]
    train_p = torch.tensor(primes[: len(primes) - 200], device=dev)
    held_p = torch.tensor(primes[len(primes) - 200 :], device=dev)
    Qt = torch.tensor(Q, device=dev)
    print(f"  primes: train {len(train_p)} | held-out {len(held_p)}", flush=True)

    def batch_from(pset, n, gen):
        p = pset[torch.randint(0, len(pset), (n,), generator=gen, device=dev)]
        x = (torch.rand(n, generator=gen, device=dev) * p.float()).long()
        y = (torch.rand(n, generator=gen, device=dev) * p.float()).long()
        prod = x * y
        res = (prod.unsqueeze(1) % Qt.unsqueeze(0))
        return res, p, (prod % p)

    model = RnsReduce().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)
    bg = torch.Generator(device=dev).manual_seed(seed)
    ge = torch.Generator(device=dev).manual_seed(999)
    best = 0.0
    for s in range(1, steps + 1):
        model.train()
        res, p, lab = batch_from(train_p, batch, bg)
        loss = F.cross_entropy(model(res, p), lab)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % 1000 == 0 or s == 1:
            model.eval()
            with torch.no_grad():
                rtr, ptr, ltr = batch_from(train_p, 8192, ge)
                rte, pte, lte = batch_from(held_p, 8192, ge)
                tr = (model(rtr, ptr).argmax(-1) == ltr).float().mean().item()
                he = (model(rte, pte).argmax(-1) == lte).float().mean().item()
            best = max(best, he)
            print(f"  step {s:6d} | loss {loss.item():.4f} | train-primes {tr:.4f} | HELD-OUT-primes {he:.4f}", flush=True)
    verdict = "PASS (cross-prime reduction learns!)" if best > 0.5 else ("PARTIAL" if best > 0.05 else "FAIL")
    print(f"  >> EXP3 best held-out-prime={best:.4f} -> {verdict}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=0, help="run only EXP n (0=all)")
    args = ap.parse_args()
    dev = device()
    print(f"device={dev}", flush=True)
    if args.only in (0, 1):
        exp1(dev)
    if args.only in (0, 2):
        exp2(dev)
    if args.only in (0, 3):
        exp3(dev)
    print("\nverify_tier3 done.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
