"""Grokking-boundary sweep for (x*y) mod p — the experiment that should have come first.

Context/correction: probe_8191b GROKKED p=8191 with plain AdamW (d256/h2048, wd=0.2,
lr=2e-3, 30% table). The "opt-stack" EXP1 only failed because Grokfast destabilized it.
So 13-bit is learnable. The real Tier-3 question: how does the minimal training set
needed to grok scale with p, and does it stay feasible up to p~65521?

Each cell trains the *recipe that grokked* (vanilla AdamW, no Grokfast/two-set) on a
fixed pool of given size, reports the grok step (first held-out > 0.9). Disjoint
held-out by construction. Edit CELLS to extend.

  python3 -u grok_sweep.py            # default cells
  python3 -u grok_sweep.py --steps 120000
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModMultNet

# (prime, pool_size). 8191 fraction sweep nails the 13-bit boundary; 16381 tests the
# scaling direction at the same 30% fraction. Extend to 32749/65521 once trend is clear.
CELLS = [
    (8191, 20_000_000),   # ~30% — confirm grok (re-establish ground truth)
    (8191, 13_400_000),   # ~20%
    (8191, 6_700_000),    # ~10% — lower boundary at 13-bit
    (16381, 80_000_000),  # ~30% at 14-bit — does grok survive a prime step up?
]


def run_cell(p, pool_n, dev, steps, batch=16384, d=256, h=2048, wd=0.2, lr=2e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    pool = torch.randint(0, p * p, (pool_n,), generator=g, dtype=torch.long)
    held = torch.unique(
        torch.randint(0, p * p, (40000,), generator=torch.Generator().manual_seed(7), dtype=torch.long)
    )[:20000]
    pool = pool[~torch.isin(pool, held)]
    px, py, pl = (pool // p).to(dev), (pool % p).to(dev), ((pool // p) * (pool % p) % p).to(dev)
    tx, ty, tl = (held // p).to(dev), (held % p).to(dev), ((held // p) * (held % p) % p).to(dev)
    frac = len(pool) / (p * p)

    model = ModMultNet(p, d, h).to(dev)
    npar = sum(t.numel() for t in model.parameters())
    print(f"\n--- cell p={p} pool={len(pool):,} frac={frac:.3e} params={npar:,} ---", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    bg = torch.Generator(device=dev).manual_seed(seed)
    n = len(pool)
    grok_step, best = -1, 0.0

    for s in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n, (batch,), generator=bg, device=dev)
        loss = F.cross_entropy(model(px[idx], py[idx]), pl[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % 2000 == 0 or s == 1:
            model.eval()
            with torch.no_grad():
                he = (model(tx, ty).argmax(-1) == tl).float().mean().item()
            best = max(best, he)
            if grok_step < 0 and he > 0.9:
                grok_step = s
            print(f"  p={p} step {s:6d} | loss {loss.item():.4f} | held-out {he:.4f}", flush=True)
            if he > 0.98:
                break
    print(f"  >> p={p} frac={frac:.2e}: best held-out={best:.4f} grok_step={grok_step}", flush=True)
    return dict(p=p, pool=len(pool), frac=frac, best=best, grok_step=grok_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100000)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}", flush=True)
    res = [run_cell(p, n, dev, args.steps) for (p, n) in CELLS]
    print("\n===== GROK-BOUNDARY SUMMARY =====", flush=True)
    for r in res:
        print(
            f"  p={r['p']:6d} pool={r['pool']:>11,} frac={r['frac']:.2e} "
            f"best={r['best']:.4f} grok_step={r['grok_step']}",
            flush=True,
        )
    print("\ngrok_sweep done.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
