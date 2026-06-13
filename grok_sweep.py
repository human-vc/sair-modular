"""Grok capacity/boundary sweep for (x*y) mod p.

Corrected understanding: the p=16381 "failure" in the first sweep was a CONFOUND —
the trunk (d_model, hidden) was held fixed at 256/2048 while only the vocab tables
grew with p, so the model was under-capacity for the larger circuit (loss never left
the chance floor: 9.72->9.60 vs ln(16381)=9.70). Epochs were fine (~20 > the ~18 that
grokked p=8191). So the real Tier-3 question is REQUIRED TRUNK CAPACITY vs p: measure
the trunk needed to grok at each p, extrapolate to 65521, compare to the budget.

Each cell specifies its own (p, pool, d_model, hidden, lr). Recipe is the one that
grokked: vanilla AdamW, wd=0.2, cosine. Logs train-batch AND held-out accuracy.

  python3 -u grok_sweep.py
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from model import ModMultNet

# (p, pool_size, d_model, hidden, lr)
CELLS = [
    (8191, 20_000_000, 256, 2048, 2e-3),    # control: known-good, must grok ~25k
    (16381, 80_000_000, 512, 4096, 2e-3),   # capacity test: DOUBLED trunk at 14-bit
]


def run_cell(p, pool_n, d, h, lr, dev, steps, batch=16384, wd=0.2, seed=0):
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
    trunk = sum(t.numel() for n, t in model.named_parameters() if "emb" not in n and "head" not in n)
    print(
        f"\n--- p={p} pool={len(pool):,} frac={frac:.3e} d={d} h={h} lr={lr} "
        f"params={npar:,} trunk={trunk:,} ---",
        flush=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    bg = torch.Generator(device=dev).manual_seed(seed)
    n = len(pool)
    grok_step, best = -1, 0.0

    for s in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n, (batch,), generator=bg, device=dev)
        out = model(px[idx], py[idx])
        loss = F.cross_entropy(out, pl[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % 2000 == 0 or s == 1:
            model.eval()
            with torch.no_grad():
                tr = (model(px[idx], py[idx]).argmax(-1) == pl[idx]).float().mean().item()
                he = (model(tx, ty).argmax(-1) == tl).float().mean().item()
            best = max(best, he)
            if grok_step < 0 and he > 0.9:
                grok_step = s
            print(
                f"  p={p} d={d} step {s:6d} | loss {loss.item():.4f} | "
                f"train {tr:.4f} | held-out {he:.4f}",
                flush=True,
            )
            if he > 0.98:
                break
    print(f"  >> p={p} d={d} h={h} frac={frac:.2e}: best held-out={best:.4f} grok_step={grok_step}", flush=True)
    return dict(p=p, d=d, h=h, frac=frac, best=best, grok_step=grok_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100000)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}", flush=True)
    res = [run_cell(p, n, d, h, lr, dev, args.steps) for (p, n, d, h, lr) in CELLS]
    print("\n===== CAPACITY SWEEP SUMMARY =====", flush=True)
    for r in res:
        print(
            f"  p={r['p']:6d} d={r['d']:4d} h={r['h']:5d} frac={r['frac']:.2e} "
            f"best={r['best']:.4f} grok_step={r['grok_step']}",
            flush=True,
        )
    print("\ngrok_sweep done.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
