"""Grok time-scaling measurement for (x*y) mod p (single prime, fixed 30% fraction).

Finding so far: time-to-grok scales super-linearly in p. p=8191 ignites ~16k, groks
20k; p=16381 (trunk doubled to d512/h4096) stayed in a bad basin until ~90k, then the
loss cliff began (9.46->5.92 over 90k-100k) and the first run was KILLED mid-ignition.
So the binding Tier-3 constraint is TRAINING TIME, not capacity or data fraction.

This harness now: (1) checkpoints + auto-resumes (no from-scratch restarts), (2) reports
CENSORED when loss is still descending at termination instead of a fake grok_step=-1,
(3) logs train-batch AND held-out accuracy. Goal: resolve p=16381 to a real grok_step,
pair with p=8191's 20k -> scaling exponent -> extrapolate to 65521.

  python3 -u grok_sweep.py --steps 180000     # resumes if a checkpoint exists
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from model import ModMultNet

# (p, pool_size, d_model, hidden, lr)
CELLS = [
    (16381, 80_000_000, 512, 4096, 2e-3),   # resolve to a real grok_step (was censored at 100k)
]


def run_cell(p, pool_n, d, h, lr, dev, steps, batch=16384, wd=0.2, seed=0, save_every=10000):
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
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    ckpt = Path(__file__).resolve().parent / f"grok_ckpt_p{p}_d{d}.pt"
    start, grok_step, best, hist = 1, -1, 0.0, []
    if ckpt.exists():
        c = torch.load(ckpt, map_location=dev)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        start, grok_step, best, hist = c["step"] + 1, c["grok_step"], c["best"], c["hist"]
        print(f"  [resumed from step {c['step']}]", flush=True)

    npar = sum(t.numel() for t in model.parameters())
    trunk = sum(t.numel() for n, t in model.named_parameters() if "emb" not in n and "head" not in n)
    print(f"--- p={p} pool={len(pool):,} frac={frac:.3e} d={d} h={h} lr={lr} params={npar:,} trunk={trunk:,} ---", flush=True)

    bg = torch.Generator(device=dev).manual_seed(seed + start)
    n = len(pool)
    for s in range(start, steps + 1):
        model.train()
        idx = torch.randint(0, n, (batch,), generator=bg, device=dev)
        loss = F.cross_entropy(model(px[idx], py[idx]), pl[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 2000 == 0 or s == start:
            model.eval()
            with torch.no_grad():
                tr = (model(px[idx], py[idx]).argmax(-1) == pl[idx]).float().mean().item()
                he = (model(tx, ty).argmax(-1) == tl).float().mean().item()
            best = max(best, he)
            hist.append((s, loss.item(), he))
            if grok_step < 0 and he > 0.9:
                grok_step = s
            print(f"  p={p} step {s:6d} | loss {loss.item():.4f} | train {tr:.4f} | held-out {he:.4f}", flush=True)
            if he > 0.98:
                break
        if s % save_every == 0 or s == steps:
            torch.save({"step": s, "model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "grok_step": grok_step, "best": best, "hist": hist}, ckpt)

    if grok_step > 0:
        verdict = f"grok_step={grok_step}"
    elif len(hist) >= 6 and hist[-1][1] < hist[-6][1] - 0.05:
        verdict = "CENSORED (loss still descending — extend steps)"
    else:
        verdict = "PLATEAU (loss flat at termination)"
    print(f"  >> p={p} d={d} h={h} frac={frac:.2e}: best held-out={best:.4f} -> {verdict}", flush=True)
    return dict(p=p, d=d, h=h, frac=frac, best=best, grok_step=grok_step, verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=180000)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}", flush=True)
    res = [run_cell(p, n, d, h, lr, dev, args.steps) for (p, n, d, h, lr) in CELLS]
    print("\n===== SUMMARY =====", flush=True)
    for r in res:
        print(f"  p={r['p']:6d} d={r['d']:4d} h={r['h']:5d} best={r['best']:.4f} -> {r['verdict']}", flush=True)
    print("\ngrok_sweep done.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
