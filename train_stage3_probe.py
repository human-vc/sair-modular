"""Stage 3 feasibility probe: single large-prime modular multiplication.

Tier 3's wall: per-prime tables (~4.3e9 at p=65521) can't be enumerated, so a
model must learn the operation from a small fraction of the table and generalise
to held-out pairs that are *disjoint* from training. This isolates that
sub-problem on one prime.

  held-out exact rises    -> the structure is learnable from a sample (Tier 3 alive)
  train fits, held-out ~0  -> memorisation only (the documented 16-bit wall)
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from model import ModMultNet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=8191)
    ap.add_argument("--pool", type=int, default=20_000_000)
    ap.add_argument("--held", type=int, default=20_000)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=2000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = args.p

    g = torch.Generator().manual_seed(args.seed)
    pool = torch.randint(0, P * P, (args.pool,), generator=g, dtype=torch.long)
    gte = torch.Generator().manual_seed(2024)
    held = torch.unique(
        torch.randint(0, P * P, (args.held * 2,), generator=gte, dtype=torch.long)
    )[: args.held]
    pool = pool[~torch.isin(pool, held)]

    pxc, pyc = pool // P, pool % P
    txc, tyc = held // P, held % P
    px, py, pl = pxc.to(device), pyc.to(device), ((pxc * pyc) % P).to(device)
    tx, ty, tl = txc.to(device), tyc.to(device), ((txc * tyc) % P).to(device)

    model = ModMultNet(P, args.d_model, args.hidden).to(device)
    print(
        f"device={device} p={P} pool={len(pool):,} frac={len(pool) / P**2:.3e} "
        f"held={len(held):,} params={sum(t.numel() for t in model.parameters()):,}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)
    loss_fn = nn.CrossEntropyLoss()
    bg = torch.Generator(device=device).manual_seed(args.seed)
    n = len(pool)

    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(0, n, (args.batch,), generator=bg, device=device)
        loss = loss_fn(model(px[idx], py[idx]), pl[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                tr = (model(px[idx], py[idx]).argmax(-1) == pl[idx]).float().mean().item()
                he = (model(tx, ty).argmax(-1) == tl).float().mean().item()
            print(
                f"step {step:7d} | loss {loss.item():.4f} | train {tr:.4f} | held-out {he:.4f}",
                flush=True,
            )
    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
