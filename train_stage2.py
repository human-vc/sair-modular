"""Stage 2: one p-conditioned model over all Tier-2 primes (the go/no-go).

Trains on the full enumerated (x, y, p) -> (x*y) mod p table for every prime in
[16, 256). Held-out residue pairs measure structure-vs-memorisation; held-out
whole primes measure cross-prime generalisation (a Tier-3 readiness probe).
For the final submission, train on everything: --holdout_frac 0 --holdout_primes 0.

    python train_stage2.py                       # dev go/no-go
    python train_stage2.py --steps 400000 --holdout_frac 0 --holdout_primes 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from datagen import build_tier2_dataset, tier2_primes
from model import PCondModMultNet, masked_logits


@torch.no_grad()
def exact_match(model, xs, ys, ps, labels, p_max, device, batch=131072) -> float:
    n = len(labels)
    if n == 0:
        return float("nan")
    model.eval()
    correct = 0
    for i in range(0, n, batch):
        sl = slice(i, i + batch)
        pb = ps[sl].to(device)
        logits = model(xs[sl].to(device), ys[sl].to(device), pb)
        pred = masked_logits(logits, pb, p_max).argmax(-1).cpu()
        correct += (pred == labels[sl]).sum().item()
    return correct / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--n_blocks", type=int, default=3)
    ap.add_argument("--activation", default="gelu", choices=["gelu", "quad"])
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--holdout_frac", type=float, default=0.05)
    ap.add_argument("--holdout_primes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--target", type=float, default=0.90)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    p_max = 256

    primes = tier2_primes()
    g = torch.Generator().manual_seed(args.seed)
    primes = [primes[i] for i in torch.randperm(len(primes), generator=g).tolist()]
    held_primes = primes[: args.holdout_primes]
    train_primes = primes[args.holdout_primes :]
    print(
        f"tier-2 primes: {len(primes)} | train {len(train_primes)} | "
        f"held-out primes {sorted(held_primes)}"
    )

    xs, ys, ps, labels = build_tier2_dataset(train_primes)
    perm = torch.randperm(len(labels), generator=g)
    k = int(len(labels) * (1 - args.holdout_frac))
    tr_idx, te_idx = perm[:k], perm[k:]
    hxs, hys, hps, hl = build_tier2_dataset(held_primes)
    print(f"train pairs {len(tr_idx):,} | held-out pairs {len(te_idx):,} | held-out-prime pairs {len(hl):,}")

    xs_tr = xs[tr_idx].to(device)
    ys_tr = ys[tr_idx].to(device)
    ps_tr = ps[tr_idx].to(device)
    y_tr = labels[tr_idx].to(device)

    model = PCondModMultNet(
        p_max, args.d_model, args.hidden, args.n_blocks, args.activation
    ).to(device)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"device={device} params={n_params:,} batch={args.batch} wd={args.wd} lr={args.lr}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1
    )
    loss_fn = nn.CrossEntropyLoss()
    bg = torch.Generator().manual_seed(args.seed)

    out = Path(__file__).resolve().parent / "stage2_tier2.pt"
    best = 0.0
    for step in range(1, args.steps + 1):
        model.train()
        sel = torch.randint(0, len(tr_idx), (args.batch,), generator=bg).to(device)
        ml = masked_logits(model(xs_tr[sel], ys_tr[sel], ps_tr[sel]), ps_tr[sel], p_max)
        loss = loss_fn(ml, y_tr[sel])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == 1:
            pair = exact_match(model, xs[te_idx], ys[te_idx], ps[te_idx], labels[te_idx], p_max, device)
            prime = exact_match(model, hxs, hys, hps, hl, p_max, device)
            print(
                f"step {step:7d} | loss {loss.item():.4f} | "
                f"held-out pairs {pair:.4f} | held-out primes {prime:.4f}"
            )
            if pair > best:
                best = pair
                torch.save({"state_dict": model.state_dict(), "config": model.config}, out)
            if len(te_idx) and pair >= args.target:
                print(f"TIER-2 GO: held-out pairs {pair:.4f} >= {args.target} at step {step}")
                break

    print(f"done. best held-out-pair exact={best:.4f}. saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
