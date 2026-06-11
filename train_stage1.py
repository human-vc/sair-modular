"""Stage 1: grok exact (x * y) mod p for a single fixed prime.

Gate: held-out exact-match >= --target. This validates that the architecture,
training loop, and exact-match evaluation all work before we add p-conditioning
(Stage 2). Defaults follow the grokking literature (AdamW, strong weight decay,
full-batch) and reliably grok p<=251 on CPU.

    python train_stage1.py --p 97
    python train_stage1.py --p 251 --batch 8192 --steps 80000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from datagen import full_table, split_indices
from model import ModMultNet


@torch.no_grad()
def exact_match(model, xs, ys, labels, idx, device, batch=65536) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(idx), batch):
        j = idx[i : i + batch]
        pred = model(xs[j].to(device), ys[j].to(device)).argmax(-1).cpu()
        correct += (pred == labels[j]).sum().item()
    return correct / len(idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--activation", default="gelu", choices=["gelu", "quad"])
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=0, help="0 = full batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=float, default=0.99)
    ap.add_argument("--eval_every", type=int, default=500)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    xs, ys, labels = full_table(args.p)
    tr, te = split_indices(len(labels), args.train_frac, args.seed)
    print(f"p={args.p} table={len(labels)} train={len(tr)} held-out={len(te)}")

    model = ModMultNet(args.p, args.d_model, args.hidden, args.activation).to(device)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"params={n_params:,} act={args.activation} wd={args.wd} lr={args.lr}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.CrossEntropyLoss()

    xs_tr, ys_tr, y_tr = xs[tr].to(device), ys[tr].to(device), labels[tr].to(device)
    g = torch.Generator().manual_seed(args.seed)

    best, best_step = 0.0, 0
    out = Path(__file__).resolve().parent / f"stage1_p{args.p}.pt"

    for step in range(1, args.steps + 1):
        model.train()
        if args.batch and args.batch < len(tr):
            sel = torch.randint(0, len(tr), (args.batch,), generator=g)
            xb, yb, yl = xs_tr[sel], ys_tr[sel], y_tr[sel]
        else:
            xb, yb, yl = xs_tr, ys_tr, y_tr
        loss = loss_fn(model(xb, yb), yl)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == 1:
            tr_acc = exact_match(model, xs, ys, labels, tr, device)
            te_acc = exact_match(model, xs, ys, labels, te, device)
            print(
                f"step {step:6d} | loss {loss.item():.4f} | "
                f"train {tr_acc:.4f} | held-out {te_acc:.4f}"
            )
            if te_acc > best:
                best, best_step = te_acc, step
                torch.save(
                    {"state_dict": model.state_dict(), "config": model.config}, out
                )
            if te_acc >= args.target:
                print(f"GROKKED at step {step}: held-out exact {te_acc:.4f}")
                break

    print(f"done. best held-out exact={best:.4f} @ step {best_step}. saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
