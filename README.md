# sair-modular

Development repo for the SAIR Modular Arithmetic Challenge — learning to compute
`(a * b) mod p` from trained parameters. The official harness lives separately
at `~/modular-arithmetic-challenge`; submissions are pushed to Hugging Face.

## Stages
- **Stage 1** — grok exact `(x * y) mod p` for a single fixed prime. Validates
  the architecture and the exact-match loop. `train_stage1.py`.
- **Stage 2** — one p-conditioned model over all Tier-2 primes, scored on the
  official harness. (next)
- **Stage 3** — novelty bake-off (p-conditioned dlog / learned RNS-CRT /
  hypernet) and the Tier-3 push. (next)

## Setup
```
python3.12 -m venv .venv
.venv/bin/pip install torch numpy
.venv/bin/python train_stage1.py --p 97
```
