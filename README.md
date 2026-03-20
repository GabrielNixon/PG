# pg

A modular HMM-gated multi-RNN project.

## Planned structure

- `models/hmm.py` — latent-state model
- `models/rnn_expert.py` — one RNN expert
- `models/mixture_model.py` — combines experts using HMM posterior
- `training/em.py` — E-step / M-step utilities
- `training/losses.py` — weighted losses
- `training/trainer.py` — training loop
- `scripts/train.py` — entry script
- `tests/test_shapes.py` — shape sanity checks