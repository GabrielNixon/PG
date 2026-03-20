# PG

A modular prototype for combining a Hidden Markov Model (HMM) with multiple RNN experts for regime-aware sequence prediction.

---

## Overview

This project explores a hybrid architecture consisting of:

- **2 latent HMM states**
- **2 separate RNN experts**
- A **soft routing mechanism** based on latent-state responsibilities
- Synthetic sequence data with regime switches

The goal is to understand how latent regimes can guide expert specialization, and how EM-style inference should interact with gradient-based neural network training.

---

## Core Idea

At each time step:

1. Each RNN expert predicts action probabilities  
2. These predictions define expert-wise likelihoods of the observed action  
3. The HMM provides a latent prior and temporal structure  
4. Responsibilities are computed from **prior × expert likelihood**  
5. Experts are trained using **responsibility-weighted losses**

This framework is inspired by:

- Hidden Markov Models (HMMs)
- Mixtures of Experts
- EM-style latent inference
- Jordan & Jacobs responsibility weighting

---

## Current Status

### Implemented

- Modular project structure
- Single RNN expert (GRU-based)
- HMM parameter container
- Multi-expert mixture wrapper
- Synthetic sequence generator
- Forward-backward posterior computation (log-space)
- Responsibility-weighted expert loss
- Debug visualization for controlled regimes

---

### Key Findings

- Posterior-weighted training alone leads to:
  - **Flat posteriors**
  - **Nearly identical experts**

- Responsibility-weighted loss:
  - Improves **expert specialization**

- Without proper HMM updates:
  - Latent routing can **collapse to one dominant state**

---

### Next Step (Critical)

Implement a proper **HMM EM update**:

- Compute:
  - `gamma` (state posterior)
  - `xi` (transition posterior)
- Update:
  - Initial distribution `π`
  - Transition matrix `A`
- Keep:
  - RNN training via gradient descent

---

## Repository Structure

```text
pg/
├── README.md
├── .gitignore
├── configs/
│   └── default_config.py
├── data/
│   ├── __init__.py
│   └── synthetic.py
├── models/
│   ├── __init__.py
│   ├── hmm.py
│   ├── posterior.py
│   ├── rnn_expert.py
│   └── mixture_model.py
├── training/
│   ├── __init__.py
│   ├── losses.py
│   ├── em.py
│   └── trainer.py
├── scripts/
│   └── train.py
├── tests/
│   └── test_shapes.py
├── notebooks/
│   └── debug.py
└── outputs/
    └── debug_plots/

## Core Components

### `models/rnn_expert.py`
Defines a GRU-based expert mapping sequence history → action probabilities.

### `models/hmm.py`
Stores HMM parameters:
- Initial state distribution `π`
- Transition matrix `A`  
(Initialized with sticky transitions)

### `models/mixture_model.py`
Wraps:
- HMM
- Multiple RNN experts
- Emission likelihood computation
- Posterior-weighted prediction

### `models/posterior.py`
Implements:
- Log-space forward-backward algorithm
- Posterior state probabilities

### `data/synthetic.py`
Generates synthetic sequences with:
- Latent regimes
- State-dependent action rules
- Oracle state labels for debugging

### `training/losses.py`
Includes:
- Standard NLL loss
- Responsibility-weighted expert loss

### `training/trainer.py`
Handles training and logs:
- Loss
- State accuracy
- Average state mass
- Expert gap

### `notebooks/debug.py`
Visualizes:
- True latent states
- Posterior assignments
- Signals and actions

---

## Synthetic Data

Environment setup:
- 2 latent states
- Sticky transitions (high persistence)
- State-dependent action rules

Debug mode includes explicit regime blocks:

```text
0 → 1 → 0 → 1
Used to verify whether inferred responsibilities track true regimes.

---

## Training Setup

- 2 latent states  
- 2 RNN experts  
- Adam optimizer  
- Responsibility-weighted updates  

### Diagnostics

- `loss`
- `acc_direct`
- `acc_flipped`
- `best_acc`
- `avg_state_mass`
- `expert_gap`

---

## Key Insight

> Good predictive performance does not imply meaningful latent state recovery.

### Observed Behavior

- Similar experts → flat posterior  
- Over-competition → state collapse  

Proper separation needed between:
- HMM updates (EM)
- RNN updates (GD)
