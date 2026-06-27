# Texas Hold-em-poker

An independent heads-up Texas Hold'em virtual duel model with a compact DQN
training loop. The project is designed for course-demo and presentation use:
it contains a runnable poker environment, baseline agents, training scripts,
evaluation, and a text demo of one virtual hand.

## What is included

- Heads-up no-limit Texas Hold'em environment with blinds, betting streets,
  fold/check-call/raise/all-in actions, showdown, and zero-sum payoffs.
- Five-action abstraction inspired by RLCard's no-limit Hold'em environment.
- PyTorch DQN agent with CUDA support, legal-action masking, replay memory,
  target network, checkpoint save/load, and epsilon-greedy exploration.
- Random and simple rule-based opponents.
- CLI scripts for training, evaluating, and showing one virtual duel.

## Install

First check whether PyTorch is already installed in the Python environment you
will use:

```bash
python -c "import importlib.util; print(importlib.util.find_spec('torch') is not None)"
```

If it prints `True`, do not reinstall PyTorch. Check CUDA visibility directly:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If PyTorch is missing or CPU-only and you want GPU training on an NVIDIA card,
install the CUDA wheel once:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then install this project in editable mode:

```bash
python -m pip install -e ".[dev]"
```

## Train

```bash
python scripts/train_dqn.py --episodes 1000 --eval-games 100 --device auto --checkpoint checkpoints/dqn.pt
```

Force CUDA and fail fast if the GPU is not available:

```bash
python scripts/train_dqn.py --episodes 1000 --eval-games 100 --device cuda --require-cuda --checkpoint checkpoints/dqn.pt
```

## Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/dqn.pt --games 200 --device auto
```

## Demo

```bash
python scripts/play_demo.py --checkpoint checkpoints/dqn.pt --seed 7 --device auto
```

The demo prints the hole cards, community cards, actions, pot movement, and
final payoff for a single virtual hand.

## Relationship to RLCard

This repository does not depend on `rlcard` at runtime. It draws design
inspiration from the RLCard project, especially the idea of a compact card-game
environment interface and an abstract no-limit Hold'em action set:

- `fold`
- `check_call`
- `raise_half_pot`
- `raise_pot`
- `all_in`

RLCard is available at https://github.com/datamllab/rlcard and is distributed
under the MIT License. This project is a smaller independent implementation
focused only on a Texas Hold'em virtual duel model.

## Test

```bash
python -m pytest
```
