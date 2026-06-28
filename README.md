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
- Local Web table for one human player against three AI opponents, including
  action buttons, AI thinking, Q-value bars, action history, and win/loss trend.

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

For a stronger CUDA run, use the Dueling Double-DQN defaults with a larger
replay buffer, slower exploration decay, mixed random/rule-based opponents, and
less frequent gradient updates:

```bash
python scripts/train_dqn.py --episodes 200000 --eval-games 5000 --device cuda --require-cuda --checkpoint checkpoints/dqn.pt --batch-size 128 --replay-start-size 4096 --replay-size 100000 --hidden-size 256 --learning-rate 0.0003 --epsilon-decay 600000 --epsilon-end 0.05 --target-update 1000 --update-every 8 --opponent-pool random,rule --evaluation-opponents random,rule --all-in-margin 0.25 --exploration-all-in-probability 0.02 --progress-every 5000
```

Training prints a live progress bar by default:

```text
train [############----------------]  42.0% 420/1000 eps=0.801 recent_reward=+0.024 avg_loss=0.0031 elapsed=00:12 eta=00:17
```

Use `--progress-every 50` to refresh more often, or `--no-progress` to disable
the live bar.

Force CUDA and fail fast if the GPU is not available:

```bash
python scripts/train_dqn.py --episodes 1000 --eval-games 100 --device cuda --require-cuda --checkpoint checkpoints/dqn.pt
```

## Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/dqn.pt --games 200 --device auto
```

Training now prints per-opponent evaluation keys such as
`eval_random_reward` and `eval_rule_reward` when multiple evaluation opponents
are requested.

## Demo

```bash
python scripts/play_demo.py --checkpoint checkpoints/dqn.pt --seed 7 --device auto
```

The demo prints the hole cards, community cards, actions, pot movement, and
final payoff for a single virtual hand.

## Web Table

Start the local interactive table:

```bash
python scripts/serve_web.py --host 127.0.0.1 --port 8765 --checkpoint checkpoints/dqn.pt
```

Then open:

```text
http://127.0.0.1:8765
```

The Web table runs one human player against three AI seats. The DQN seat loads
the included `checkpoints/dqn.pt` model when it exists; otherwise the interface
keeps working with rule-based fallback decisions. The page supports:

- Chinese chip betting controls: click a chip multiple times to bet multiple chips.
- `弃牌`
- `过牌 / 跟注`
- `下注所选`
- `全押`
- `开始游戏`
- `重置游戏`
- AI 5-second thinking timers and human 15-second action timers.
- Bankroll carry-over between hands until a player reaches zero; reset starts a fresh game.

The right-side strategy panel shows the latest AI intent, legal actions,
Q-values, recent betting actions, and a running win/loss trend.

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

## 快速运行方法（GitHub Code 页面底部）

GitHub 会在仓库的 **Code** 页面文件列表下方自动显示这个根目录
`README.md`，所以这里就是 Code 界面最下边能看到的运行说明。

```bash
git clone https://github.com/hengrui900-cloud/Texas-Hold-em-poker.git
cd Texas-Hold-em-poker
python -m pip install -e ".[dev]"
python scripts/serve_web.py --host 127.0.0.1 --port 8765 --checkpoint checkpoints/dqn.pt
```

然后打开：

```text
http://127.0.0.1:8765
```

如果本地还没有 `checkpoints/dqn.pt`，网页仍会用规则 AI 兜底运行。想生成训练模型可执行：

```bash
python scripts/train_dqn.py --episodes 200000 --eval-games 5000 --device auto --checkpoint checkpoints/dqn.pt --batch-size 128 --replay-start-size 4096 --replay-size 100000 --hidden-size 256 --learning-rate 0.0003 --epsilon-decay 600000 --epsilon-end 0.05 --target-update 1000 --update-every 8 --opponent-pool random,rule --evaluation-opponents random,rule --all-in-margin 0.25 --exploration-all-in-probability 0.02 --progress-every 5000
```

测试：

```bash
python -m pytest -q
```
