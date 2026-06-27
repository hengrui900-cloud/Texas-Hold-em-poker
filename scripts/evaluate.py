from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_dqn import evaluate
from texas_holdem.agents import DQNAgent


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved DQN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/dqn.pt"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    agent = DQNAgent.load(args.checkpoint)
    reward = evaluate(agent, games=args.games, seed=args.seed)
    print(f"average_reward={reward:.4f}")


if __name__ == "__main__":
    main()
