from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texas_holdem.actions import action_name
from texas_holdem.agents import DQNAgent, RandomAgent
from texas_holdem.env import TexasHoldemEnv, format_cards


def load_or_random_agent(path: Path, state_size: int):
    if path.exists():
        return DQNAgent.load(path)
    return DQNAgent(state_size=state_size, epsilon_start=1.0, epsilon_end=1.0)


def main():
    parser = argparse.ArgumentParser(description="Print one virtual Texas Hold'em duel.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/dqn.pt"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    env = TexasHoldemEnv(seed=args.seed)
    agent = load_or_random_agent(args.checkpoint, env.observation_size)
    opponent = RandomAgent(seed=args.seed + 1)
    observation = env.reset()
    done = False

    print("Initial state")
    print(env.render())
    print()

    while not done:
        actor = env.current_player
        if actor == 0:
            action = agent.act(observation, training=False)
        else:
            action = opponent.act(observation)
        observation, reward, done, info = env.step(action)
        print(f"player {actor}: {action_name(action)}")
        print(env.render())
        print()

    print(f"community: {format_cards(info['public_cards'])}")
    print(f"payoffs: {info['payoffs']}")
    print(f"player0_reward: {reward:.3f}")


if __name__ == "__main__":
    main()
