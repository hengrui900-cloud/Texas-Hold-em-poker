from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texas_holdem.agents import DQNAgent, RandomAgent
from texas_holdem.env import TexasHoldemEnv


def play_episode(env: TexasHoldemEnv, agent: DQNAgent, opponent, training: bool = True):
    observation = env.reset()
    done = False
    info = {"payoffs": [0, 0]}

    while not done:
        if env.current_player != 0:
            action = opponent.act(observation, training=training)
            observation, _, done, info = env.step(action)
            continue

        state = env.observe()
        action = agent.act(state, training=training)
        observation, reward, done, info = env.step(action)

        while not done and env.current_player != 0:
            opp_obs = env.observe()
            opp_action = opponent.act(opp_obs, training=training)
            observation, reward, done, info = env.step(opp_action)

        next_state = env.observe()
        if training:
            agent.remember(
                state["obs"],
                int(action),
                reward,
                next_state["obs"],
                done,
                next_state["legal_actions"],
            )
            agent.train_step()

    return info["payoffs"][0] / float(env.starting_stack)


def evaluate(agent: DQNAgent, games: int = 100, seed: int = 0) -> float:
    rewards = []
    for game_index in range(games):
        env = TexasHoldemEnv(seed=seed + game_index)
        opponent = RandomAgent(seed=seed + 10_000 + game_index)
        rewards.append(play_episode(env, agent, opponent, training=False))
    return float(sum(rewards) / max(1, len(rewards)))


def train(
    episodes: int = 1000,
    eval_games: int = 100,
    seed: int = 0,
    checkpoint_path: str | Path = "checkpoints/dqn.pt",
    batch_size: int = 64,
    replay_start_size: int = 64,
    hidden_size: int = 128,
    device: str = "auto",
    require_cuda: bool = False,
):
    env = TexasHoldemEnv(seed=seed)
    agent = DQNAgent(
        state_size=env.observation_size,
        hidden_size=hidden_size,
        batch_size=batch_size,
        replay_start_size=replay_start_size,
        replay_size=max(1000, replay_start_size * 4),
        seed=seed,
        device=device,
    )
    if require_cuda and agent.device.type != "cuda":
        raise RuntimeError(f"CUDA was required, but the DQN agent is running on {agent.device_name}.")
    opponent = RandomAgent(seed=seed + 1)
    episode_rewards = []

    for episode in range(episodes):
        env = TexasHoldemEnv(seed=seed + episode)
        reward = play_episode(env, agent, opponent, training=True)
        episode_rewards.append(reward)

    average_eval_reward = evaluate(agent, games=eval_games, seed=seed + 50_000)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(checkpoint_path)
    return {
        "episode_rewards": episode_rewards,
        "average_eval_reward": average_eval_reward,
        "checkpoint_path": str(checkpoint_path),
        "device": agent.device_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a PyTorch DQN agent for heads-up Texas Hold'em.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/dqn.pt"))
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is not available.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-start-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    args = parser.parse_args()

    metrics = train(
        episodes=args.episodes,
        eval_games=args.eval_games,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        replay_start_size=args.replay_start_size,
        hidden_size=args.hidden_size,
        device=args.device,
        require_cuda=args.require_cuda,
    )
    print(f"saved_checkpoint={metrics['checkpoint_path']}")
    print(f"device={metrics['device']}")
    print(f"average_eval_reward={metrics['average_eval_reward']:.4f}")


if __name__ == "__main__":
    main()
