from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texas_holdem.agents import DQNAgent, RandomAgent
from texas_holdem.env import TexasHoldemEnv


def play_episode(env: TexasHoldemEnv, agent: DQNAgent, opponent, training: bool = True, return_losses: bool = False):
    observation = env.reset()
    done = False
    info = {"payoffs": [0, 0]}
    losses = []

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
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

    reward = info["payoffs"][0] / float(env.starting_stack)
    if return_losses:
        return reward, losses
    return reward


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_line(
    episode: int,
    total: int,
    width: int,
    epsilon: float,
    recent_reward: float,
    average_loss: float | None,
    elapsed: float,
) -> str:
    total = max(1, total)
    fraction = min(1.0, max(0.0, episode / total))
    filled = int(round(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    rate = episode / elapsed if elapsed > 0 else 0.0
    remaining = (total - episode) / rate if rate > 0 else 0.0
    loss_text = "n/a" if average_loss is None else f"{average_loss:.4f}"
    return (
        f"train [{bar}] {fraction * 100:5.1f}% {episode}/{total} "
        f"eps={epsilon:.3f} recent_reward={recent_reward:+.3f} avg_loss={loss_text} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(remaining)}"
    )


def write_progress(line: str, final: bool = False) -> None:
    sys.stdout.write("\r" + line)
    if final:
        sys.stdout.write("\n")
    sys.stdout.flush()


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
    show_progress: bool = False,
    progress_every: int = 100,
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
    recent_rewards = deque(maxlen=100)
    recent_losses = deque(maxlen=100)
    start_time = time.perf_counter()
    progress_every = max(1, progress_every)

    if show_progress:
        print(
            f"training device={agent.device_name} episodes={episodes} "
            f"batch_size={batch_size} hidden_size={hidden_size}"
        )

    for episode in range(episodes):
        env = TexasHoldemEnv(seed=seed + episode)
        reward, losses = play_episode(env, agent, opponent, training=True, return_losses=True)
        episode_rewards.append(reward)
        recent_rewards.append(reward)
        recent_losses.extend(losses)

        episode_number = episode + 1
        if show_progress and (
            episode_number == 1 or episode_number == episodes or episode_number % progress_every == 0
        ):
            average_loss = sum(recent_losses) / len(recent_losses) if recent_losses else None
            line = progress_line(
                episode=episode_number,
                total=episodes,
                width=28,
                epsilon=agent.epsilon,
                recent_reward=sum(recent_rewards) / len(recent_rewards),
                average_loss=average_loss,
                elapsed=time.perf_counter() - start_time,
            )
            write_progress(line, final=episode_number == episodes)

    if show_progress:
        print(f"evaluating games={eval_games} device={agent.device_name}")
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
    parser.add_argument("--progress-every", type=int, default=100, help="Refresh progress every N episodes.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the training progress bar.")
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
        show_progress=not args.no_progress,
        progress_every=args.progress_every,
    )
    print(f"saved_checkpoint={metrics['checkpoint_path']}")
    print(f"device={metrics['device']}")
    print(f"average_eval_reward={metrics['average_eval_reward']:.4f}")


if __name__ == "__main__":
    main()
