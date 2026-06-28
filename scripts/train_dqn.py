from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texas_holdem.agents import DQNAgent, RandomAgent, RuleBasedAgent
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


def normalize_opponent_names(opponents: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(opponents, str):
        raw_names = opponents.split(",")
    else:
        raw_names = list(opponents)
    aliases = {
        "random": "random",
        "rand": "random",
        "rule": "rule",
        "rules": "rule",
        "rule_based": "rule",
        "rule-based": "rule",
    }
    names = []
    for name in raw_names:
        normalized = aliases.get(str(name).strip().lower().replace(" ", "_"))
        if normalized is None:
            raise ValueError(f"Unsupported opponent '{name}'. Use random or rule.")
        names.append(normalized)
    if not names:
        raise ValueError("At least one opponent is required.")
    return tuple(names)


def make_opponent(name: str, seed: int):
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "rule":
        return RuleBasedAgent()
    raise ValueError(f"Unsupported opponent '{name}'.")


def evaluate(agent: DQNAgent, games: int = 100, seed: int = 0, opponent: str = "random") -> float:
    rewards = []
    opponent_name = normalize_opponent_names((opponent,))[0]
    for game_index in range(games):
        env = TexasHoldemEnv(seed=seed + game_index)
        opponent_agent = make_opponent(opponent_name, seed=seed + 10_000 + game_index)
        rewards.append(play_episode(env, agent, opponent_agent, training=False))
    return float(sum(rewards) / max(1, len(rewards)))


def evaluate_suite(
    agent: DQNAgent,
    games: int = 100,
    seed: int = 0,
    opponents: str | Iterable[str] = ("random",),
) -> dict[str, float]:
    return {
        opponent: evaluate(agent, games=games, seed=seed + index * 100_000, opponent=opponent)
        for index, opponent in enumerate(normalize_opponent_names(opponents))
    }


def train(
    episodes: int = 1000,
    eval_games: int = 100,
    seed: int = 0,
    checkpoint_path: str | Path = "checkpoints/dqn.pt",
    batch_size: int = 64,
    replay_start_size: int = 64,
    replay_size: int | None = None,
    hidden_size: int = 128,
    learning_rate: float = 0.0005,
    gamma: float = 0.99,
    epsilon_decay: int = 50_000,
    epsilon_end: float = 0.05,
    target_update: int = 500,
    update_every: int = 1,
    opponent_pool: str | Iterable[str] = "random,rule",
    evaluation_opponents: str | Iterable[str] = ("random", "rule"),
    dueling: bool = True,
    double_dqn: bool = True,
    all_in_margin: float = 0.20,
    exploration_all_in_probability: float = 0.03,
    device: str = "auto",
    require_cuda: bool = False,
    show_progress: bool = False,
    progress_every: int = 100,
):
    env = TexasHoldemEnv(seed=seed)
    replay_size = replay_size or max(50_000, replay_start_size * 16)
    agent = DQNAgent(
        state_size=env.observation_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        epsilon_end=epsilon_end,
        batch_size=batch_size,
        replay_start_size=replay_start_size,
        replay_size=replay_size,
        target_update=target_update,
        update_every=update_every,
        dueling=dueling,
        double_dqn=double_dqn,
        all_in_margin=all_in_margin,
        exploration_all_in_probability=exploration_all_in_probability,
        seed=seed,
        device=device,
    )
    if require_cuda and agent.device.type != "cuda":
        raise RuntimeError(f"CUDA was required, but the DQN agent is running on {agent.device_name}.")
    opponent_names = normalize_opponent_names(opponent_pool)
    opponent_selector = agent.rng
    episode_rewards = []
    recent_rewards = deque(maxlen=100)
    recent_losses = deque(maxlen=100)
    start_time = time.perf_counter()
    progress_every = max(1, progress_every)

    if show_progress:
        print(
            f"training device={agent.device_name} episodes={episodes} "
            f"batch_size={batch_size} hidden_size={hidden_size} replay_size={replay_size} "
            f"epsilon_decay={epsilon_decay} opponents={','.join(opponent_names)} "
            f"update_every={update_every} dueling={dueling} double_dqn={double_dqn} "
            f"all_in_margin={all_in_margin} exploration_all_in_probability={exploration_all_in_probability}"
        )

    for episode in range(episodes):
        env = TexasHoldemEnv(seed=seed + episode)
        opponent_name = opponent_names[int(opponent_selector.integers(len(opponent_names)))]
        opponent = make_opponent(opponent_name, seed=seed + 10_000 + episode)
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
    evaluation = evaluate_suite(agent, games=eval_games, seed=seed + 50_000, opponents=evaluation_opponents)
    average_eval_reward = evaluation.get("random", next(iter(evaluation.values())))
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(checkpoint_path)
    return {
        "episode_rewards": episode_rewards,
        "average_eval_reward": average_eval_reward,
        "evaluation": evaluation,
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
    parser.add_argument("--replay-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-decay", type=int, default=50_000)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--opponent-pool", type=str, default="random,rule")
    parser.add_argument("--evaluation-opponents", type=str, default="random,rule")
    parser.add_argument("--no-dueling", action="store_true")
    parser.add_argument("--no-double-dqn", action="store_true")
    parser.add_argument("--all-in-margin", type=float, default=0.20)
    parser.add_argument("--exploration-all-in-probability", type=float, default=0.03)
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
        replay_size=args.replay_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        epsilon_end=args.epsilon_end,
        target_update=args.target_update,
        update_every=args.update_every,
        opponent_pool=args.opponent_pool,
        evaluation_opponents=args.evaluation_opponents,
        dueling=not args.no_dueling,
        double_dqn=not args.no_double_dqn,
        all_in_margin=args.all_in_margin,
        exploration_all_in_probability=args.exploration_all_in_probability,
        device=args.device,
        require_cuda=args.require_cuda,
        show_progress=not args.no_progress,
        progress_every=args.progress_every,
    )
    print(f"saved_checkpoint={metrics['checkpoint_path']}")
    print(f"device={metrics['device']}")
    for opponent, reward in metrics["evaluation"].items():
        print(f"eval_{opponent}_reward={reward:.4f}")
    print(f"average_eval_reward={metrics['average_eval_reward']:.4f}")


if __name__ == "__main__":
    main()
