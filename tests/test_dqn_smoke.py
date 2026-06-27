from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.train_dqn import train
from texas_holdem.agents import DQNAgent
from texas_holdem.env import TexasHoldemEnv


def test_dqn_training_smoke_saves_checkpoint(tmp_path):
    checkpoint = tmp_path / "dqn.pt"

    metrics = train(
        episodes=8,
        eval_games=4,
        seed=3,
        checkpoint_path=checkpoint,
        batch_size=4,
        replay_start_size=4,
        hidden_size=16,
        device="auto",
    )

    assert checkpoint.exists()
    assert len(metrics["episode_rewards"]) == 8
    assert isinstance(metrics["average_eval_reward"], float)
    assert metrics["device"].startswith(("cpu", "cuda"))


def test_dqn_agent_can_be_placed_on_cuda_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment")

    env = TexasHoldemEnv(seed=17)
    agent = DQNAgent(state_size=env.observation_size, hidden_size=16, batch_size=4, device="cuda")

    assert agent.device.type == "cuda"
