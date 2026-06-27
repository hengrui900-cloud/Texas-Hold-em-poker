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


def test_training_progress_output_shows_percent_eta_and_device(tmp_path, capsys):
    checkpoint = tmp_path / "dqn.pt"

    train(
        episodes=5,
        eval_games=2,
        seed=4,
        checkpoint_path=checkpoint,
        batch_size=4,
        replay_start_size=4,
        hidden_size=16,
        device="auto",
        show_progress=True,
        progress_every=2,
    )

    output = capsys.readouterr().out
    assert "training device=" in output
    assert "100.0%" in output
    assert "eta=" in output
    assert "evaluating games=" in output


def test_dqn_agent_can_be_placed_on_cuda_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment")

    env = TexasHoldemEnv(seed=17)
    agent = DQNAgent(state_size=env.observation_size, hidden_size=16, batch_size=4, device="cuda")

    assert agent.device.type == "cuda"
