from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.train_dqn import train
from texas_holdem.actions import Action
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


def test_dqn_agent_persists_dueling_and_double_dqn_config(tmp_path):
    checkpoint = tmp_path / "dueling_double.pt"
    agent = DQNAgent(
        state_size=109,
        hidden_size=16,
        batch_size=4,
        replay_size=32,
        dueling=True,
        double_dqn=True,
        device="cpu",
    )

    agent.save(checkpoint)
    loaded = DQNAgent.load(checkpoint, device="cpu")

    assert loaded.dueling is True
    assert loaded.double_dqn is True
    assert loaded.predict(torch.zeros(109).numpy()).shape == (5,)


def test_dqn_avoids_marginal_all_in_when_safer_action_is_close():
    agent = DQNAgent(
        state_size=109,
        hidden_size=16,
        batch_size=4,
        replay_size=32,
        all_in_margin=0.25,
        device="cpu",
    )
    agent.predict = lambda state: torch.tensor([0.0, 0.72, 0.65, 0.7, 0.8]).numpy()
    observation = {
        "obs": torch.zeros(109).numpy(),
        "legal_actions": [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN],
    }

    assert agent.act(observation, training=False) == Action.CHECK_CALL


def test_dqn_allows_all_in_when_q_edge_is_large():
    agent = DQNAgent(
        state_size=109,
        hidden_size=16,
        batch_size=4,
        replay_size=32,
        all_in_margin=0.25,
        device="cpu",
    )
    agent.predict = lambda state: torch.tensor([0.0, 0.1, 0.2, 0.3, 0.9]).numpy()
    observation = {
        "obs": torch.zeros(109).numpy(),
        "legal_actions": [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN],
    }

    assert agent.act(observation, training=False) == Action.ALL_IN


def test_dqn_replay_memory_keeps_capacity_with_latest_experiences():
    agent = DQNAgent(
        state_size=109,
        hidden_size=16,
        batch_size=2,
        replay_start_size=2,
        replay_size=3,
        device="cpu",
    )
    state = torch.zeros(109).numpy()

    for reward in range(5):
        agent.remember(state, 1, reward, state, True, [])

    assert agent.memory.supports_fast_random_access is True
    assert len(agent.memory) == 3
    assert {item.reward for item in agent.memory} == {2.0, 3.0, 4.0}


def test_dqn_update_every_skips_gradient_steps_between_updates():
    agent = DQNAgent(
        state_size=109,
        hidden_size=16,
        batch_size=2,
        replay_start_size=2,
        replay_size=8,
        update_every=3,
        device="cpu",
    )
    state = torch.zeros(109).numpy()
    for reward in range(3):
        agent.remember(state, 1, reward, state, True, [])

    assert agent.train_step() is None
    assert agent.train_step() is None
    assert agent.train_step() is not None
    assert agent.steps == 3
    assert agent.updates == 1


def test_training_exposes_long_run_hyperparameters_and_eval_breakdown(tmp_path):
    checkpoint = tmp_path / "stronger_dqn.pt"

    metrics = train(
        episodes=8,
        eval_games=2,
        seed=11,
        checkpoint_path=checkpoint,
        batch_size=4,
        replay_start_size=4,
        replay_size=64,
        hidden_size=16,
        learning_rate=0.0005,
        epsilon_decay=1234,
        target_update=17,
        update_every=3,
        opponent_pool="random,rule",
        evaluation_opponents=("random", "rule"),
        dueling=True,
        double_dqn=True,
        device="cpu",
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert payload["config"]["replay_size"] == 64
    assert payload["config"]["learning_rate"] == 0.0005
    assert payload["config"]["epsilon_decay"] == 1234
    assert payload["config"]["target_update"] == 17
    assert payload["config"]["update_every"] == 3
    assert payload["config"]["dueling"] is True
    assert payload["config"]["double_dqn"] is True
    assert set(metrics["evaluation"]) == {"random", "rule"}
    assert metrics["average_eval_reward"] == metrics["evaluation"]["random"]
