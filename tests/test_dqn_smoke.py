from pathlib import Path

from scripts.train_dqn import train


def test_dqn_training_smoke_saves_checkpoint(tmp_path):
    checkpoint = tmp_path / "dqn.pt"

    metrics = train(
        episodes=8,
        eval_games=4,
        seed=3,
        checkpoint_path=checkpoint,
        batch_size=4,
        replay_start_size=4,
    )

    assert checkpoint.exists()
    assert len(metrics["episode_rewards"]) == 8
    assert isinstance(metrics["average_eval_reward"], float)
