from texas_holdem.actions import Action
from texas_holdem.env import TexasHoldemEnv
from texas_holdem.game.engine import TexasHoldemGame


def test_initial_blinds_and_check_call_update_pot_and_stack():
    game = TexasHoldemGame(seed=11, starting_stack=100)
    game.reset()
    actor = game.current_player
    stack_before = game.players[actor].stack
    pot_before = game.pot

    game.step(Action.CHECK_CALL)

    assert game.pot >= pot_before
    assert game.players[actor].stack <= stack_before
    assert game.players[actor].committed >= 2


def test_fold_finishes_hand_with_zero_sum_payoff():
    env = TexasHoldemEnv(seed=5)
    env.reset()

    _, reward, done, info = env.step(Action.FOLD)

    assert done is True
    assert reward != 0.0
    assert sum(info["payoffs"]) == 0


def test_all_in_sequence_reaches_terminal_showdown():
    env = TexasHoldemEnv(seed=9)
    env.reset()

    done = False
    steps = 0
    while not done and steps < 20:
        action = Action.ALL_IN if Action.ALL_IN in env.legal_actions() else Action.CHECK_CALL
        _, _, done, info = env.step(action)
        steps += 1

    assert done is True
    assert len(info["public_cards"]) == 5
    assert sum(info["payoffs"]) == 0


def test_environment_observation_contains_vector_and_action_mask():
    env = TexasHoldemEnv(seed=13)
    observation = env.reset()

    assert observation["obs"].shape == (env.observation_size,)
    assert observation["action_mask"].shape == (len(Action),)
    assert set(observation["legal_actions"]) == set(env.legal_actions())
