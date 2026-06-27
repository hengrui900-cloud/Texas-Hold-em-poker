from texas_holdem.actions import Action
from texas_holdem.multiplayer import MultiplayerHoldemGame
from texas_holdem.web_session import PokerWebSession


def test_multiplayer_game_starts_four_players_with_blinds():
    game = MultiplayerHoldemGame(seed=8, num_players=4, starting_stack=100)
    game.reset()

    assert len(game.players) == 4
    assert game.pot == 3
    assert len(game.public_cards) == 0
    assert 0 <= game.current_player < 4
    assert Action.FOLD in game.legal_actions()
    assert Action.CHECK_CALL in game.legal_actions()


def test_multiplayer_game_can_reach_terminal_zero_sum_payoff():
    game = MultiplayerHoldemGame(seed=12, num_players=4, starting_stack=100)
    game.reset()

    steps = 0
    while not game.terminal and steps < 60:
        action = Action.CHECK_CALL if Action.CHECK_CALL in game.legal_actions() else game.legal_actions()[0]
        game.step(action)
        steps += 1

    assert game.terminal is True
    assert len(game.payoffs) == 4
    assert sum(game.payoffs) == 0


def test_postflop_street_starts_left_of_dealer_after_preflop_calls():
    game = MultiplayerHoldemGame(seed=3, num_players=4, starting_stack=100)
    game.dealer = 0
    game.small_blind_player = 1
    game.big_blind_player = 2
    game.current_player = 3
    game.acted = [False, False, False, False]

    for player_id in [3, 0, 1, 2]:
        assert game.current_player == player_id
        game.step(Action.CHECK_CALL)

    assert game.stage.name == "FLOP"
    assert game.current_player == 1


def test_web_session_auto_advances_ai_until_human_turn_or_terminal():
    session = PokerWebSession(seed=4, ai_count=3)

    state = session.new_hand()

    assert len(state["players"]) == 4
    assert state["players"][0]["is_human"] is True
    assert state["terminal"] or state["current_player"] == 0
    assert "ai_thinking" in state
    assert "legal_actions" in state


def test_web_session_human_action_returns_strategy_panel_data():
    session = PokerWebSession(seed=9, ai_count=3)
    state = session.new_hand()

    if not state["terminal"] and "check_call" in state["legal_actions"]:
        state = session.act("check_call")

    assert "last_actions" in state
    assert "q_values" in state["ai_thinking"]
    assert "win_loss_trend" in state
