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
    assert "ai_thinking" in state
    assert "legal_actions" in state
    assert "turn_timer" in state
    assert state["stakes"] == {
        "currency": "$",
        "starting_stack": 1000,
        "small_blind": 20,
        "big_blind": 80,
    }
    assert state["chip_rack"] == [
        {"value": 20, "count": 1},
        {"value": 80, "count": 1},
        {"value": 100, "count": 5},
        {"value": 200, "count": 2},
    ]
    assert "rule_tip" in state


def test_web_session_reset_uses_fresh_random_hand_seed():
    session = PokerWebSession(seed=21, ai_count=3)
    first = session.reset_game(seed=21)
    second = session.reset_game(seed=21)

    assert first["hand_seed"] != second["hand_seed"]


def test_web_session_ai_action_advances_one_bot_turn():
    session = PokerWebSession(seed=4, ai_count=3)
    state = session.new_hand()

    if state["current_player"] == 0 or state["terminal"]:
        state = session.new_hand()

    if not state["terminal"] and state["current_player"] != 0:
        before_actions = len(state["last_actions"])
        next_state = session.ai_act()

        assert len(next_state["last_actions"]) == before_actions + 1


def test_turn_timer_distinguishes_ai_and_human_turns():
    session = PokerWebSession(seed=4, ai_count=3)
    state = session.new_hand()

    expected = 15 if state["current_player"] == 0 else 5

    assert state["turn_timer"]["seconds"] == expected
    assert state["turn_timer"]["player"] == state["current_player"]


def test_web_session_human_action_returns_strategy_panel_data():
    session = PokerWebSession(seed=9, ai_count=3)
    state = session.new_hand()

    if not state["terminal"] and "check_call" in state["legal_actions"]:
        state = session.act("check_call")

    assert "last_actions" in state
    assert "q_values" in state["ai_thinking"]
    assert "win_loss_trend" in state


def test_web_session_reset_game_clears_results_and_restarts_bankroll():
    session = PokerWebSession(seed=10, ai_count=3)
    session.results.extend([1.0, -0.5])
    session.hand_number = 8

    state = session.reset_game(seed=10)

    assert state["hand_number"] == 1
    assert state["win_loss_trend"]["hands"] == 0
    assert all(player["stack"] <= 1000 for player in state["players"])
    assert state["stakes"]["starting_stack"] == 1000


def test_new_hand_inherits_previous_hand_bankrolls_until_bankrupt():
    session = PokerWebSession(seed=10, ai_count=3)
    session.game.terminal = True
    session.game.players[0].stack = 1200
    session.game.players[1].stack = 900
    session.game.players[2].stack = 900
    session.game.players[3].stack = 1000
    session.game.payoffs = [200, -100, -100, 0]

    next_state = session.new_hand()

    assert next_state["hand_number"] == 2
    assert sum(player["stack"] for player in next_state["players"]) + next_state["pot"] == 4000
    assert next_state["bankrolls"] == [1200, 900, 900, 1000]


def test_new_hand_blocks_when_any_bankroll_is_zero():
    session = PokerWebSession(seed=10, ai_count=3)
    session.game.terminal = True
    session.game.players[0].stack = 0
    session.game.players[1].stack = 1300
    session.game.players[2].stack = 1300
    session.game.players[3].stack = 1400
    session.game.payoffs = [-1000, 300, 300, 400]

    state = session.new_hand()

    assert state["bankrupt"] is True
    assert "有一方已经资产归0" in state["rule_tip"]


def test_reset_game_discards_bankrupt_terminal_hand_and_starts_fresh():
    session = PokerWebSession(seed=10, ai_count=3)
    session.game.terminal = True
    session.game.players[0].stack = 0
    session.game.players[1].stack = 1300
    session.game.players[2].stack = 1300
    session.game.players[3].stack = 1400
    session.game.payoffs = [-1000, 300, 300, 400]

    blocked = session.new_hand()
    reset = session.reset_game(seed=10)

    assert blocked["bankrupt"] is True
    assert reset["bankrupt"] is False
    assert reset["hand_number"] == 1
    assert reset["terminal"] is False
    assert reset["bankrolls"] == [1000, 1000, 1000, 1000]
    assert sum(player["stack"] for player in reset["players"]) + reset["pot"] == 4000
