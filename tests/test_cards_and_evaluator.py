import pytest

from texas_holdem.game.cards import Card, Deck
from texas_holdem.game.evaluator import evaluate_seven


def cards(text):
    return [Card.from_str(token) for token in text.split()]


def test_deck_deals_52_unique_cards():
    deck = Deck(seed=7)

    dealt = [deck.deal() for _ in range(52)]

    assert len(dealt) == 52
    assert len(set(dealt)) == 52
    with pytest.raises(IndexError):
        deck.deal()


def test_hand_evaluator_orders_core_poker_hands():
    straight_flush = evaluate_seven(cards("SA SK SQ SJ ST C2 D3"))
    quads = evaluate_seven(cards("SA HA DA CA S9 H2 D3"))
    full_house = evaluate_seven(cards("SA HA DA CK SK H2 D3"))
    flush = evaluate_seven(cards("SA S9 S7 S4 S2 HQ DJ"))
    pair = evaluate_seven(cards("SA HA D9 C7 S4 H2 D3"))
    high_card = evaluate_seven(cards("SA HK D9 C7 S4 H2 D3"))

    assert straight_flush > quads > full_house > flush > pair > high_card


def test_hand_evaluator_supports_wheel_straight():
    wheel = evaluate_seven(cards("SA H5 D4 C3 S2 H9 DK"))
    trips = evaluate_seven(cards("SA HA DA C7 S4 H2 D3"))

    assert wheel.name == "straight"
    assert wheel > trips
