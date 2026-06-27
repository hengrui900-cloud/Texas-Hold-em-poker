from __future__ import annotations

from texas_holdem.actions import Action
from texas_holdem.game.evaluator import evaluate_seven


class RuleBasedAgent:
    def act(self, observation: dict, legal_actions=None, training: bool = False):
        legal = list(legal_actions or observation["legal_actions"])
        raw = observation["raw_state"]
        cards = raw["hand"] + raw["public_cards"]

        if len(cards) == 7:
            strength = evaluate_seven(cards)
            if strength.category >= 4 and Action.RAISE_POT in legal:
                return Action.RAISE_POT

        ranks = [card.rank for card in raw["hand"]]
        suited = len({card.suit for card in raw["hand"]}) == 1
        high = any(card.rank in {"A", "K", "Q"} for card in raw["hand"])
        if ranks[0] == ranks[1] and Action.RAISE_HALF_POT in legal:
            return Action.RAISE_HALF_POT
        if suited and high and Action.CHECK_CALL in legal:
            return Action.CHECK_CALL
        if Action.CHECK_CALL in legal:
            return Action.CHECK_CALL
        return legal[0]
