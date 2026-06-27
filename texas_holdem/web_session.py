from __future__ import annotations

from pathlib import Path
import math

import numpy as np

from texas_holdem.actions import Action, action_name
from texas_holdem.agents import DQNAgent
from texas_holdem.game.cards import Card
from texas_holdem.game.evaluator import evaluate_seven
from texas_holdem.multiplayer import MultiplayerHoldemGame, Stage


ACTION_LABELS = {
    "fold": "Fold",
    "check_call": "Check / Call",
    "raise_half_pot": "Raise 1/2 Pot",
    "raise_pot": "Raise Pot",
    "all_in": "All-in",
}


class PokerWebSession:
    def __init__(
        self,
        seed: int | None = 7,
        ai_count: int = 3,
        checkpoint_path: str | Path = "checkpoints/dqn.pt",
    ):
        self.seed = seed or 7
        self.ai_count = ai_count
        self.checkpoint_path = Path(checkpoint_path)
        self.hand_number = 0
        self.results: list[float] = []
        self.last_ai_info = self._empty_ai_info()
        self.dqn_agent = self._load_dqn_agent()
        self.game: MultiplayerHoldemGame | None = None
        self.new_hand()

    def new_hand(self, ai_count: int | None = None, seed: int | None = None) -> dict:
        if ai_count is not None:
            self.ai_count = max(1, min(5, int(ai_count)))
        if seed is not None:
            self.seed = int(seed)
        self.hand_number += 1
        self.game = MultiplayerHoldemGame(
            seed=self.seed + self.hand_number,
            num_players=self.ai_count + 1,
            starting_stack=100,
            small_blind=1,
            big_blind=2,
        )
        self.last_ai_info = self._empty_ai_info()
        self._auto_play_until_human()
        return self.state()

    def act(self, action_name_or_id: str | int) -> dict:
        if self.game is None:
            return self.new_hand()
        if self.game.terminal:
            return self.state()
        action = self._parse_action(action_name_or_id)
        if self.game.current_player != 0:
            self._auto_play_until_human()
            return self.state()
        self.game.step(action)
        self._auto_play_until_human()
        return self.state()

    def state(self) -> dict:
        assert self.game is not None
        game = self.game
        if game.terminal and (not self.results or len(self.results) < self.hand_number):
            self.results.append(game.payoffs[0] / float(game.starting_stack))

        legal = [action_name(action) for action in game.legal_actions()] if game.current_player == 0 else []
        return {
            "hand_number": self.hand_number,
            "stage": game.stage.name.title(),
            "street_index": int(game.stage),
            "pot": game.pot,
            "current_bet": game.current_bet,
            "current_player": game.current_player,
            "dealer": game.dealer,
            "terminal": game.terminal,
            "players": self._players_payload(),
            "public_cards": [self._card_payload(card) for card in game.public_cards],
            "legal_actions": legal,
            "action_options": self._action_options(),
            "last_actions": self._history_payload(),
            "ai_thinking": self.last_ai_info,
            "win_loss_trend": self._trend_payload(),
            "payoffs": list(game.payoffs),
            "checkpoint_loaded": self.dqn_agent is not None,
            "checkpoint_path": str(self.checkpoint_path),
        }

    def _auto_play_until_human(self) -> None:
        assert self.game is not None
        guard = 0
        while not self.game.terminal and self.game.current_player != 0 and guard < 80:
            player_id = self.game.current_player
            action, info = self._choose_ai_action(player_id)
            self.last_ai_info = info
            self.game.step(action)
            guard += 1

    def _choose_ai_action(self, player_id: int) -> tuple[Action, dict]:
        assert self.game is not None
        personality = self._personality(player_id)
        legal = self.game.legal_actions()
        q_values = self._synthetic_q_values(legal, personality)
        intent = "Pot Control / Check"
        confidence = 0.52

        if personality == "DQN" and self.dqn_agent is not None:
            obs = self._dqn_observation(player_id)
            raw_q = self.dqn_agent.predict(obs)
            q_values = {action_name(action): float(raw_q[int(action)]) for action in legal}
            action = max(legal, key=lambda candidate: raw_q[int(candidate)])
            confidence = self._confidence([raw_q[int(action)] for action in legal], raw_q[int(action)])
            intent = self._intent_for_action(action)
        else:
            action = self._rule_action(player_id, legal, personality)
            intent = self._intent_for_action(action)
            if action in (Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN):
                confidence = 0.72

        return action, {
            "player": player_id,
            "name": self._player_name(player_id),
            "personality": personality,
            "street": self.game.stage.name.title(),
            "intent": intent,
            "confidence": round(confidence, 2),
            "q_values": self._q_value_payload(q_values, legal),
            "hand_range": self._hand_range_text(player_id),
        }

    def _rule_action(self, player_id: int, legal: list[Action], personality: str) -> Action:
        assert self.game is not None
        player = self.game.players[player_id]
        ranks = sorted((card.rank_value for card in player.hand), reverse=True)
        suited = player.hand[0].suit == player.hand[1].suit
        pair = ranks[0] == ranks[1]
        high_card = ranks[0] >= 13
        facing_call = self.game.call_amount(player_id) > 0

        if personality == "Aggressive":
            if Action.RAISE_POT in legal and (pair or high_card or suited):
                return Action.RAISE_POT
            if Action.RAISE_HALF_POT in legal and not facing_call:
                return Action.RAISE_HALF_POT
            return Action.CHECK_CALL if Action.CHECK_CALL in legal else legal[0]

        if personality == "Conservative":
            if pair and Action.RAISE_HALF_POT in legal:
                return Action.RAISE_HALF_POT
            if facing_call and not (high_card or pair) and Action.FOLD in legal:
                return Action.FOLD
            return Action.CHECK_CALL if Action.CHECK_CALL in legal else legal[0]

        if Action.RAISE_HALF_POT in legal and (pair or (suited and high_card)):
            return Action.RAISE_HALF_POT
        return Action.CHECK_CALL if Action.CHECK_CALL in legal else legal[0]

    def _dqn_observation(self, player_id: int) -> np.ndarray:
        assert self.game is not None
        state = self.game.get_state(player_id)
        obs = np.zeros(109, dtype=np.float32)
        for card in state["hand"]:
            obs[card.index] = 1.0
        for card in state["public_cards"]:
            obs[52 + card.index] = 1.0
        obs[104] = state["my_stack"] / 100.0
        obs[105] = state["opponent_stack"] / 100.0
        obs[106] = state["pot"] / 200.0
        obs[107] = float(state["stage"]) / 4.0
        obs[108] = self.game.current_bet / 100.0
        return obs

    def _synthetic_q_values(self, legal: list[Action], personality: str) -> dict[str, float]:
        base = {action_name(action): -0.25 for action in Action}
        for action in legal:
            base[action_name(action)] = 0.05
        if personality == "Aggressive":
            base["raise_pot"] = 0.42
            base["raise_half_pot"] = 0.31
            base["all_in"] = 0.16
        elif personality == "Conservative":
            base["check_call"] = 0.38
            base["fold"] = 0.18
            base["raise_half_pot"] = 0.08
        else:
            base["check_call"] = 0.24
            base["raise_half_pot"] = 0.22
        return base

    def _action_options(self) -> dict:
        assert self.game is not None
        amounts = self.game.raise_amounts(0)
        call = self.game.call_amount(0)
        options = {}
        for action in Action:
            key = action_name(action)
            amount = 0
            if action == Action.CHECK_CALL:
                amount = call
            elif action == Action.RAISE_HALF_POT:
                amount = amounts["half_pot"]
            elif action == Action.RAISE_POT:
                amount = amounts["pot"]
            elif action == Action.ALL_IN:
                amount = amounts["all_in"]
            options[key] = {
                "label": ACTION_LABELS[key],
                "amount": amount,
                "enabled": key in ([action_name(item) for item in self.game.legal_actions()] if self.game.current_player == 0 else []),
            }
        return options

    def _players_payload(self) -> list[dict]:
        assert self.game is not None
        payload = []
        for player in self.game.players:
            visible = player.player_id == 0 or self.game.terminal
            payload.append(
                {
                    "id": player.player_id,
                    "name": self._player_name(player.player_id),
                    "personality": "Human" if player.player_id == 0 else self._personality(player.player_id),
                    "is_human": player.player_id == 0,
                    "stack": player.stack,
                    "committed": player.committed,
                    "total_committed": player.total_committed,
                    "folded": player.folded,
                    "all_in": player.all_in,
                    "is_dealer": player.player_id == self.game.dealer,
                    "cards": [self._card_payload(card) for card in player.hand] if visible else [],
                    "hidden_cards": 0 if visible else len(player.hand),
                }
            )
        return payload

    def _history_payload(self) -> list[dict]:
        assert self.game is not None
        rows = []
        for item in self.game.action_history[-8:]:
            rows.append(
                {
                    "player": item["player"],
                    "name": self._player_name(item["player"]),
                    "stage": item["before"]["stage"].title(),
                    "action": item["action"],
                    "label": ACTION_LABELS[item["action"]],
                    "pot": item["after"]["pot"],
                }
            )
        return rows

    def _trend_payload(self) -> dict:
        if not self.results:
            values = [0]
        else:
            running = 0
            values = []
            for result in self.results[-50:]:
                running += result
                values.append(round(running, 2))
        wins = sum(1 for result in self.results if result > 0)
        return {
            "values": values,
            "hands": len(self.results),
            "win_rate": round((wins / len(self.results)) * 100, 1) if self.results else 0.0,
            "ev_per_hand": round(sum(self.results) / len(self.results), 3) if self.results else 0.0,
        }

    def _q_value_payload(self, q_values: dict[str, float], legal: list[Action]) -> list[dict]:
        legal_names = {action_name(action) for action in legal}
        return [
            {
                "action": key,
                "label": ACTION_LABELS[key],
                "value": round(float(value), 3),
                "legal": key in legal_names,
            }
            for key, value in q_values.items()
        ]

    def _hand_range_text(self, player_id: int) -> str:
        assert self.game is not None
        cards = self.game.players[player_id].hand + self.game.public_cards
        if len(cards) == 7:
            rank = evaluate_seven(cards)
            return rank.name.replace("_", " ").title()
        ranks = sorted((card.rank for card in self.game.players[player_id].hand), reverse=True)
        return f"Top Pair / Broadway potential ({', '.join(ranks)})"

    def _confidence(self, values: list[float], chosen: float) -> float:
        exp = [math.exp(value - max(values)) for value in values]
        total = sum(exp)
        return exp[values.index(chosen)] / total if total else 0.0

    def _intent_for_action(self, action: Action) -> str:
        return {
            Action.FOLD: "Risk Avoidance / Fold",
            Action.CHECK_CALL: "Pot Control / Check-Call",
            Action.RAISE_HALF_POT: "Pressure / Half-Pot Raise",
            Action.RAISE_POT: "Value Pressure / Pot Raise",
            Action.ALL_IN: "Max Pressure / All-in",
        }[action]

    def _load_dqn_agent(self):
        if not self.checkpoint_path.exists():
            return None
        try:
            return DQNAgent.load(self.checkpoint_path, device="auto")
        except Exception:
            return None

    def _empty_ai_info(self) -> dict:
        return {
            "player": None,
            "name": "Waiting",
            "personality": "Mixed",
            "street": "Preflop",
            "intent": "Waiting for action",
            "confidence": 0.0,
            "q_values": [],
            "hand_range": "No AI action yet",
        }

    def _player_name(self, player_id: int) -> str:
        if player_id == 0:
            return "You"
        return f"AI-Bot {player_id}"

    def _personality(self, player_id: int) -> str:
        personalities = ["Human", "Conservative", "DQN", "Aggressive", "Balanced", "Tight"]
        return personalities[player_id] if player_id < len(personalities) else "Balanced"

    def _parse_action(self, value: str | int) -> Action:
        if isinstance(value, int):
            return Action(value)
        normalized = value.strip().lower().replace("-", "_")
        for action in Action:
            if action_name(action) == normalized:
                return action
        raise ValueError(f"Unknown action: {value}")

    def _card_payload(self, card: Card) -> dict:
        return {
            "code": str(card),
            "rank": card.rank,
            "suit": card.suit,
            "color": "red" if card.suit in ("H", "D") else "black",
        }
