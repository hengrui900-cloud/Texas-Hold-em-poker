from __future__ import annotations

from pathlib import Path
import math
import random

import numpy as np

from texas_holdem.actions import Action, action_name
from texas_holdem.agents import DQNAgent
from texas_holdem.env import OBSERVATION_SIZE, fill_public_context_features
from texas_holdem.game.cards import Card
from texas_holdem.game.evaluator import evaluate_seven
from texas_holdem.multiplayer import MultiplayerHoldemGame


ACTION_LABELS = {
    "fold": "弃牌",
    "check_call": "过牌 / 跟注",
    "raise_half_pot": "加注半池",
    "raise_pot": "加注底池",
    "all_in": "全押",
}

STARTING_STACK = 1000
SMALL_BLIND = 20
BIG_BLIND = 80
CHIP_RACK = (
    {"value": 20, "count": 1},
    {"value": 80, "count": 1},
    {"value": 100, "count": 5},
    {"value": 200, "count": 2},
)


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
        self.bankrolls = [STARTING_STACK for _ in range(self.ai_count + 1)]
        self.current_hand_seed: int | None = None
        self._used_hand_seeds: set[int] = set()
        self._system_random = random.SystemRandom()
        self._settled_hand_number: int | None = None
        self.bankrupt = False
        self.new_hand()

    def reset_game(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.seed = int(seed)
        self.hand_number = 0
        self.results = []
        self.last_ai_info = self._empty_ai_info()
        self.bankrolls = [STARTING_STACK for _ in range(self.ai_count + 1)]
        self.current_hand_seed = None
        self.game = None
        self._used_hand_seeds.clear()
        self._settled_hand_number = None
        self.bankrupt = False
        return self.new_hand()

    def new_hand(self, ai_count: int | None = None, seed: int | None = None) -> dict:
        if ai_count is not None:
            next_count = max(1, min(5, int(ai_count)))
            if next_count != self.ai_count:
                self.ai_count = next_count
                self.bankrolls = [STARTING_STACK for _ in range(self.ai_count + 1)]
                self.hand_number = 0
                self.results = []
                self.bankrupt = False
        if seed is not None:
            self.seed = int(seed)

        self._settle_current_hand()
        if any(stack <= 0 for stack in self.bankrolls):
            self.bankrupt = True
            return self.state()

        self.hand_number += 1
        self.current_hand_seed = self._fresh_hand_seed()
        self.game = MultiplayerHoldemGame(
            seed=self.current_hand_seed,
            num_players=self.ai_count + 1,
            starting_stack=STARTING_STACK,
            starting_stacks=list(self.bankrolls),
            small_blind=SMALL_BLIND,
            big_blind=BIG_BLIND,
        )
        self.last_ai_info = self._empty_ai_info()
        self._settled_hand_number = None
        self.bankrupt = False
        return self.state()

    def act(self, action_name_or_id: str | int) -> dict:
        if self.game is None:
            return self.new_hand()
        if self.game.terminal or self.bankrupt:
            return self.state()
        if self.game.current_player != 0:
            return self.state()
        action = self._parse_action(action_name_or_id)
        self.game.step(action)
        return self.state()

    def ai_act(self) -> dict:
        if self.game is None:
            return self.new_hand()
        if self.game.terminal or self.bankrupt or self.game.current_player == 0:
            return self.state()
        player_id = self.game.current_player
        action, info = self._choose_ai_action(player_id)
        self.last_ai_info = info
        self.game.step(action)
        return self.state()

    def state(self) -> dict:
        if self.game is None:
            return self.new_hand()

        self._settle_current_hand()
        game = self.game
        legal = [action_name(action) for action in game.legal_actions()] if game.current_player == 0 and not self.bankrupt else []
        return {
            "hand_number": self.hand_number,
            "hand_seed": self.current_hand_seed,
            "bankrolls": list(self.bankrolls),
            "bankrupt": self.bankrupt,
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
            "turn_timer": self._turn_timer_payload(),
            "win_loss_trend": self._trend_payload(),
            "payoffs": list(game.payoffs),
            "stakes": self._stakes_payload(),
            "chip_rack": list(CHIP_RACK),
            "rule_tip": self._rule_tip(),
            "checkpoint_loaded": self.dqn_agent is not None,
            "checkpoint_path": str(self.checkpoint_path),
        }

    def _choose_ai_action(self, player_id: int) -> tuple[Action, dict]:
        assert self.game is not None
        personality = self._personality(player_id)
        legal = self.game.legal_actions()
        q_values = self._synthetic_q_values(legal, personality)
        intent = "控池 / 等待价值"
        confidence = 0.52

        if personality == "DQN" and self.dqn_agent is not None:
            obs = self._dqn_observation(player_id)
            raw_q = self.dqn_agent.predict(obs)
            q_values = {action_name(action): float(raw_q[int(action)]) for action in legal}
            observation = {"obs": obs, "legal_actions": legal}
            action = self.dqn_agent.act(observation, legal_actions=legal, training=False)
            confidence = self._confidence([raw_q[int(candidate)] for candidate in legal], raw_q[int(action)])
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
        size = int(getattr(self.dqn_agent, "state_size", OBSERVATION_SIZE) or OBSERVATION_SIZE)
        obs = np.zeros(size, dtype=np.float32)
        for card in state["hand"]:
            obs[card.index] = 1.0
        for card in state["public_cards"]:
            obs[52 + card.index] = 1.0
        normalizer = float(max(STARTING_STACK, max(self.bankrolls, default=STARTING_STACK)))
        if size >= 109:
            obs[104] = state["my_stack"] / normalizer
            obs[105] = state["opponent_stack"] / normalizer
            obs[106] = state["pot"] / (2.0 * normalizer)
            obs[107] = float(state["stage"]) / 4.0
            obs[108] = self.game.current_bet / normalizer
        fill_public_context_features(obs, self.game, player_id, normalizer)
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
        human_legal = [action_name(item) for item in self.game.legal_actions()] if self.game.current_player == 0 and not self.bankrupt else []
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
                "enabled": key in human_legal,
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

    def _turn_timer_payload(self) -> dict | None:
        assert self.game is not None
        if self.game.terminal or self.bankrupt:
            return None
        return {
            "player": self.game.current_player,
            "seconds": 15 if self.game.current_player == 0 else 5,
        }

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

    def _stakes_payload(self) -> dict:
        return {
            "currency": "$",
            "starting_stack": STARTING_STACK,
            "small_blind": SMALL_BLIND,
            "big_blind": BIG_BLIND,
        }

    def _rule_tip(self) -> str:
        assert self.game is not None
        game = self.game
        if self.bankrupt:
            return "有一方已经资产归0，请点击“重置游戏”。"
        if game.terminal:
            winner = max(range(len(game.payoffs)), key=lambda index: game.payoffs[index])
            payoff = game.payoffs[winner]
            return f"{self._player_name(winner)} 赢得 {self._money(max(0, payoff))} 筹码。点击“开始游戏”进入下一小局，资产会继承。"
        if game.current_player == 0:
            call = game.call_amount(0)
            if call > 0:
                return f"请下注：当前需要跟注 {self._money(call)}，也可以加注、全押或弃牌。你有 15 秒行动时间。"
            return "请下注：你可以过牌、点击筹码加注，或选择全押。你有 15 秒行动时间。"
        return f"{self._player_name(game.current_player)} 正在思考，请等待 5 秒。"

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
        if len(cards) >= 7:
            rank = evaluate_seven(cards[:7])
            return rank.name.replace("_", " ").title()
        ranks = sorted((card.rank for card in self.game.players[player_id].hand), reverse=True)
        return f"公开信息推断中（手牌 {', '.join(ranks)}）"

    def _confidence(self, values: list[float], chosen: float) -> float:
        if not values:
            return 0.0
        exp = [math.exp(value - max(values)) for value in values]
        total = sum(exp)
        return exp[values.index(chosen)] / total if total else 0.0

    def _intent_for_action(self, action: Action) -> str:
        return {
            Action.FOLD: "风险控制 / 弃牌",
            Action.CHECK_CALL: "控池 / 过牌跟注",
            Action.RAISE_HALF_POT: "施压 / 半池加注",
            Action.RAISE_POT: "价值施压 / 底池加注",
            Action.ALL_IN: "最大压力 / 全押",
        }[action]

    def _settle_current_hand(self) -> None:
        if self.game is None or not self.game.terminal:
            return
        if self._settled_hand_number == self.hand_number:
            return
        self.bankrolls = [player.stack for player in self.game.players]
        initial = max(1, self.game.initial_stacks[0])
        self.results.append(self.game.payoffs[0] / float(initial))
        self._settled_hand_number = self.hand_number
        self.bankrupt = any(stack <= 0 for stack in self.bankrolls)

    def _fresh_hand_seed(self) -> int:
        while True:
            candidate = self._system_random.randrange(1, 2**31)
            if candidate not in self._used_hand_seeds:
                self._used_hand_seeds.add(candidate)
                return candidate

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
            "name": "等待中",
            "personality": "混合",
            "street": "翻牌前",
            "intent": "等待行动",
            "confidence": 0.0,
            "q_values": [],
            "hand_range": "AI 尚未行动",
        }

    def _player_name(self, player_id: int) -> str:
        if player_id == 0:
            return "你"
        return f"AI-Bot {player_id}"

    def _money(self, amount: int | float) -> str:
        return f"${int(round(amount))}"

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
