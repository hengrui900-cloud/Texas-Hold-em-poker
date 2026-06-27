from __future__ import annotations

import numpy as np

from texas_holdem.actions import Action
from texas_holdem.game.cards import Card
from texas_holdem.game.engine import TexasHoldemGame


class TexasHoldemEnv:
    def __init__(self, seed: int | None = None, starting_stack: int = 100):
        self.seed = seed
        self.starting_stack = starting_stack
        self.game = TexasHoldemGame(seed=seed, starting_stack=starting_stack)
        self.observation_size = 52 + 52 + 5

    @property
    def current_player(self) -> int:
        return self.game.current_player

    def reset(self, seed: int | None = None) -> dict:
        self.game.reset(seed=self.seed if seed is None else seed)
        return self.observe()

    def observe(self, player_id: int | None = None) -> dict:
        if player_id is None:
            player_id = self.game.current_player
        state = self.game.get_state(player_id)
        obs = np.zeros(self.observation_size, dtype=np.float32)
        for card in state["hand"]:
            obs[card.index] = 1.0
        for card in state["public_cards"]:
            obs[52 + card.index] = 1.0

        feature_offset = 104
        normalizer = float(self.starting_stack)
        obs[feature_offset] = state["my_stack"] / normalizer
        obs[feature_offset + 1] = state["opponent_stack"] / normalizer
        obs[feature_offset + 2] = state["pot"] / (2.0 * normalizer)
        obs[feature_offset + 3] = float(state["stage"]) / 4.0
        obs[feature_offset + 4] = self.game.current_bet / normalizer

        legal = state["legal_actions"] if player_id == self.current_player else []
        mask = np.zeros(len(Action), dtype=np.float32)
        for action in legal:
            mask[int(action)] = 1.0

        return {
            "obs": obs,
            "action_mask": mask,
            "legal_actions": legal,
            "raw_state": state,
        }

    def legal_actions(self) -> list[Action]:
        return self.game.legal_actions()

    def step(self, action: Action | int):
        self.game.step(Action(action))
        done = self.game.terminal
        reward = self.game.payoffs[0] / float(self.starting_stack) if done else 0.0
        info = {
            "payoffs": list(self.game.payoffs),
            "public_cards": list(self.game.public_cards),
            "action_history": list(self.game.action_history),
        }
        return self.observe(), reward, done, info

    def render(self) -> str:
        return self.game.render()


def format_cards(cards: list[Card]) -> str:
    return " ".join(str(card) for card in cards) or "-"
