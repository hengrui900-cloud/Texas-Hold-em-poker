from __future__ import annotations

import numpy as np

from texas_holdem.actions import Action, action_name
from texas_holdem.game.cards import Card
from texas_holdem.game.engine import TexasHoldemGame


BASE_OBSERVATION_SIZE = 52 + 52 + 5
PUBLIC_CONTEXT_OFFSET = BASE_OBSERVATION_SIZE
PUBLIC_CONTEXT_SIZE = 17
OBSERVATION_SIZE = BASE_OBSERVATION_SIZE + PUBLIC_CONTEXT_SIZE


def fill_public_context_features(obs: np.ndarray, game, player_id: int, normalizer: float) -> None:
    """Encode only public betting/table context after the original 109 features.

    These features intentionally avoid hidden opponent cards. They give the agent
    a compact view of mutually observable betting pressure and recent actions so
    it can infer ranges from behavior instead of treating every hand as isolated.
    """

    if obs.shape[0] <= PUBLIC_CONTEXT_OFFSET:
        return

    players = list(game.players)
    player = players[player_id]
    opponents = [other for other in players if other.player_id != player_id]
    normalizer = max(1.0, float(normalizer))
    table_size = max(1, len(players))
    call_amount = max(0, game.current_bet - player.committed)
    recent_actions = list(game.action_history[-6:])
    recent_counts = {int(action): 0 for action in Action}
    for item in recent_actions:
        recorded_action = item.get("action")
        for action in Action:
            if recorded_action == action_name(action):
                recent_counts[int(action)] += 1
                break

    features = [
        player.committed / normalizer,
        player.total_committed / normalizer,
        max((other.committed for other in opponents), default=0) / normalizer,
        max((other.total_committed for other in opponents), default=0) / normalizer,
        min((other.stack for other in opponents), default=0) / normalizer,
        call_amount / normalizer,
        game.current_bet / normalizer,
        1.0 if game.current_player == player_id else 0.0,
        sum(1 for item in players if not item.folded) / table_size,
        sum(1 for item in players if item.folded) / table_size,
        sum(1 for item in players if item.all_in) / table_size,
        min(6, len(game.action_history)) / 6.0,
    ]
    features.extend(recent_counts[int(action)] / 6.0 for action in Action)

    end = min(obs.shape[0], PUBLIC_CONTEXT_OFFSET + len(features))
    obs[PUBLIC_CONTEXT_OFFSET:end] = np.asarray(features[: end - PUBLIC_CONTEXT_OFFSET], dtype=np.float32)


class TexasHoldemEnv:
    def __init__(self, seed: int | None = None, starting_stack: int = 100):
        self.seed = seed
        self.starting_stack = starting_stack
        self.game = TexasHoldemGame(seed=seed, starting_stack=starting_stack)
        self.observation_size = OBSERVATION_SIZE

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
        fill_public_context_features(obs, self.game, player_id, normalizer)

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
