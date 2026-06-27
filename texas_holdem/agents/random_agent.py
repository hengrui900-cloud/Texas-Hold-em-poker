from __future__ import annotations

import random


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, observation: dict, legal_actions=None, training: bool = False):
        legal = legal_actions or observation["legal_actions"]
        return self.rng.choice(list(legal))
