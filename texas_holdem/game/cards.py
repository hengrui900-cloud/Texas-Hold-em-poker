from __future__ import annotations

from dataclasses import dataclass
import random


SUITS = ("S", "H", "D", "C")
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
RANK_TO_VALUE = {rank: value for value, rank in enumerate(RANKS, start=2)}


@dataclass(frozen=True, order=True)
class Card:
    suit: str
    rank: str

    def __post_init__(self):
        if self.suit not in SUITS:
            raise ValueError(f"Unknown suit: {self.suit}")
        if self.rank not in RANKS:
            raise ValueError(f"Unknown rank: {self.rank}")

    @classmethod
    def from_str(cls, value: str) -> "Card":
        if len(value) != 2:
            raise ValueError(f"Card must be two characters like SA or HT: {value}")
        return cls(suit=value[0].upper(), rank=value[1].upper())

    @property
    def rank_value(self) -> int:
        return RANK_TO_VALUE[self.rank]

    @property
    def index(self) -> int:
        return SUITS.index(self.suit) * len(RANKS) + RANKS.index(self.rank)

    def __str__(self) -> str:
        return f"{self.suit}{self.rank}"


class Deck:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self.cards = [Card(suit, rank) for suit in SUITS for rank in RANKS]
        self._rng.shuffle(self.cards)

    def deal(self) -> Card:
        if not self.cards:
            raise IndexError("Cannot deal from an empty deck")
        return self.cards.pop()

    def __len__(self) -> int:
        return len(self.cards)
