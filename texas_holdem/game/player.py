from __future__ import annotations

from dataclasses import dataclass, field

from texas_holdem.game.cards import Card


@dataclass
class Player:
    player_id: int
    stack: int
    hand: list[Card] = field(default_factory=list)
    committed: int = 0
    total_committed: int = 0
    folded: bool = False
    all_in: bool = False

    def bet(self, amount: int) -> int:
        if amount < 0:
            raise ValueError("Bet amount cannot be negative")
        paid = min(amount, self.stack)
        self.stack -= paid
        self.committed += paid
        self.total_committed += paid
        if self.stack == 0:
            self.all_in = True
        return paid

    def reset_street_commitment(self) -> None:
        self.committed = 0
