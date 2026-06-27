from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from texas_holdem.game.cards import Card


CATEGORY_NAMES = {
    0: "high_card",
    1: "pair",
    2: "two_pair",
    3: "three_of_a_kind",
    4: "straight",
    5: "flush",
    6: "full_house",
    7: "four_of_a_kind",
    8: "straight_flush",
}


@dataclass(frozen=True)
class HandRank:
    category: int
    kickers: tuple[int, ...]

    @property
    def name(self) -> str:
        return CATEGORY_NAMES[self.category]

    @property
    def value(self) -> tuple[int, tuple[int, ...]]:
        return self.category, self.kickers

    def __lt__(self, other: "HandRank") -> bool:
        return self.value < other.value

    def __le__(self, other: "HandRank") -> bool:
        return self.value <= other.value

    def __gt__(self, other: "HandRank") -> bool:
        return self.value > other.value

    def __ge__(self, other: "HandRank") -> bool:
        return self.value >= other.value


def evaluate_seven(cards: list[Card]) -> HandRank:
    if len(cards) != 7:
        raise ValueError("Texas Hold'em evaluation requires exactly seven cards")
    return max(_evaluate_five(list(candidate)) for candidate in combinations(cards, 5))


def _straight_high(ranks: list[int]) -> int | None:
    unique = sorted(set(ranks), reverse=True)
    if 14 in unique:
        unique.append(1)
    for i in range(len(unique) - 4):
        window = unique[i : i + 5]
        if window[0] - window[-1] == 4 and len(set(window)) == 5:
            return 5 if window[0] == 5 else window[0]
    return None


def _evaluate_five(cards: list[Card]) -> HandRank:
    ranks = sorted((card.rank_value for card in cards), reverse=True)
    counts = Counter(ranks)
    count_rank = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    is_flush = len({card.suit for card in cards}) == 1
    straight = _straight_high(ranks)

    if is_flush and straight:
        return HandRank(8, (straight,))

    if count_rank[0][1] == 4:
        quad = count_rank[0][0]
        kicker = max(rank for rank in ranks if rank != quad)
        return HandRank(7, (quad, kicker))

    triples = sorted((rank for rank, count in counts.items() if count == 3), reverse=True)
    pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if triples and pairs:
        return HandRank(6, (triples[0], pairs[0]))

    if is_flush:
        return HandRank(5, tuple(ranks))

    if straight:
        return HandRank(4, (straight,))

    if triples:
        triple = triples[0]
        kickers = tuple(rank for rank in ranks if rank != triple)
        return HandRank(3, (triple, *kickers))

    if len(pairs) >= 2:
        top_two = tuple(pairs[:2])
        kicker = max(rank for rank in ranks if rank not in top_two)
        return HandRank(2, (*top_two, kicker))

    if pairs:
        pair = pairs[0]
        kickers = tuple(rank for rank in ranks if rank != pair)
        return HandRank(1, (pair, *kickers))

    return HandRank(0, tuple(ranks))
