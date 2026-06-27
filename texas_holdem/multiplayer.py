from __future__ import annotations

from enum import IntEnum
import random

from texas_holdem.actions import Action, action_name
from texas_holdem.game.cards import Card, Deck
from texas_holdem.game.evaluator import evaluate_seven
from texas_holdem.game.player import Player


class Stage(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


class MultiplayerHoldemGame:
    def __init__(
        self,
        seed: int | None = None,
        num_players: int = 4,
        starting_stack: int = 100,
        small_blind: int = 1,
        big_blind: int = 2,
    ):
        if not 2 <= num_players <= 6:
            raise ValueError("Texas Hold'em table supports 2 to 6 players")
        self.seed = seed
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.rng = random.Random(seed)
        self.reset(seed=seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)

        self.deck = Deck(seed=self.rng.randrange(2**31))
        self.players = [Player(player_id, self.starting_stack) for player_id in range(self.num_players)]
        self.public_cards: list[Card] = []
        self.action_history: list[dict] = []
        self.payoffs = [0 for _ in range(self.num_players)]
        self.pot = 0
        self.current_bet = 0
        self.terminal = False
        self.stage = Stage.PREFLOP
        self.acted = [False for _ in range(self.num_players)]

        self.dealer = self.rng.randrange(self.num_players)
        self.small_blind_player = self.dealer if self.num_players == 2 else (self.dealer + 1) % self.num_players
        self.big_blind_player = (self.dealer + 1) % self.num_players if self.num_players == 2 else (self.dealer + 2) % self.num_players

        for _ in range(2):
            for player in self.players:
                player.hand.append(self.deck.deal())

        self._put_chips(self.small_blind_player, self.small_blind)
        self._put_chips(self.big_blind_player, self.big_blind)
        self.current_bet = self.big_blind
        start = self.small_blind_player if self.num_players == 2 else (self.big_blind_player + 1) % self.num_players
        self.current_player = self._next_actor_from(start)
        return self.get_state(self.current_player)

    def get_state(self, player_id: int) -> dict:
        other_stacks = [player.stack for player in self.players if player.player_id != player_id]
        return {
            "player_id": player_id,
            "hand": list(self.players[player_id].hand),
            "public_cards": list(self.public_cards),
            "my_stack": self.players[player_id].stack,
            "opponent_stack": max(other_stacks) if other_stacks else 0,
            "my_committed": self.players[player_id].committed,
            "opponent_committed": max(
                (player.committed for player in self.players if player.player_id != player_id),
                default=0,
            ),
            "pot": self.pot,
            "stage": self.stage,
            "current_player": self.current_player,
            "legal_actions": self.legal_actions(),
            "payoffs": list(self.payoffs),
            "terminal": self.terminal,
        }

    def legal_actions(self) -> list[Action]:
        if self.terminal:
            return []
        player = self.players[self.current_player]
        if player.folded or player.all_in:
            return []

        legal = [Action.FOLD, Action.CHECK_CALL]
        diff = max(0, self.current_bet - player.committed)
        if player.stack > 0:
            legal.append(Action.ALL_IN)
        if player.stack - diff <= 0:
            return legal

        half_pot_raise = max(1, self.pot // 2)
        pot_raise = max(1, self.pot)
        if player.stack >= diff + half_pot_raise and player.committed + diff + half_pot_raise > self.current_bet:
            legal.append(Action.RAISE_HALF_POT)
        if player.stack >= diff + pot_raise and player.committed + diff + pot_raise > self.current_bet:
            legal.append(Action.RAISE_POT)
        return sorted(legal, key=int)

    def step(self, action: Action | int):
        action = Action(action)
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action_name(action)} for player {self.current_player}")

        actor = self.current_player
        before = self._snapshot()

        if action == Action.FOLD:
            self.players[actor].folded = True
            self.acted[actor] = True
        else:
            old_bet = self.current_bet
            if action == Action.CHECK_CALL:
                self._call(actor)
            elif action == Action.RAISE_HALF_POT:
                self._raise(actor, max(1, self.pot // 2))
            elif action == Action.RAISE_POT:
                self._raise(actor, max(1, self.pot))
            elif action == Action.ALL_IN:
                self._put_chips(actor, self.players[actor].stack)

            if self.players[actor].committed > old_bet:
                self.current_bet = self.players[actor].committed
                self.acted = [False for _ in range(self.num_players)]
            self.acted[actor] = True

        self.action_history.append(self._action_record(actor, action, before))
        previous_stage = self.stage
        self._maybe_advance()
        if not self.terminal and self.stage == previous_stage:
            self.current_player = self._next_actor_from((actor + 1) % self.num_players)
            self._maybe_advance()
        return self.get_state(self.current_player)

    def call_amount(self, player_id: int | None = None) -> int:
        if player_id is None:
            player_id = self.current_player
        return max(0, self.current_bet - self.players[player_id].committed)

    def raise_amounts(self, player_id: int | None = None) -> dict[str, int]:
        if player_id is None:
            player_id = self.current_player
        player = self.players[player_id]
        diff = self.call_amount(player_id)
        return {
            "half_pot": min(player.stack, diff + max(1, self.pot // 2)),
            "pot": min(player.stack, diff + max(1, self.pot)),
            "all_in": player.stack,
        }

    def _call(self, player_id: int) -> None:
        self._put_chips(player_id, self.call_amount(player_id))

    def _raise(self, player_id: int, raise_amount: int) -> None:
        self._call(player_id)
        self._put_chips(player_id, raise_amount)

    def _put_chips(self, player_id: int, amount: int) -> int:
        paid = self.players[player_id].bet(amount)
        self.pot += paid
        return paid

    def _maybe_advance(self) -> None:
        if self.terminal:
            return
        if len(self._active_players()) == 1:
            self._finish_by_fold()
            return
        if self._all_contenders_all_in():
            self._deal_to_river()
            self._showdown()
            return
        if self._betting_round_over():
            if self.stage == Stage.RIVER:
                self._showdown()
            else:
                self._advance_stage()

    def _betting_round_over(self) -> bool:
        for index, player in enumerate(self.players):
            if player.folded or player.all_in:
                continue
            if not self.acted[index]:
                return False
            if player.committed != self.current_bet:
                return False
        return True

    def _advance_stage(self) -> None:
        for player in self.players:
            player.reset_street_commitment()
        self.current_bet = 0
        self.acted = [False for _ in range(self.num_players)]

        if self.stage == Stage.PREFLOP:
            self.public_cards.extend(self.deck.deal() for _ in range(3))
            self.stage = Stage.FLOP
        elif self.stage == Stage.FLOP:
            self.public_cards.append(self.deck.deal())
            self.stage = Stage.TURN
        elif self.stage == Stage.TURN:
            self.public_cards.append(self.deck.deal())
            self.stage = Stage.RIVER

        self.current_player = self._first_postflop_actor()

    def _next_actor_from(self, start: int) -> int:
        for offset in range(self.num_players):
            candidate = (start + offset) % self.num_players
            player = self.players[candidate]
            if not player.folded and not player.all_in:
                return candidate
        return start % self.num_players

    def _first_postflop_actor(self) -> int:
        return self._next_actor_from((self.dealer + 1) % self.num_players)

    def _active_players(self) -> list[int]:
        return [index for index, player in enumerate(self.players) if not player.folded]

    def _all_contenders_all_in(self) -> bool:
        active = [self.players[index] for index in self._active_players()]
        return bool(active) and all(player.all_in for player in active)

    def _finish_by_fold(self) -> None:
        winner = self._active_players()[0]
        self.players[winner].stack += self.pot
        self.pot = 0
        self.terminal = True
        self.stage = Stage.SHOWDOWN
        self.payoffs = [player.stack - self.starting_stack for player in self.players]
        self.current_player = winner

    def _deal_to_river(self) -> None:
        while len(self.public_cards) < 5:
            if len(self.public_cards) == 0:
                self.public_cards.extend(self.deck.deal() for _ in range(3))
                self.stage = Stage.FLOP
            else:
                self.public_cards.append(self.deck.deal())
                self.stage = Stage.TURN if len(self.public_cards) == 4 else Stage.RIVER

    def _showdown(self) -> None:
        active = set(self._active_players())
        ranks = {
            player_id: evaluate_seven(self.players[player_id].hand + self.public_cards)
            for player_id in active
        }
        contributions = [player.total_committed for player in self.players]
        previous = 0
        for level in sorted({amount for amount in contributions if amount > 0}):
            contestants = [i for i, amount in enumerate(contributions) if amount >= level]
            eligible = [i for i in contestants if i in active]
            side_pot = (level - previous) * len(contestants)
            if eligible and side_pot:
                best = max(ranks[player_id] for player_id in eligible)
                winners = [player_id for player_id in eligible if ranks[player_id].value == best.value]
                share, remainder = divmod(side_pot, len(winners))
                for offset, player_id in enumerate(winners):
                    self.players[player_id].stack += share + (1 if offset < remainder else 0)
            previous = level

        self.pot = 0
        self.stage = Stage.SHOWDOWN
        self.terminal = True
        self.payoffs = [player.stack - self.starting_stack for player in self.players]
        self.current_player = max(range(self.num_players), key=lambda index: self.players[index].stack)

    def _snapshot(self) -> dict:
        return {
            "pot": self.pot,
            "stage": self.stage.name.lower(),
            "stacks": [player.stack for player in self.players],
            "committed": [player.committed for player in self.players],
        }

    def _action_record(self, actor: int, action: Action, before: dict) -> dict:
        return {
            "player": actor,
            "action": action_name(action),
            "before": before,
            "after": self._snapshot(),
        }
