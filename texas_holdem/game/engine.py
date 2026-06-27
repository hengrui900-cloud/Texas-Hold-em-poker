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


class TexasHoldemGame:
    def __init__(
        self,
        seed: int | None = None,
        starting_stack: int = 100,
        small_blind: int = 1,
        big_blind: int = 2,
    ):
        self.seed = seed
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.rng = random.Random(seed)
        self.players: list[Player] = []
        self.public_cards: list[Card] = []
        self.action_history: list[dict] = []
        self.payoffs = [0, 0]
        self.reset(seed=seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)
        deck_seed = self.rng.randrange(2**31)
        self.deck = Deck(seed=deck_seed)
        self.players = [Player(0, self.starting_stack), Player(1, self.starting_stack)]
        self.public_cards = []
        self.action_history = []
        self.payoffs = [0, 0]
        self.pot = 0
        self.current_bet = 0
        self.terminal = False
        self.stage = Stage.PREFLOP
        self.dealer = self.rng.randrange(2)
        self.small_blind_player = self.dealer
        self.big_blind_player = 1 - self.dealer
        self.acted = [False, False]

        for _ in range(2):
            for player in self.players:
                player.hand.append(self.deck.deal())

        self._put_chips(self.small_blind_player, self.small_blind)
        self._put_chips(self.big_blind_player, self.big_blind)
        self.current_bet = self.big_blind
        self.current_player = self.small_blind_player
        return self.get_state(self.current_player)

    def get_state(self, player_id: int) -> dict:
        opponent_id = 1 - player_id
        return {
            "player_id": player_id,
            "hand": list(self.players[player_id].hand),
            "public_cards": list(self.public_cards),
            "my_stack": self.players[player_id].stack,
            "opponent_stack": self.players[opponent_id].stack,
            "my_committed": self.players[player_id].committed,
            "opponent_committed": self.players[opponent_id].committed,
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
        stack_after_call = player.stack - diff
        if player.stack > 0:
            legal.append(Action.ALL_IN)
        if stack_after_call <= 0:
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
            self.action_history.append(self._action_record(actor, action, before))
            self._finish_by_fold()
            return self.get_state(self.current_player)

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
            self.acted = [False, False]
        self.acted[actor] = True
        self.action_history.append(self._action_record(actor, action, before))
        self._move_to_next_player()
        self._maybe_advance()
        return self.get_state(self.current_player)

    def _call(self, player_id: int) -> None:
        diff = max(0, self.current_bet - self.players[player_id].committed)
        self._put_chips(player_id, diff)

    def _raise(self, player_id: int, raise_amount: int) -> None:
        self._call(player_id)
        self._put_chips(player_id, raise_amount)

    def _put_chips(self, player_id: int, amount: int) -> int:
        paid = self.players[player_id].bet(amount)
        self.pot += paid
        return paid

    def _move_to_next_player(self) -> None:
        for offset in (1, 2):
            candidate = (self.current_player + offset) % 2
            player = self.players[candidate]
            if not player.folded and not player.all_in:
                self.current_player = candidate
                return

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
        self.acted = [False, False]

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
        if self._all_contenders_all_in():
            self._deal_to_river()
            self._showdown()

    def _first_postflop_actor(self) -> int:
        first = self.big_blind_player
        for offset in (0, 1):
            candidate = (first + offset) % 2
            player = self.players[candidate]
            if not player.folded and not player.all_in:
                return candidate
        return first

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
        active = self._active_players()
        ranks = {
            player_id: evaluate_seven(self.players[player_id].hand + self.public_cards)
            for player_id in active
        }
        best = max(ranks.values())
        winners = [player_id for player_id, rank in ranks.items() if rank.value == best.value]
        share, remainder = divmod(self.pot, len(winners))
        for offset, player_id in enumerate(winners):
            self.players[player_id].stack += share + (1 if offset < remainder else 0)
        self.pot = 0
        self.stage = Stage.SHOWDOWN
        self.terminal = True
        self.payoffs = [player.stack - self.starting_stack for player in self.players]
        self.current_player = winners[0]

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

    def render(self, reveal_hands: bool = True) -> str:
        public = " ".join(str(card) for card in self.public_cards) or "-"
        lines = [
            f"stage={self.stage.name.lower()} pot={self.pot} current_player={self.current_player}",
            f"public: {public}",
        ]
        for player in self.players:
            hand = " ".join(str(card) for card in player.hand) if reveal_hands else "hidden"
            status = "folded" if player.folded else "all-in" if player.all_in else "active"
            lines.append(
                f"p{player.player_id}: stack={player.stack} committed={player.committed} "
                f"status={status} hand={hand}"
            )
        return "\n".join(lines)
