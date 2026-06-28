from __future__ import annotations

from collections import namedtuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from texas_holdem.actions import Action


Transition = namedtuple("Transition", "state action reward next_state done next_legal_actions")


class ReplayMemory:
    supports_fast_random_access = True

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("Replay memory capacity must be at least 1.")
        self.capacity = int(capacity)
        self.items = []
        self.position = 0

    def append(self, item: Transition) -> None:
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            self.items[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Transition:
        return self.items[index]

    def __iter__(self):
        return iter(self.items)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        requested = device.type
    else:
        requested = str(device).lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


class QNetwork(nn.Module):
    def __init__(self, state_size: int, hidden_size: int, num_actions: int, dueling: bool = False):
        super().__init__()
        self.dueling = dueling
        if dueling:
            self.feature = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.value = nn.Linear(hidden_size, 1)
            self.advantage = nn.Linear(hidden_size, num_actions)
        else:
            self.net = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_actions),
            )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if not self.dueling:
            return self.net(states)
        features = self.feature(states)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        num_actions: int = 5,
        hidden_size: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 2000,
        replay_size: int = 10000,
        batch_size: int = 64,
        replay_start_size: int = 64,
        target_update: int = 100,
        update_every: int = 1,
        dueling: bool = False,
        double_dqn: bool = False,
        all_in_margin: float = 0.20,
        exploration_all_in_probability: float = 0.03,
        seed: int | None = None,
        device: str | torch.device = "auto",
    ):
        self.state_size = state_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.target_update = target_update
        self.update_every = max(1, int(update_every))
        self.dueling = dueling
        self.double_dqn = double_dqn
        self.all_in_margin = float(all_in_margin)
        self.exploration_all_in_probability = float(exploration_all_in_probability)
        self.device = resolve_device(device)
        self.rng = np.random.default_rng(seed)
        self.memory = ReplayMemory(replay_size)
        self.steps = 0
        self.updates = 0

        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        self.q_network = QNetwork(state_size, hidden_size, num_actions, dueling=dueling).to(self.device)
        self.target_network = QNetwork(state_size, hidden_size, num_actions, dueling=dueling).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.copy_target()

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.steps / float(max(1, self.epsilon_decay)))
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return f"cuda:{torch.cuda.get_device_name(self.device)}"
        return str(self.device)

    def act(self, observation: dict, legal_actions=None, training: bool = True):
        legal = [int(action) for action in (legal_actions or observation["legal_actions"])]
        if not legal:
            raise ValueError("Cannot act without legal actions")
        if training and self.rng.random() < self.epsilon:
            return Action(int(self.rng.choice(self._exploration_legal_actions(legal))))

        q_values = self.predict(observation["obs"])
        masked = np.full(self.num_actions, -np.inf, dtype=np.float32)
        masked[legal] = q_values[legal]
        chosen = int(np.argmax(masked))
        return Action(self._risk_adjusted_action(chosen, q_values, legal))

    def _exploration_legal_actions(self, legal: list[int]) -> list[int]:
        all_in = int(Action.ALL_IN)
        if all_in not in legal or len(legal) == 1:
            return legal
        if self.rng.random() < self.exploration_all_in_probability:
            return legal
        safer = [action for action in legal if action != all_in]
        return safer or legal

    def _risk_adjusted_action(self, chosen: int, q_values: np.ndarray, legal: list[int]) -> int:
        all_in = int(Action.ALL_IN)
        if chosen != all_in or all_in not in legal or self.all_in_margin <= 0:
            return chosen
        safer = [action for action in legal if action != all_in]
        if not safer:
            return chosen
        best_safer = max(safer, key=lambda action: float(q_values[action]))
        edge = float(q_values[all_in]) - float(q_values[best_safer])
        if edge <= self.all_in_margin:
            return int(best_safer)
        return chosen

    def predict(self, state: np.ndarray) -> np.ndarray:
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
            q_values = self.q_network(state_tensor)
        return q_values.squeeze(0).detach().cpu().numpy()

    def remember(self, state, action, reward, next_state, done, next_legal_actions):
        self.memory.append(
            Transition(
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
                [int(action) for action in next_legal_actions],
            )
        )

    def train_step(self):
        self.steps += 1
        min_memory = max(self.batch_size, self.replay_start_size)
        if len(self.memory) < min_memory:
            return None
        if self.steps % self.update_every != 0:
            return None

        indices = self.rng.choice(len(self.memory), size=self.batch_size, replace=False)
        batch = [self.memory[int(index)] for index in indices]
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device)
        done = torch.as_tensor([item.done for item in batch], dtype=torch.bool, device=self.device)

        self.q_network.train()
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_target_q = self.target_network(next_states)
            if self.double_dqn:
                next_policy_q = self.q_network(next_states)
                masked_policy_q = self._masked_next_q(next_policy_q, batch)
                best_next_actions = masked_policy_q.argmax(dim=1)
                max_next = next_target_q.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
            else:
                masked_target_q = self._masked_next_q(next_target_q, batch)
                max_next = masked_target_q.max(dim=1).values
            max_next = torch.where(done, torch.zeros_like(max_next), max_next)
            targets = rewards + (~done).float() * self.gamma * max_next

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.target_update == 0:
            self.copy_target()
        return float(loss.detach().cpu().item())

    def _masked_next_q(self, next_q: torch.Tensor, batch) -> torch.Tensor:
        masked_next_q = torch.full_like(next_q, -1.0e9)
        for row, item in enumerate(batch):
            if item.next_legal_actions:
                masked_next_q[row, item.next_legal_actions] = next_q[row, item.next_legal_actions]
        return masked_next_q

    def copy_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def save(self, path):
        payload = {
            "format": "texas_holdem_pytorch_dqn_v1",
            "config": {
                "state_size": self.state_size,
                "num_actions": self.num_actions,
                "hidden_size": self.hidden_size,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "replay_size": self.replay_size,
                "batch_size": self.batch_size,
                "replay_start_size": self.replay_start_size,
                "target_update": self.target_update,
                "update_every": self.update_every,
                "dueling": self.dueling,
                "double_dqn": self.double_dqn,
                "all_in_margin": self.all_in_margin,
                "exploration_all_in_probability": self.exploration_all_in_probability,
            },
            "steps": self.steps,
            "updates": self.updates,
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device: str | torch.device = "auto"):
        resolved_device = resolve_device(device)
        try:
            payload = torch.load(path, map_location=resolved_device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=resolved_device)
        if payload.get("format") != "texas_holdem_pytorch_dqn_v1":
            raise ValueError("Unsupported checkpoint format. Retrain with the PyTorch DQN version.")

        config = dict(payload["config"])
        config.setdefault("all_in_margin", 0.20)
        config.setdefault("exploration_all_in_probability", 0.03)
        agent = cls(**config, device=resolved_device)
        agent.steps = payload["steps"]
        agent.updates = payload.get("updates", 0)
        agent.q_network.load_state_dict(payload["q_network"])
        agent.target_network.load_state_dict(payload["target_network"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        for state in agent.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(agent.device)
        agent.q_network.to(agent.device)
        agent.target_network.to(agent.device)
        return agent
