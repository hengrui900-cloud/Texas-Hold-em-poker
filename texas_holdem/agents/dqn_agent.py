from __future__ import annotations

from collections import deque, namedtuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from texas_holdem.actions import Action


Transition = namedtuple("Transition", "state action reward next_state done next_legal_actions")


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
    def __init__(self, state_size: int, hidden_size: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


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
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.target_update = target_update
        self.device = resolve_device(device)
        self.rng = np.random.default_rng(seed)
        self.memory = deque(maxlen=replay_size)
        self.steps = 0

        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        self.q_network = QNetwork(state_size, hidden_size, num_actions).to(self.device)
        self.target_network = QNetwork(state_size, hidden_size, num_actions).to(self.device)
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
            return Action(int(self.rng.choice(legal)))

        q_values = self.predict(observation["obs"])
        masked = np.full(self.num_actions, -np.inf, dtype=np.float32)
        masked[legal] = q_values[legal]
        return Action(int(np.argmax(masked)))

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
        min_memory = max(self.batch_size, self.replay_start_size)
        if len(self.memory) < min_memory:
            self.steps += 1
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
            next_q = self.target_network(next_states)
            masked_next_q = torch.full_like(next_q, -1.0e9)
            for row, item in enumerate(batch):
                if item.next_legal_actions:
                    masked_next_q[row, item.next_legal_actions] = next_q[row, item.next_legal_actions]
            max_next = masked_next_q.max(dim=1).values
            max_next = torch.where(done, torch.zeros_like(max_next), max_next)
            targets = rewards + (~done).float() * self.gamma * max_next

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.copy_target()
        return float(loss.detach().cpu().item())

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
                "batch_size": self.batch_size,
                "replay_start_size": self.replay_start_size,
                "target_update": self.target_update,
            },
            "steps": self.steps,
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

        agent = cls(**payload["config"], device=resolved_device)
        agent.steps = payload["steps"]
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
