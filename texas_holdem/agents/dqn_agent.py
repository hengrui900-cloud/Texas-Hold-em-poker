from __future__ import annotations

from collections import deque, namedtuple
import pickle

import numpy as np

from texas_holdem.actions import Action


Transition = namedtuple("Transition", "state action reward next_state done next_legal_actions")


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        num_actions: int = 5,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 2000,
        replay_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 100,
        seed: int | None = None,
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
        self.target_update = target_update
        self.rng = np.random.default_rng(seed)
        self.memory = deque(maxlen=replay_size)
        self.steps = 0

        scale1 = np.sqrt(2.0 / state_size)
        scale2 = np.sqrt(2.0 / hidden_size)
        self.w1 = self.rng.normal(0.0, scale1, size=(state_size, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = self.rng.normal(0.0, scale2, size=(hidden_size, num_actions)).astype(np.float32)
        self.b2 = np.zeros(num_actions, dtype=np.float32)
        self.copy_target()

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.steps / float(max(1, self.epsilon_decay)))
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def act(self, observation: dict, legal_actions=None, training: bool = True):
        legal = [int(action) for action in (legal_actions or observation["legal_actions"])]
        if training and self.rng.random() < self.epsilon:
            return Action(int(self.rng.choice(legal)))
        q_values = self.predict(observation["obs"])
        masked = np.full(self.num_actions, -np.inf, dtype=np.float32)
        masked[legal] = q_values[legal]
        return Action(int(np.argmax(masked)))

    def predict(self, state: np.ndarray, target: bool = False) -> np.ndarray:
        state_batch = np.asarray(state, dtype=np.float32).reshape(1, -1)
        q_values, _ = self._forward(state_batch, target=target)
        return q_values[0]

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
        if len(self.memory) < self.batch_size:
            self.steps += 1
            return None

        indices = self.rng.choice(len(self.memory), size=self.batch_size, replace=False)
        batch = [self.memory[int(index)] for index in indices]
        states = np.stack([item.state for item in batch])
        actions = np.array([item.action for item in batch], dtype=np.int64)
        rewards = np.array([item.reward for item in batch], dtype=np.float32)
        next_states = np.stack([item.next_state for item in batch])
        done = np.array([item.done for item in batch], dtype=bool)

        next_q, _ = self._forward(next_states, target=True)
        max_next = np.zeros(self.batch_size, dtype=np.float32)
        for row, item in enumerate(batch):
            if item.done or not item.next_legal_actions:
                max_next[row] = 0.0
            else:
                max_next[row] = np.max(next_q[row, item.next_legal_actions])

        targets = rewards + (~done).astype(np.float32) * self.gamma * max_next
        q_values, cache = self._forward(states, target=False)
        chosen_q = q_values[np.arange(self.batch_size), actions]
        errors = chosen_q - targets
        loss = float(np.mean(errors**2))

        grad_q = np.zeros_like(q_values)
        grad_q[np.arange(self.batch_size), actions] = (2.0 / self.batch_size) * errors
        self._backward(cache, grad_q)

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.copy_target()
        return loss

    def _forward(self, states: np.ndarray, target: bool = False):
        if target:
            w1, b1, w2, b2 = self.target_w1, self.target_b1, self.target_w2, self.target_b2
        else:
            w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2
        z1 = states @ w1 + b1
        hidden = np.maximum(z1, 0.0)
        q_values = hidden @ w2 + b2
        return q_values, (states, z1, hidden)

    def _backward(self, cache, grad_q):
        states, z1, hidden = cache
        grad_w2 = hidden.T @ grad_q
        grad_b2 = grad_q.sum(axis=0)
        grad_hidden = grad_q @ self.w2.T
        grad_z1 = grad_hidden * (z1 > 0)
        grad_w1 = states.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)

        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1

    def copy_target(self):
        self.target_w1 = self.w1.copy()
        self.target_b1 = self.b1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b2 = self.b2.copy()

    def save(self, path):
        payload = {
            "state_size": self.state_size,
            "num_actions": self.num_actions,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "target_update": self.target_update,
            "steps": self.steps,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }
        with open(path, "wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            payload = pickle.load(file)
        agent = cls(
            state_size=payload["state_size"],
            num_actions=payload["num_actions"],
            hidden_size=payload["hidden_size"],
            learning_rate=payload["learning_rate"],
            gamma=payload["gamma"],
            epsilon_start=payload["epsilon_start"],
            epsilon_end=payload["epsilon_end"],
            epsilon_decay=payload["epsilon_decay"],
            batch_size=payload["batch_size"],
            target_update=payload["target_update"],
        )
        agent.steps = payload["steps"]
        agent.w1 = payload["w1"]
        agent.b1 = payload["b1"]
        agent.w2 = payload["w2"]
        agent.b2 = payload["b2"]
        agent.copy_target()
        return agent
