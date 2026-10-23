

from collections import deque
import random 
import numpy as np

from Helpers.Hyperparameters import HyperparameterData

class Memory:
    def __init__(self, hyperparameters : HyperparameterData):
        self.capacity = hyperparameters.buffer_size

        self.states = deque([], maxlen=self.capacity)
        self.actions = deque([], maxlen=self.capacity)
        self.next_states = deque([], maxlen=self.capacity)
        self.rewards = deque([], maxlen=self.capacity)
        self.isTerminateds = deque([], maxlen=self.capacity)

    def push(self, old_state, action, state, reward, isTerminated):
        self.states.append(old_state)
        self.actions.append(action)
        self.next_states.append(state.tolist())
        self.rewards.append(reward)
        self.isTerminateds.append(isTerminated)

    def _get_indexes(self, lst, samples):
        return [lst[i] for i in samples]

    def sample(self, size):
        samples = np.random.randint(0, len(self.states)-1, (size,))
        return self._get_indexes(self.states, samples), self._get_indexes(self.actions, samples), self._get_indexes(self.next_states, samples), self._get_indexes(self.rewards, samples),self._get_indexes(self.isTerminateds, samples)


