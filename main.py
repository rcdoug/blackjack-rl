from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch 
from tqdm import tqdm

import gymnasium as gym

env = gym.make("Blackjack-v1", sab=True)

# reset the environment to get the first observation
done = False
observation, info = env.reset()

# observation = (16, 9, False)

class BlackjackAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """
            Init RL agent with empty dictionary of state-action values (q_values)
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    
    def get_action(self, obs: tuple[int, int, bool]) -> int:
        if np.random.random() < self.epsilon:
                return env.action_space.sample()
        
        else:
            return int(np.argmax(self.q_values[obs]))
