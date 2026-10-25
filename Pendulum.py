


import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import math 

from Helpers.ControllerDDPG import ActorDDPG, CriticDDPG
from Helpers.ControllerDDPG import get_action_array, optimize, hard_weight_update, soft_weight_update

from Helpers.Memory import Memory
from Helpers.Results import Results
from Helpers.Hyperparameters import HyperparameterData

EPISODES = 1000

hyperparameters = HyperparameterData()
hyperparameters.lr = 0.003
hyperparameters.tau = 0.005
hyperparameters.neurons = [3, 32, 32, 1]
hyperparameters.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

replay_memory = Memory(hyperparameters)

tracked_results = Results(EPISODES)

# Networks
policy = ActorDDPG(hyperparameters)
q_function = CriticDDPG(hyperparameters)
policy_target = ActorDDPG(hyperparameters)
q_function_target = CriticDDPG(hyperparameters)

# Optimizers
policy_optimizer = torch.optim.Adam(policy.parameters(), lr=hyperparameters.lr)
q_optimizer = torch.optim.Adam(q_function.parameters(), lr=hyperparameters.lr)
hard_weight_update(policy_target, policy)
hard_weight_update(q_function_target, q_function)

env = gym.make("Pendulum-v1", render_mode="human")
steps_done = 0

try:
    for episode in range(EPISODES):
        observation, info = env.reset()
        terminated = truncated = False 
        while not (terminated or truncated):
            with torch.no_grad():
                action = get_action_array(policy, hyperparameters, observation, steps_done)
            old_state = observation.reshape((3,))
            observation, reward, terminated, truncated, info = env.step(action)
            replay_memory.push(old_state, action, observation, reward, terminated)
            optimize(replay_memory, hyperparameters, policy, policy_target, q_function, q_function_target, policy_optimizer, q_optimizer)
            soft_weight_update(policy_target, policy, hyperparameters)
            soft_weight_update(q_function_target, q_function, hyperparameters)
            steps_done += 1
except KeyboardInterrupt:
    pass 

tracked_results.cleanup()


