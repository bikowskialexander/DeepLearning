
# =====================================================
# 
# RunPusher.py
# Runs the trained pusher agent.
# 
# =====================================================

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import math 

from Helpers.ControllerSAC import get_action_array, hard_weight_update, PolicySAC
from Helpers.Hyperparameters import HyperparameterData

env = gym.make('Pusher-v5', render_mode="human")

hyperparameters = HyperparameterData.load_from_json("Runs/1792738068.020177/config.cfg")

weights = torch.load("Runs/1792738068.020177/Final.pt", weights_only=True)
actor = PolicySAC(hyperparameters)
actor.load_state_dict(weights)

terminated = truncated = True
while True:
    if (terminated or truncated):
        observation, info = env.reset()
    action, _ = actor(observation)
    action = action.detach().numpy()
    observation, _, terminated, truncated, _ = env.step(action)

