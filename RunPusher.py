
# =====================================================
# 
# RunPusher.py
# Runs the trained pusher agent.
# 
# =====================================================

import gymnasium as gym
import torch

from Helpers.ControllerSAC import get_action_array, hard_weight_update, PolicySAC
from Helpers.Hyperparameters import HyperparameterData

# Run Name 
RUN_NAME = "Pusher"

# Set up Env
env = gym.make('Pusher-v5', render_mode="human")

# Load hyperparameters 
hyperparameters = HyperparameterData.load_from_json("Runs/" + RUN_NAME + "/config.cfg")

# Load Model
weights = torch.load("Runs/" + RUN_NAME + "/Final.pt", weights_only=True)
actor = PolicySAC(hyperparameters)
actor.load_state_dict(weights)

# Main Loop
terminated = truncated = True
while True:
    if (terminated or truncated):
        observation, info = env.reset()
    action, _ = actor(observation)
    action = action.cpu().detach().numpy()
    observation, _, terminated, truncated, _ = env.step(action)

