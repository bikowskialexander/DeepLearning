
# =====================================================
# 
# RecordPusher.py
# Records n episodes of the pusher actor in the env.
# 
# =====================================================

import os
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import cv2 as cv

import gymnasium as gym
import torch

from Helpers.ControllerSAC import PolicySAC
from Helpers.Hyperparameters import HyperparameterData

# Episodes in Video
EPISODE_COUNT = 3

# File name
RUN_NAME = "Pusher"

# Set up Env
env = gym.make('Pusher-v5', render_mode="rgb_array")
observation, info = env.reset()

# Load Hyperparameters
hyperparameters = HyperparameterData.load_from_json("Runs/" + RUN_NAME + "/config.cfg")

# Load Model
weights = torch.load("Runs/" + RUN_NAME + "/Final.pt", weights_only=True)
actor = PolicySAC(hyperparameters)
actor.load_state_dict(weights)

# Set Up Video parameters
video_filename = RUN_NAME + '.mp4'
fourcc = cv.VideoWriter_fourcc(*'mp4v') 
fps = 120.0
frame = env.render()
frame_shape = frame.shape

# Set up recorder
video_writer = cv.VideoWriter(video_filename, fourcc, fps, frame_shape[0:2])

# Main Loop
for episode in tqdm(range(EPISODE_COUNT)):
    terminated = truncated = False

    # Episode Loop
    while not (terminated or truncated):
        action, _ = actor(observation)
        action = action.cpu().detach().numpy()
        observation, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        video_writer.write(frame.copy())
        if (terminated or truncated):
            observation, info = env.reset()

# Save Video
video_writer.release()

# Close Env
env.close()


