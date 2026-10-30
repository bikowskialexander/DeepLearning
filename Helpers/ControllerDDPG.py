
# =====================================================
# 
# ControllerDDPG.py
# Actor, critic, and optimizer for DDPG.
# 
# =====================================================

import torch 
import numpy as np
import random

from Helpers.Hyperparameters import HyperparameterData
from Helpers.Memory import Memory
from Helpers.ControllerDQN import get_eps, soft_weight_update, hard_weight_update


class ActorDDPG(torch.nn.Module):
    def __init__(self, hyperparameters : HyperparameterData):
        super().__init__() 
        self.device = hyperparameters.device
        sizes = hyperparameters.neurons.copy()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(

            torch.nn.Linear(sizes[0], sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(sizes[1], sizes[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(sizes[2], sizes[3]),
            torch.nn.Tanh()

        ).to(self.device)
        torch.nn.init.uniform_(self.linear_relu_stack[4].weight, -0.003, 0.003)
        torch.nn.init.uniform_(self.linear_relu_stack[4].bias, -0.003, 0.003)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device).float()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class CriticDDPG(torch.nn.Module):
    def __init__(self, hyperparameters : HyperparameterData):
        super().__init__() 
        self.device = hyperparameters.device
        sizes = hyperparameters.neurons.copy()
        sizes[0] += sizes[-1]
        sizes[-1] = 1
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(

            torch.nn.Linear(sizes[0], sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(sizes[1], sizes[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(sizes[2], sizes[3])

        ).to(self.device)

    def forward(self, x1, x2):
        x1 = self.flatten(x1).to(self.device)
        x2 = self.flatten(x2).to(self.device)
        x = torch.cat((x1, x2), dim=1)
        logits = self.linear_relu_stack(x)
        return logits


# Functions for array
def get_action_array(controller : torch.nn, hyperparameters : HyperparameterData, observation, steps_done=None, noise_range=1.0):
    rng = np.random.default_rng()
    noise_array = np.zeros((hyperparameters.neurons[-1]))
    if steps_done != None:
        noise_array = rng.normal(0, noise_range, size=(noise_array.shape[0]))
    tensor_observation = torch.from_numpy(observation.reshape((1, -1))).float()
    action_tensor = controller(tensor_observation)
    action_array = np.array(action_tensor.to(hyperparameters.cpu).detach().numpy())
    action_array += noise_array
    return action_array

def optimize(sample_memory : Memory, hyperparameters : HyperparameterData, actor : ActorDDPG, actor_target : ActorDDPG, critic : CriticDDPG, critic_target : CriticDDPG, critic_optimizer, actor_optimizer):

    batch_size = hyperparameters.batch_size
    if len(sample_memory.states) < batch_size:
        return 
    device = hyperparameters.device
    gamma = hyperparameters.gamma

    # Extract and Convert Sample
    sample = sample_memory.sample(batch_size)
    old_states = torch.tensor(np.array(sample[0])).to(device).float()
    action = torch.tensor(np.array(sample[1])).to(device).float()
    states = torch.tensor(np.array(sample[2])).to(device).float()
    rewards = torch.tensor(sample[3]).to(device).float().unsqueeze(1)
    dones = torch.tensor(sample[4], dtype=torch.bool).to(device).unsqueeze(1)

    with torch.no_grad():
        # Future rewards prediction
        next_action = actor_target(states)
        critic_q = critic_target(states, next_action)
        critic_q[dones] = 0.0

        # bellman (future and present)
        target_q = rewards + gamma * critic_q

    # Current loss (For sampled moves)
    current_q = critic(old_states, action)
    critic_loss = torch.nn.MSELoss()(current_q.float(), target_q.float())

    # Adjust Critic 
    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_optimizer.step()

    for param in critic.parameters():
        param.requires_grad = False

    predicted_action = actor(old_states)
    actor_loss = -critic(old_states, predicted_action).mean()

    # Adjust Actor
    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optimizer.step()

    for param in critic.parameters():
        param.requires_grad = True


