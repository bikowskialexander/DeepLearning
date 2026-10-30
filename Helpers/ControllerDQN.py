
import torch 
import random 
import math 
import numpy as np 

from Helpers.Hyperparameters import HyperparameterData
from Helpers.Memory import Memory


class Controller(torch.nn.Module):
    def __init__(self, hyperparameters : HyperparameterData):
        super().__init__()

        sizes = hyperparameters.neurons
        self.device = hyperparameters.device 
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential( 

            torch.nn.Linear(sizes[0], sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(sizes[1], sizes[2])

        ).to(self.device)

    def forward(self, x):
        x = self.flatten(x).to(self.device)
        logits = self.linear_relu_stack(x)
        return logits

# Functions to update Network
def hard_weight_update(target, policy):
    w = policy.state_dict()
    target.load_state_dict(w)

def soft_weight_update(target, policy, hyperparameters : HyperparameterData):
    tau = hyperparameters.tau
    target_state_dict = target.state_dict()
    policy_state_dict = policy.state_dict()
    for key in policy_state_dict:
        target_state_dict[key] = policy_state_dict[key]*tau + target_state_dict[key]*(1-tau)
    target.load_state_dict(target_state_dict)

def get_eps(steps_done, hyperparameters : HyperparameterData):
    start = hyperparameters.eps_start
    end = hyperparameters.eps_end
    decay = hyperparameters.eps_decay
    return end + (start - end) * math.exp(-1.0 * steps_done / decay)

# Functions for discreate output
def get_action_during_training(controller : Controller, observation, steps_done :int, hyperparameters : HyperparameterData):
    random_factor = get_eps(steps_done, hyperparameters)
    if (random.random() <= random_factor):
        return random.randint(0, 3)
    tensor_observation = torch.from_numpy(observation.reshape((1, -1))).float()
    action_tensor = controller.forward(tensor_observation)
    action_array = np.array(action_tensor.to(hyperparameters.cpu).detach().numpy())
    action = np.argmax(action_array)
    return action 

def optimize(sample_memory : Memory, optimizer : torch.optim.AdamW, controller : Controller, target : Controller, hyperparameters : HyperparameterData):

    batch_size = hyperparameters.batch_size
    gamma = hyperparameters.gamma
    device = hyperparameters.device
    if len(sample_memory.states) < batch_size:
        return 

    # Extract and decode samples
    samples = sample_memory.sample(batch_size)
    next_state_samples = np.array(samples[2]).reshape((batch_size, 8))
    state_samples = np.array(samples[0]).reshape((batch_size, 8))
    action_sample = torch.tensor(np.array(samples[1])).to(device).unsqueeze(1)
    dones = torch.tensor(np.array(samples[4]), dtype=torch.bool, device=device)

    # For Target (To get reward)
    with torch.no_grad():
        tensor_observation = torch.from_numpy(next_state_samples.reshape((-1, 8))).float()
        next_state_values = target.forward(tensor_observation).max(1).values
        next_state_values[dones] = 0.0
        expected_state_action_values = (next_state_values * gamma) + torch.tensor(samples[3], device=device)

    # For Policy
    tensor_observation = torch.from_numpy(state_samples.reshape((-1, 8))).float()
    state_values = controller.forward(tensor_observation).gather(1, action_sample).squeeze()

    criterion = torch.nn.MSELoss()
    loss = criterion(state_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

