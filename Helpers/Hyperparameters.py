
from dataclasses import dataclass, field, asdict
import json 
import torch 

# Constants
CPU = "cpu"
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 100000
LR = 0.0001
BATCH_SIZE = 64
GAMMA = 0.99
CAPACITY = 1000000
TAU = 0.005

@dataclass
class HyperparameterData():
    lr : float = LR 
    gamma : float = GAMMA 
    buffer_size : int = CAPACITY
    eps_start : float = EPS_START
    eps_decay : float = EPS_DECAY 
    eps_end : float= EPS_END
    batch_size : int = BATCH_SIZE 
    tau : float = TAU 
    device : torch.device = CPU
    cpu : str = CPU 
    episodes_completed : int = 0 
    neurons : list = field(default_factory=list)

    @classmethod
    def load_from_json(cls, filename : str) :
        with open(filename) as file:
            data = json.load(file)
            file.close()
            return cls(**data)

    def get_dict(self):
        return asdict(self)

    