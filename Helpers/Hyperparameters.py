

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

class HyperparameterData:
    def __init__(self, device=CPU):
        self.lr = LR 
        self.gamma = GAMMA 
        self.buffer_size = CAPACITY
        self.eps_start = EPS_START
        self.eps_decay = EPS_DECAY
        self.eps_end = EPS_END
        self.batch_size = BATCH_SIZE
        self.tau = TAU
        self.device = device 
        self.cpu = CPU
        self.episodes_completed = 0
        self.neurons = [8, 512, 4]
