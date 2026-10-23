
from pathlib import Path
import json 
import matplotlib.pyplot as plt
import time 
import torch 
import numpy as np 

from Hyperparameters import HyperparameterData

class ResultsBackend:
    def __init__(self, episodes, trial_name=str(time.time()), y_range=400):
        plt.ion()
        self.runs_dir = "Runs/"
        self.run_dir = Path(self.runs_dir + trial_name)
        plt.style.use("bmh")
        self.fig, ax_list = plt.subplots(1, 2)
        self.ax, self.ax_eps_decay = ax_list
        self.ax.set_xlim(0, episodes)
        self.ax.set_ylim(-y_range, y_range)
        self.ax.set_title("Reward vs Episodes")
        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True)
        self.reward_xs = []
        self.reward_ys = []
        self.average_reward_xs = []
        self.average_reward_ys = []
        self.decay_xs = [] 
        self.decay_ys = []
        self.ax_eps_decay.set_ylim(0, 1.0)
        self.ax_eps_decay.set_title("Decay vs Steps")
        self.ax_eps_decay.set_xlabel("Steps")
        self.ax_eps_decay.set_ylabel("Decay")
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Scatter for rewards
        self.scatter = self.ax.scatter(self.reward_xs, self.reward_ys, s=5, alpha=0.80, label="Reward")

        # Line for rewards average
        self.line = self.ax.plot(self.average_reward_xs, self.average_reward_ys, color='r', label="Mean")

        # Line for decay
        self.decay_line = self.ax_eps_decay.plot(self.decay_xs, self.decay_ys)
        
        self.ax.legend()

    def update_rewards(self, new_x, new_y):
        self.reward_xs.append(new_x)
        self.reward_ys.append(new_y)
        offsets = np.c_[self.reward_xs, self.reward_ys]
        self.scatter.set_offsets(offsets)
        self.average_reward_xs = []
        self.average_reward_ys = []
        for x in range(50, len(self.reward_xs)):
            self.average_reward_xs.append(x)
            y = np.mean(self.reward_ys[x-50:x])
            self.average_reward_ys.append(y)
        self.line[0].set_data(self.average_reward_xs, self.average_reward_ys)

    def update_decay(self, new_x, new_y):
        self.decay_xs.append(new_x)
        self.decay_ys.append(new_y)
        self.decay_line[0].set_data(self.decay_xs, self.decay_ys)
        self.ax_eps_decay.set_xlim(0, new_x)

    def draw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events() 

    def save_result_graph(self):
        self.fig.savefig(str(self.run_dir) + "/results.png")


def main(argv):
    import zmq 
    import struct 
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("ipc:///tmp/zeromq_ipc.sock")
    socket.subscribe(b'results')
    message = socket.recv_multipart()
    if (message[1] != b'init'):
        print("Graph Failed")
        return 

    try:

        expected_episodes = struct.unpack('f', message[2])[0]
        y_range = struct.unpack('f', message[3])[0]
        results = ResultsBackend(expected_episodes, trial_name=str(argv[1]), y_range=y_range)
        while True:
            results.draw()
            try:
                message = socket.recv_multipart(zmq.NOBLOCK)
                if (message[1] == b'kill'):
                    break 
                if (message[1] == b'results'):
                    new_x = struct.unpack('f', message[2])[0]
                    new_y = struct.unpack('f', message[3])[0]
                    results.update_rewards(new_x, new_y)
                if (message[1] == b'decay'):
                    new_x = struct.unpack('f', message[2])[0]
                    new_y = struct.unpack('f', message[3])[0]
                    results.update_decay(new_x, new_y)
                if (message[1] == b'save_fig'):
                    print("Saving")
                    results.save_result_graph()

            except zmq.error.Again:
                pass 

    except KeyboardInterrupt:
        pass 
    
    socket.close()

if __name__ == "__main__":
    import sys 
    main(sys.argv)