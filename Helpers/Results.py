

from pathlib import Path
import json 
import time 
import torch 
import numpy as np 
import subprocess 
import zmq
import struct 
import os 

from Helpers.Hyperparameters import HyperparameterData



class Results:
    def __init__(self, episodes, trial_name=str(time.time()), y_range=400):
        self.process = subprocess.Popen(["python3", "Helpers/ResultsBackend.py", trial_name],
                                        close_fds=True)
        
        self.runs_dir = "Runs/"
        self.run_dir = Path(self.runs_dir + trial_name)
        self.run_dir.mkdir()
        time.sleep(1)
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("ipc:///tmp/zeromq_ipc.sock")
        time.sleep(5)
        expected_episodes = struct.pack('f', episodes)
        self.socket.send_multipart([b'results', b'init', expected_episodes, struct.pack('f', y_range)])
        time.sleep(5)

    def update_rewards(self, new_x : float, new_y : float):
        new_x = struct.pack('f', new_x)
        new_y = struct.pack('f', new_y)
        self.socket.send_multipart([b'results', b'results', new_x, new_y]) 

    def update_decay(self, new_x, new_y):
        new_x = struct.pack('f', new_x)
        new_y = struct.pack('f', new_y)
        self.socket.send_multipart([b'results', b'decay', new_x, new_y])  

    def save_model(self, model, name):
        state_dict = model.state_dict()
        torch.save(state_dict, str(self.run_dir) + "/" + name)

    def save_config(self, data : HyperparameterData):
        file = open(str(self.run_dir) + "/config.cfg", "w")
        saving = data.get_dict()
        json.dump(saving, file)
        file.close()

    def save_result_graph(self):
        self.socket.send_multipart([b'results', b'save_fig'])  

    def update_image(self, image : np.array):
        image_shape_x = str(image.shape[0]).encode()
        image_shape_y = str(image.shape[1]).encode()
        bye_image = image.astype(np.uint8).tobytes()
        self.socket.send_multipart([b'images', b'image', image_shape_x, image_shape_y, bye_image])

    def cleanup(self):
        time.sleep(2)
        self.socket.send_multipart([b'results', b'kill'])
        self.socket.send_multipart([b'images', b'kill'])
        time.sleep(10)
        self.process.kill() 
        self.socket.close()



