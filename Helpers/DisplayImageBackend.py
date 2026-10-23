

import zmq 
import numpy as np
import cv2 as cv 

class DisplayImageBackend():
    def __init__(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect("ipc:///tmp/zeromq_ipc.sock")
        self.socket.subscribe(b'images')

    def update_image(self, message):
        x_shape = int(message[2].decode())
        y_shape = int(message[3].decode())
        shape = (x_shape, y_shape, 3)
        image = message[4]
        display_image = np.frombuffer(image, dtype=np.uint8).reshape(shape)
        cv.imshow("Display", display_image)
        cv.waitKey(1)

    def cleanup(self):
        self.socket.close()
        exit()

    def parse_message(self, message):
        if (message == None):
            return 
        if (message[1] == b'image'):
            self.update_image(message)
        if (message[1] == b'kill'):
            self.cleanup()

    def check_for_message(self):
        try:
            message = self.socket.recv_multipart(zmq.NOBLOCK)
            return message 
        except zmq.error.Again:
            return None 


def main():
    image_display = DisplayImageBackend()
    while True:
        message = image_display.check_for_message()
        image_display.parse_message(message)

if __name__ == "__main__":
    main()
