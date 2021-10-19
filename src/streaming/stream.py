# import the necessary packages
import datetime
from threading import Thread
import cv2
import collections

import numpy as np


class Stream:

    def __init__(self,cam_id:int,src:str, cam_name:str):
        self.stream = cv2.VideoCapture(src)
        self.cam_id = cam_id
        self.cam_name = cam_name
        (self.grabbed, self.frame) = self.stream.read()
        if self.grabbed:
            print(f'{src} Connected :{self.grabbed}')
            self.h, self.w, self.channels = self.frame.shape
            self.stopped = False
        else:
            print(f' Failed Connection {src}')
            self.stopped = True



        self.deque = collections.deque(maxlen=3)


    def start(self):
        print(f' {self.cam_name} {self.cam_id} Threadi basladi ')
        Thread(target=self.update, args=()).start()

        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()
            self.deque.append(self.frame)

    def get_frame_from_deque(self):

        return self.deque[-1]

    def image_show(self,frame:np.ndarray ,cam_id:int):
        cv2.imshow(cam_id, frame)


    def set_frame(self,frame:np.ndarray) -> np.ndarray:
        self.frame = frame

    def cam_state(self) -> bool:
        return self.grabbed

    def get_frame(self) -> np.ndarray:
        return self.frame.copy()

    def get_cam_id(self) -> int:
        return self.cam_id

    def get_width(self) -> int:
        return self.w

    def get_height(self) -> int:
        return self.h

    def stop(self) :
        self.stopped = True

    def get_camName(self):
        return self.cam_name


if __name__ == '__main__':
    stream= Stream(1,"rtsp://admin:Istech2021@192.168.1.189","ic_kapi").start()
    print(stream.get_camName(),stream.get_cam_id())