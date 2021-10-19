import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
import matplotlib.pyplot as plt

# deep sort imports

from deep_sort import nn_matching
from deep_sort.tracker_try import Tracker

from src.Util.personTrackInfo import PersonTrackInfo

###################################################################################################################
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))



'TESPİT VE TAKİP SINIFI OLUŞTURMA'
class PersonTrackerDeepSort:

    def __init__(self):
        self.tracker = Tracker()

    def personTracker(self, detections: list,  currentCamIndex: int) -> PersonTrackInfo:

        ids = []
        info = PersonTrackInfo()
        self.tracker.predict(currentCamIndex)
        self.tracker.update(detections, currentCamIndex)

        """ Update tracks """
        for track in self.tracker.get_tracks(currentCamIndex):

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            ids.append(track.track_id)




        info.ids =ids
        info.cam = currentCamIndex
        return info
