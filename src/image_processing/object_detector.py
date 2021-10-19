import os

from src.Util.objectDetectInfo import ObjectDetectInfo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
import cv2
import numpy as np
import core.ob_utils as ob_utils

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))



object_saved_model_loaded = tf.saved_model.load("./model_files/files/models/checkpoints_market/market_yolov4_best-416",tags=[tag_constants.SERVING])

class ObjectDetector:

    def __init__(self):
        self.object_infer = object_saved_model_loaded.signatures['serving_default']

        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    ####################################################################################################################
    def objectDetection(self, frame: np.ndarray,)-> ObjectDetectInfo:

        info = ObjectDetectInfo()
        input_size = 416
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        # start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = self.object_infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.25
        )
        # classes = read_class_names(cfg.YOLO.CLASSES2)
        # fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        # image=ob_utils.draw_bbox(frame, pred_bbox)
        """  read in all class names from config """
        class_names = utils.read_class_names(cfg.YOLO.CLASSES2)
        satin_alinan_urun = ob_utils.nesne_Adi(frame, pred_bbox)
        # print(satin_alinan_urun)

        info.object_name = satin_alinan_urun
        info.pred_box = pred_bbox

        return info






