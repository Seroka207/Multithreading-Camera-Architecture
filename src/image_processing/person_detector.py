# *******************************************************************************
"KASİYERSİZ AKILLI MARKET PERSON TRACKING CODE"
# *******************************************************************************
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tools.utils as utils

from tensorflow.python.saved_model import tag_constants
from tools.config import cfg
import cv2
import numpy as np
from deep_sort import preprocessing
from deep_sort.detection import Detection
from tools import generate_detections as gdet

from src.Util.personDetectInfo import PersonDetectInfo

###################################################################################################################
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

saved_model_loaded_tiny = tf.saved_model.load("./model_files/files/models/checkpoints/yolov4-tiny-416",
                                              tags=[tag_constants.SERVING])
saved_model_loaded = tf.saved_model.load("./model_files/files/models/checkpoints/yolov4-416",
                                         tags=[tag_constants.SERVING])



class PersonDetector:
    def __init__(self, tiny_state=False):
        model_filename = "model_files/files/models/mars-small128V4.pb"
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        if tiny_state:
            self.infer = saved_model_loaded.signatures['serving_default']
        else:
            self.infer = saved_model_loaded_tiny.signatures['serving_default']
        self.max_output_size_per_class = 50
        self.max_total_size = 50
        self.iou_threshold = 0.45
        self.score_threshold = 0.70
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    def personDetection(self, frame: np.ndarray) -> PersonDetectInfo:

        is_exist_person = False
        info = PersonDetectInfo()
        image_data = cv2.resize(frame, (416, 416))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        batch_data = tf.constant(image_data)
        pred_bbox = self.infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=self.max_output_size_per_class,
            max_total_size=self.max_total_size,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold
        )

        """  convert data to numpy arrays and slice out unused elements """
        ### verileri numpy dizilere dönüştürün ve kullanılmayan öğeleri ayırın ###

        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        """ format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height """
        ### Format Çevirimi  ymin, xmin, ymax, xmax ---> xmin, ymin, genişlik, yükseklikten biçimlendir ###
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        """  read in all class names from config """
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        """ custom allowed classes (uncomment line below to customize tracker for only people) """
        allowed_classes = ['person']

        """ loop through objects and use class index to get class name, allow only classes in allowed_classes list """
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)

        """ delete detections that are not in allowed_classes """
        ### person olmayan tespitleri sil ###

        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        """ encode yolo detections and feed to tracker """
        ### yolo tespitlerini tracker için hazırla ###
        features = self.encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        """  run non-maxima supression """
        ###  maximum olmayanları bastırma  ###
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, 1.0, scores)
        detections = [detections[i] for i in indices]

        if len(names) != 0:
            is_exist_person = True

        info.is_exist_person = is_exist_person
        info.class_name = names
        info.detections = detections

        return info

