import os

import attr
import cv2
import numpy as np
from loguru import logger

from utils import *


@attr.s
class FaceDetector:
    method = attr.ib(default="haarcascades", type=str)

    def __attrs_post_init__(self):
        if self.method == "dnn":
            logger.info("Loading face detection model")
            self.net = cv2.dnn.readNetFromCaffe(
                os.path.join(
                    os.path.abspath(__file__),
                    "../models/face_detection_model/deploy.prototxt",
                ),
                os.path.join(
                    os.path.abspath(__file__),
                    "../models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel",
                ),
            )

    def detect(self, image, threshold=0.5):
        if self.method == "haarcascades":
            # https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faceCascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        elif self.method == "dnn":
            # https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300), (0, 0, 0), True, False,
            )
            self.net.setInput(blob)
            preds = self.net.forward()
            face_idx = np.argwhere(preds[0, 0, :, 2] > threshold)
            h, w = image.shape[:2]
            faces = preds[0, 0, face_idx, 3:] * np.array([w, h, w, h])
            faces = faces[:, 0, :].astype(int)
            faces[:, 2:] = faces[:, 2:] - faces[:, :2]

        return faces
