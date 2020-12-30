import cv2
import face_recognition
from loguru import logger

from utils import *


album_dir = "/Users/xiaojun.bao/oss-repos/find-friends-face/album"

for image, fname in image_iterator(album_dir):
    logger.info(f"Processing {fname}")
    face_locations = face_recognition.face_locations(image)
    print(len(face_locations))
    out = image.copy()
    for (y1, x2, y2, x1) in face_locations:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
    show_image(scale_image(out, 1000))
