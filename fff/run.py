from collections import defaultdict

import cv2
import face_recognition
import pandas as pd
from loguru import logger

from utils import image_iterator


album_dir = "/Users/xiaojun.bao/oss-repos/find-friends-face/album"
template = cv2.imread("/Users/xiaojun.bao/oss-repos/find-friends-face/template.jpg")
template_encoding = face_recognition.face_encodings(template)[0]

res = defaultdict(list)
ct = 0
for image, fname in image_iterator(album_dir):
    logger.info(f"Processing {fname}")
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) > 0:
        dist = face_recognition.face_distance(face_encodings, template_encoding)
        res["fname"].append(fname)
        res["face_dist"].append(dist.min())
    ct += 1
    if ct % 100 == 0:
        logger.info(f"Finished {ct} files")

res_df = pd.DataFrame(res)
prefix = album_dir.split("/")[-1]
res_df.sort_values("face_dist").to_csv(f"{prefix}_result.csv", index=False)
