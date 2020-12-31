from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from loguru import logger

from face_detection import *
from face_match import *
from utils import *


def extract_face(image, fd, inspect=False):
    image_pyr = image_pyramid(image)
    faces_all = []
    for lvl in range(len(image_pyr)):
        pyr = image_pyr[lvl]
        for patch, pos in sliding_window(pyr):
            faces = fd.detect(patch)
            if len(faces) > 0:
                faces_all.append((faces + np.array([pos[0], pos[1], 0, 0])) * 2 ** lvl)
    if len(faces_all) > 0:
        faces_all = np.vstack(faces_all)
        faces_all, _ = cv2.groupRectangles(faces_all.tolist(), 1)
    if inspect:
        out = image.copy()
        for (x, y, w, h) in faces_all:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
        show_image(scale_image(out, 1000))
    face_patches = []
    for (x, y, w, h) in faces_all:
        face_patch = cv2.resize(image[y : y + h, x : x + w, :], (160, 160))
        face_patches.append(face_patch)
    return np.array(face_patches)


def main():
    template = load_template(r"C:\Users\Xiaojun\repos\find_friends_face\template.jpg")
    # template = load_template(
    #     "/Users/xiaojun.bao/oss-repos/find-friends-face/template.jpg"
    # )

    album_dir = r"C:\Users\Xiaojun\Pictures\wedding"
    # album_dir = "/Users/xiaojun.bao/oss-repos/find-friends-face/album"

    ct = 0
    fd = FaceDetector(method="dnn")
    fm = FaceMatch()

    template_face = extract_face(template, fd)
    template_embedding = fm.get_embeddings(template_face)

    res = defaultdict(list)
    for image, fname in image_iterator(album_dir):
        logger.info(f"Processing {fname}")
        image = scale_image(image, 1000)
        faces = extract_face(image, fd)
        if len(faces) > 0:
            face_embeddings = fm.get_embeddings(faces)
            match_score = ...
            res["fname"].append(fname)
            res["match_score"].append(match_score.max())
        ct += 1
        if ct % 100 == 0:
            logger.info(f"Finished {ct} files")
    res_df = pd.DataFrame(res)
    res_df.sort_values("match_score", ascending=False).to_csv("result.csv", index=False)


if __name__ == "__main__":
    main()
