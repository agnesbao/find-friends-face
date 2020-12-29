import cv2
import numpy as np
from loguru import logger

from face_detection import *
from face_match import *
from utils import *


def extract_face(image, fd, inspect=False):
    image_pyr = image_pyramid(image)
    print(image_pyr[-1].shape)
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
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        show_image(out)
    face_patches = []
    for (x, y, w, h) in faces_all:
        face_patch = cv2.resize(image[y : y + h, x : x + w, :], (220, 200))
        face_patches.append(face_patch)
    return np.array(face_patches)


def main():
    template = load_template(r"C:\Users\Xiaojun\repos\find_friends_face\template.jpg")

    album_dir = r"C:\Users\Xiaojun\Pictures\wedding"

    ct = 0
    fd = FaceDetector(method="dnn")
    fm = FaceMatch()

    template_embedding = fm.get_embeddings(template)
    logger.info(template_embedding.shape)

    for image, fname in image_iterator(album_dir):
        logger.info(f"Processing {fname}")
        image = scale_image(image, 1000)
        faces = extract_face(image, fd)
        print(faces.shape)
        ct += 1
        if ct > 5:
            exit()
