import os
from typing import List, Tuple

import cv2


def show_image(image, title: str = ""):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_iterator(dir: str, ext: List[str] = ["jpg", "png", "jpeg"]):
    for f in os.listdir(dir):
        if f.lower().endswith(tuple(ext)):
            fname = os.path.join(dir, f)
            image = cv2.imread(fname)
            yield image, fname


def scale_image(image, max_edge):
    while max(image.shape[:2]) > max_edge:
        h, w = image.shape[:2]
        image = cv2.resize(image, (w // 2, h // 2))
    return image


def sliding_window(image, window_size: Tuple[int] = (300, 300), stride: int = 100):
    h, w = image.shape[:2]
    wh, ww = window_size
    for x in range(0, w, stride):
        for y in range(0, h, stride):
            window = image[y : y + wh, x : x + ww, :]
            yield window, (x, y)


def image_pyramid(image, level=3):
    h, w = image.shape[:2]
    out = [image.copy()]
    for i in range(1, level):
        out.append(cv2.resize(image, (w // (2 ** i), h // (2 ** i))))
    return out


def load_template(path: str):
    template = cv2.imread(path)
    template = cv2.resize(template, (220, 220))
    return template
