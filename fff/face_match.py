import os

from loguru import logger
from tensorflow.keras.models import load_model


class FaceMatch:
    def __init__(self):
        logger.info("Loading facenet model")
        self.facenet_model = load_model(
            os.path.abspath(
                os.path.join(__file__, "../models/keras-facenet/facenet_keras.h5",)
            ),
        )

    def get_embeddings(self, image):
        return self.facenet_model.predict(image)
