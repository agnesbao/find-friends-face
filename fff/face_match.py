import attr
import os

from loguru import logger
from tensorflow.keras.models import load_model


@attr.s
class FaceMatch:
    logger.info("Loading facenet model")
    facenet_model = load_model(
        os.path.join(
            os.path.abspath(__file__), "../models/keras-facenet/facenet_keras.h5",
        ),
    )

    def get_embeddings(self, image):
        return self.facenet_model.predict(image)
