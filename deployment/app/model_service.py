import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

from app.config import settings
from app.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)


class ModelService:
    _instance: "ModelService | None" = None

    def __init__(self) -> None:
        self.model: tf.keras.Model | None = None
        self.class_names: list[str] = []

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        logger.info("Loading model from %s", settings.model_path)
        self.model = tf.keras.models.load_model(
            settings.model_path,
            custom_objects={"preprocess_input": preprocess_input},
            safe_mode=False,
        )

        with open(settings.class_names_path) as f:
            self.class_names = json.load(f)

        logger.info("Model loaded. Classes: %s", self.class_names)

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, image_array: np.ndarray) -> dict:
        if not self.is_ready():
            raise ModelNotLoadedError("Model is not loaded yet")

        probs = self.model.predict(image_array, verbose=0)[0]
        predicted_idx = int(np.argmax(probs))

        return {
            "predicted_class": self.class_names[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "probabilities": [
                {"class_name": name, "probability": float(p)}
                for name, p in zip(self.class_names, probs)
            ],
        }