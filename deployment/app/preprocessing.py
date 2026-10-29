import io
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.exceptions import InvalidImageError, UnsupportedFileTypeError, FileTooLargeError


def validate_upload(content_type: str, raw_bytes: bytes) -> None:
    if content_type not in settings.allowed_content_types:
        raise UnsupportedFileTypeError(
            f"Unsupported content type: {content_type}. "
            f"Allowed: {sorted(settings.allowed_content_types)}"
        )

    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise FileTooLargeError(
            f"File size {size_mb:.2f}MB exceeds the {settings.max_upload_size_mb}MB limit"
        )


def load_and_preprocess_image(raw_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise InvalidImageError("The uploaded file is not a readable image") from exc

    image = image.resize(settings.image_size)
    array = np.asarray(image, dtype=np.float32)   # kept in 0-255 range: the model's
                                                    # internal Lambda layer applies
                                                    # ResNet50's own preprocess_input
    return np.expand_dims(array, axis=0)