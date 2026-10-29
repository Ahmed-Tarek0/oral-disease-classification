class InvalidImageError(Exception):
    """Raised when the uploaded file is not a valid, readable image."""


class UnsupportedFileTypeError(Exception):
    """Raised when the uploaded file's content type is not allowed."""


class FileTooLargeError(Exception):
    """Raised when the uploaded file exceeds the size limit."""


class ModelNotLoadedError(Exception):
    """Raised when a prediction is requested before the model has loaded."""