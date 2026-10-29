from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(protected_namespaces=(), env_prefix="ORAL_API_")

    model_path: Path = Path("models/model.keras")
    class_names_path: Path = Path("models/class_names.json")
    image_size: tuple[int, int] = (224, 224)
    max_upload_size_mb: int = 5
    allowed_content_types: set[str] = {"image/jpeg", "image/png", "image/jpg"}


settings = Settings()