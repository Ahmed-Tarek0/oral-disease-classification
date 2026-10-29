from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    class_name: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: list[ClassProbability]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    detail: str