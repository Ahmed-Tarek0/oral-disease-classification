import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.model_service import ModelService
from app.preprocessing import validate_upload, load_and_preprocess_image
from app.schemas import PredictionResponse, HealthResponse
from app.exceptions import (
    InvalidImageError,
    UnsupportedFileTypeError,
    FileTooLargeError,
    ModelNotLoadedError,
)

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ModelService.get_instance().load()
    yield


app = FastAPI(
    title="Oral Disease Classification API",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("app/static/index.html")


@app.get("/health", response_model=HealthResponse)
def health_check():
    service = ModelService.get_instance()
    return HealthResponse(status="ok", model_loaded=service.is_ready())


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    raw_bytes = await file.read()

    try:
        validate_upload(file.content_type, raw_bytes)
        image_array = load_and_preprocess_image(raw_bytes)
        result = ModelService.get_instance().predict(image_array)
        return PredictionResponse(**result)

    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    

