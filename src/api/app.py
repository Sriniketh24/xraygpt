"""
FastAPI backend for XRayGPT.

Provides a REST API for generating radiology reports from chest X-ray images.

Endpoints:
    POST /predict       - Generate report from uploaded image
    GET  /health        - Health check
    GET  /model/info    - Model information

Usage:
    uvicorn src.api.app:create_app --host 0.0.0.0 --port 8000
"""

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.inference.generate import ReportGenerator

# Global generator instance (loaded at startup)
_generator: ReportGenerator | None = None


class PredictionResponse(BaseModel):
    report: str
    processing_time_ms: float


class ModelInfoResponse(BaseModel):
    model_name: str
    total_params: int
    trainable_params: int
    device: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def get_generator() -> ReportGenerator:
    if _generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Set XRAYGPT_CHECKPOINT environment variable.",
        )
    return _generator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global _generator
    import os

    checkpoint_path = os.environ.get("XRAYGPT_CHECKPOINT", "checkpoints/best_model.pt")
    config_path = os.environ.get("XRAYGPT_CONFIG", None)

    if Path(checkpoint_path).exists():
        print(f"Loading model from {checkpoint_path}...")
        _generator = ReportGenerator.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )
        print("Model loaded successfully!")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        print("API will start but /predict will return 503.")

    yield

    # Cleanup
    _generator = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="XRayGPT API",
        description=(
            "Multimodal AI for generating radiology reports from chest X-ray images. "
            "**DISCLAIMER**: This is a research/educational tool and is NOT intended "
            "for clinical diagnosis or medical decision-making."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="ok",
            model_loaded=_generator is not None,
        )

    @app.get("/model/info", response_model=ModelInfoResponse)
    async def model_info():
        gen = get_generator()
        return ModelInfoResponse(
            model_name="XRayGPT",
            total_params=gen.model.get_total_params(),
            trainable_params=gen.model.get_trainable_params(),
            device=str(gen.device),
            disclaimer=(
                "This model is for research/educational purposes only. "
                "It is NOT a medical device and should NOT be used for "
                "clinical diagnosis or treatment decisions."
            ),
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)):
        """
        Generate a radiology report from an uploaded chest X-ray image.

        Accepts PNG, JPEG, or DICOM images.
        """
        gen = get_generator()

        # Validate file type
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected an image.",
            )

        try:
            # Read and process image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read image: {str(e)}",
            )

        # Generate report
        start_time = time.time()

        pixel_values = gen.preprocessor.transform_image(image, is_train=False)
        pixel_values = pixel_values.unsqueeze(0).to(gen.device)
        reports = gen.model.generate(pixel_values)
        report = reports[0]

        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            report=report,
            processing_time_ms=round(processing_time, 1),
        )

    return app


# For direct execution: uvicorn src.api.app:app
app = create_app()
