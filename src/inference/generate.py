"""
Inference module for generating radiology reports from chest X-ray images.

Supports:
  - Single image inference
  - Batch inference
  - Loading from checkpoint
"""

from pathlib import Path

import torch
from PIL import Image

from src.data.preprocessing import XRayPreprocessor
from src.models.xraygpt import XRayGPT
from src.utils.config import XRayGPTConfig, load_config


class ReportGenerator:
    """
    Inference wrapper for XRayGPT.

    Usage:
        generator = ReportGenerator.from_checkpoint("checkpoints/best_model.pt")
        report = generator.predict("path/to/xray.png")
    """

    def __init__(
        self,
        model: XRayGPT,
        preprocessor: XRayPreprocessor,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocessor = preprocessor

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str | None = None,
        device: torch.device | None = None,
    ) -> "ReportGenerator":
        """
        Load a trained model from a checkpoint file.

        Args:
            checkpoint_path: path to .pt checkpoint
            config_path: optional YAML config override
            device: compute device

        Returns:
            Initialized ReportGenerator
        """
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)

        # Load config from checkpoint or file
        if config_path:
            config = load_config(config_path)
        elif "config" in checkpoint:
            config = checkpoint["config"]
        else:
            config = XRayGPTConfig()

        # Build model and load weights
        model = XRayGPT(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        preprocessor = XRayPreprocessor(
            image_size=config.data.image_size,
            max_length=config.data.max_length,
            tokenizer_name=config.text.model_name,
        )

        return cls(model=model, preprocessor=preprocessor, device=dev)

    def predict(
        self,
        image_path: str | Path,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """
        Generate a radiology report for a single chest X-ray image.

        Args:
            image_path: path to chest X-ray image
            max_new_tokens: override max generation length
            temperature: override sampling temperature
            top_p: override nucleus sampling threshold

        Returns:
            Generated radiology report string
        """
        # Load and preprocess image
        image = self.preprocessor.load_image(str(image_path))
        pixel_values = self.preprocessor.transform_image(image, is_train=False)
        pixel_values = pixel_values.unsqueeze(0).to(self.device)

        # Generate
        reports = self.model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return reports[0]

    def predict_batch(
        self,
        image_paths: list[str | Path],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """
        Generate reports for multiple images.

        Args:
            image_paths: list of paths to chest X-ray images
            max_new_tokens: override max generation length

        Returns:
            List of generated report strings
        """
        images = []
        for path in image_paths:
            image = self.preprocessor.load_image(str(path))
            tensor = self.preprocessor.transform_image(image, is_train=False)
            images.append(tensor)

        pixel_values = torch.stack(images).to(self.device)

        return self.model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
        )
