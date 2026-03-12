"""Tests for data preprocessing."""

import torch
from PIL import Image

from src.data.preprocessing import XRayPreprocessor


class TestPreprocessor:
    def test_transform_image(self):
        preprocessor = XRayPreprocessor(image_size=224)
        # Create a dummy grayscale image (like an X-ray)
        img = Image.new("L", (512, 512), color=128)
        tensor = preprocessor.transform_image(img, is_train=False)
        assert tensor.shape == (3, 224, 224)
        assert isinstance(tensor, torch.Tensor)

    def test_tokenize_report(self):
        preprocessor = XRayPreprocessor(max_length=64)
        tokens = preprocessor.tokenize_report("The heart size is normal.")
        assert tokens["input_ids"].shape == (64,)
        assert tokens["attention_mask"].shape == (64,)

    def test_clean_report(self):
        dirty = "  The   heart is   normal.  \n\n  "
        clean = XRayPreprocessor.clean_report(dirty)
        assert "  " not in clean  # no double spaces
        assert clean == "The heart is normal."

    def test_roundtrip_tokenization(self):
        preprocessor = XRayPreprocessor(max_length=64)
        original = "The lungs are clear. No focal consolidation."
        tokens = preprocessor.tokenize_report(original)
        decoded = preprocessor.decode_tokens(tokens["input_ids"])
        assert original in decoded
