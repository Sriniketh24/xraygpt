"""
Unit tests for XRayGPT model components.

Verifies that each module produces correct output shapes and
the full forward pass works end-to-end.
"""

import pytest
import torch

from src.models.projection import ProjectionLayer
from src.models.report_decoder import ReportDecoder
from src.models.vision_encoder import VisionEncoder
from src.models.xraygpt import XRayGPT
from src.utils.config import XRayGPTConfig


@pytest.fixture
def config():
    """Create a test config with small sizes for fast testing."""
    cfg = XRayGPTConfig()
    cfg.vision.model_name = "vit_tiny_patch16_224"  # smaller for tests
    cfg.vision.hidden_size = 192  # ViT-Tiny hidden size
    cfg.vision.num_prefix_tokens = 4
    cfg.projection.hidden_size = 768  # GPT-2 hidden size
    cfg.text.model_name = "gpt2"
    cfg.data.max_length = 32
    return cfg


class TestVisionEncoder:
    def test_output_shape(self):
        encoder = VisionEncoder(
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            hidden_size=192,
            freeze=True,
        )
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        # ViT-Tiny: 197 tokens (1 CLS + 196 patches), 192 dim
        assert out.shape == (2, 197, 192)

    def test_frozen_params(self):
        encoder = VisionEncoder(
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            freeze=True,
        )
        for param in encoder.encoder.parameters():
            assert not param.requires_grad

    def test_unfreeze(self):
        encoder = VisionEncoder(
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            freeze=True,
        )
        encoder.unfreeze()
        for param in encoder.encoder.parameters():
            assert param.requires_grad


class TestProjectionLayer:
    def test_output_shape(self):
        proj = ProjectionLayer(
            vision_hidden_size=192,
            lm_hidden_size=768,
            num_prefix_tokens=4,
        )
        x = torch.randn(2, 197, 192)
        out = proj(x)
        assert out.shape == (2, 4, 768)


class TestReportDecoder:
    def test_forward_shape(self):
        decoder = ReportDecoder(model_name="gpt2")
        prefix = torch.randn(2, 4, 768)
        input_ids = torch.randint(0, 50257, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)

        out = decoder(prefix, input_ids, attention_mask)
        assert "loss" in out
        assert "logits" in out
        # Logits shape: (B, prefix_len + seq_len, vocab_size)
        assert out["logits"].shape[0] == 2
        assert out["logits"].shape[2] == 50257  # GPT-2 vocab size


class TestXRayGPT:
    def test_forward(self, config):
        model = XRayGPT(config)
        pixel_values = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 50257, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)

        out = model(pixel_values, input_ids, attention_mask)
        assert "loss" in out
        assert out["loss"].requires_grad

    def test_generate(self, config):
        model = XRayGPT(config)
        model.eval()
        pixel_values = torch.randn(1, 3, 224, 224)

        reports = model.generate(pixel_values, max_new_tokens=10)
        assert len(reports) == 1
        assert isinstance(reports[0], str)

    def test_param_count(self, config):
        model = XRayGPT(config)
        total = model.get_total_params()
        trainable = model.get_trainable_params()
        assert total > 0
        assert trainable > 0
        assert trainable <= total
