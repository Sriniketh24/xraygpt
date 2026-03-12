"""Tests for configuration loading."""

from src.utils.config import XRayGPTConfig, load_config


class TestConfig:
    def test_default_config(self):
        config = XRayGPTConfig()
        assert config.data.image_size == 224
        assert config.vision.model_name == "vit_base_patch16_224"
        assert config.text.model_name == "gpt2"
        assert config.training.batch_size == 16

    def test_load_from_yaml(self):
        config = load_config("configs/base.yaml")
        assert config.data.dataset == "iu_xray"
        assert config.vision.freeze is True
        assert config.training.num_epochs == 30

    def test_default_when_no_file(self):
        config = load_config(None)
        assert isinstance(config, XRayGPTConfig)
