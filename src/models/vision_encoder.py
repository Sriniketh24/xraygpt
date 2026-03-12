"""
Vision encoder module using a pretrained Vision Transformer (ViT).

Extracts visual features from chest X-ray images using a ViT backbone.
We use timm for access to pretrained ViT models with clean APIs.
The encoder outputs patch-level features that are later projected into
the language model's embedding space.
"""

import timm
import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """
    Vision Transformer encoder for chest X-ray images.

    Takes (B, 3, H, W) images and produces (B, num_patches, hidden_size) features.
    By default, the encoder is frozen to prevent overfitting on small datasets.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        hidden_size: int = 768,
        freeze: bool = True,
    ):
        super().__init__()

        # Load pretrained ViT (without the classification head)
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,         # remove classification head
            global_pool="",        # keep all patch tokens (no pooling)
        )
        self.hidden_size = hidden_size

        # Verify the encoder's output dimension matches expected hidden_size
        # ViT-B/16 outputs 768-dim features
        encoder_dim = self.encoder.num_features
        self.dim_match = encoder_dim == hidden_size
        if not self.dim_match:
            self.dim_proj = nn.Linear(encoder_dim, hidden_size)

        if freeze:
            self._freeze()

    def _freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze encoder parameters for end-to-end fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from images.

        Args:
            pixel_values: (B, 3, H, W) normalized image tensor

        Returns:
            features: (B, num_patches, hidden_size) patch-level features
                      For ViT-B/16 with 224x224 input: (B, 197, 768)
                      197 = 1 CLS token + 196 patch tokens (14x14)
        """
        features = self.encoder.forward_features(pixel_values)

        if not self.dim_match:
            features = self.dim_proj(features)

        return features
