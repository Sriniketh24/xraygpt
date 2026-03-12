"""
Projection layer that maps visual features into the language model's embedding space.

This is the key learned component of the multimodal architecture. It takes
high-dimensional visual features from the ViT encoder and compresses them into
a fixed number of "visual prefix" tokens that the language model can attend to.

Inspired by:
  - ClipCap (Mokady et al., 2021): prefix-based image captioning
  - LLaVA (Liu et al., 2023): visual instruction tuning with projections
"""

import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    Projects ViT features into LM-compatible prefix tokens.

    Architecture:
        1. Mean-pool patch tokens → single visual embedding
        2. MLP projects to (num_prefix_tokens * lm_hidden_size)
        3. Reshape to (num_prefix_tokens, lm_hidden_size)

    The result is a sequence of "visual prefix" tokens that are prepended
    to the text tokens before being fed to the language model.
    """

    def __init__(
        self,
        vision_hidden_size: int = 768,
        lm_hidden_size: int = 768,
        num_prefix_tokens: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_prefix_tokens = num_prefix_tokens
        self.lm_hidden_size = lm_hidden_size

        # Build MLP
        output_dim = num_prefix_tokens * lm_hidden_size
        layers: list[nn.Module] = []

        in_dim = vision_hidden_size
        hidden_dim = (vision_hidden_size + output_dim) // 2  # interpolated size

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        # Remove dropout and activation from last layer
        layers = layers[:-2]

        self.mlp = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(lm_hidden_size)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Project visual features to prefix tokens.

        Args:
            visual_features: (B, num_patches, vision_hidden_size)
                             Output of the vision encoder

        Returns:
            prefix_tokens: (B, num_prefix_tokens, lm_hidden_size)
                          Visual prefix tokens for the language model
        """
        # Mean pool over patch tokens → (B, vision_hidden_size)
        pooled = visual_features.mean(dim=1)

        # Project → (B, num_prefix_tokens * lm_hidden_size)
        projected = self.mlp(pooled)

        # Reshape → (B, num_prefix_tokens, lm_hidden_size)
        prefix = projected.view(-1, self.num_prefix_tokens, self.lm_hidden_size)

        # Layer norm for training stability
        prefix = self.layer_norm(prefix)

        return prefix
