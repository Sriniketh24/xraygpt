"""
XRayGPT: Full multimodal model combining vision encoder, projection, and text decoder.

Architecture:
    Image → VisionEncoder → ProjectionLayer → ReportDecoder → Report Text

This module ties all components together and provides a clean interface
for both training (forward with loss) and inference (generate).
"""

import torch
import torch.nn as nn

from src.models.projection import ProjectionLayer
from src.models.report_decoder import ReportDecoder
from src.models.vision_encoder import VisionEncoder
from src.utils.config import XRayGPTConfig


class XRayGPT(nn.Module):
    """
    End-to-end multimodal model for radiology report generation.

    Combines:
        1. VisionEncoder: pretrained ViT for visual feature extraction
        2. ProjectionLayer: MLP mapping visual features to LM embedding space
        3. ReportDecoder: GPT-2 for autoregressive report generation
    """

    def __init__(self, config: XRayGPTConfig):
        super().__init__()

        self.config = config

        # Vision encoder (frozen by default)
        self.vision_encoder = VisionEncoder(
            model_name=config.vision.model_name,
            pretrained=config.vision.pretrained,
            hidden_size=config.vision.hidden_size,
            freeze=config.vision.freeze,
        )

        # Projection layer (trainable)
        self.projection = ProjectionLayer(
            vision_hidden_size=config.vision.hidden_size,
            lm_hidden_size=config.projection.hidden_size,
            num_prefix_tokens=config.vision.num_prefix_tokens,
            num_layers=config.projection.num_layers,
            dropout=config.projection.dropout,
        )

        # Text decoder (fine-tuned)
        self.decoder = ReportDecoder(
            model_name=config.text.model_name,
            freeze_embeddings=config.text.freeze_embeddings,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            pixel_values: (B, 3, H, W) chest X-ray images
            input_ids: (B, seq_len) tokenized report
            attention_mask: (B, seq_len) attention mask

        Returns:
            dict with 'loss' and 'logits'
        """
        # 1. Extract visual features
        visual_features = self.vision_encoder(pixel_values)

        # 2. Project to prefix tokens
        prefix_embeds = self.projection(visual_features)

        # 3. Decode with visual prefix + text
        outputs = self.decoder(
            prefix_embeds=prefix_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        beam_size: int | None = None,
        repetition_penalty: float | None = None,
    ) -> list[str]:
        """
        Generate reports from chest X-ray images.

        Args:
            pixel_values: (B, 3, H, W) chest X-ray images
            max_new_tokens: override config max_new_tokens
            temperature: override config temperature
            top_p: override config top_p
            beam_size: override config beam_size
            repetition_penalty: override config repetition_penalty

        Returns:
            List of generated report strings
        """
        gen_cfg = self.config.generation

        # Extract visual features and project
        visual_features = self.vision_encoder(pixel_values)
        prefix_embeds = self.projection(visual_features)

        # Generate text
        reports = self.decoder.generate(
            prefix_embeds=prefix_embeds,
            max_new_tokens=max_new_tokens or self.config.text.max_new_tokens,
            temperature=temperature or gen_cfg.temperature,
            top_p=top_p or gen_cfg.top_p,
            beam_size=beam_size or gen_cfg.beam_size,
            repetition_penalty=repetition_penalty or gen_cfg.repetition_penalty,
        )

        return reports

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def print_param_summary(self) -> None:
        """Print a summary of trainable vs frozen parameters."""
        total = self.get_total_params()
        trainable = self.get_trainable_params()
        frozen = total - trainable

        print(f"{'Parameter Summary':=^50}")
        print(f"  Total parameters:     {total:>12,}")
        print(f"  Trainable parameters: {trainable:>12,}")
        print(f"  Frozen parameters:    {frozen:>12,}")
        print(f"  Trainable ratio:      {trainable / total:.2%}")
        print("=" * 50)
