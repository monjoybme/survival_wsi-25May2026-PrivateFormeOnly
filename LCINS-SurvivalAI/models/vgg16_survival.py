"""
models/vgg16_survival.py
------------------------
Modified VGG-16 network that outputs a single scalar log-hazard risk score
per image tile, as used in the survival analysis pipeline.

Architecture changes relative to standard VGG-16:
  • Classifier head outputs 1 neuron (log-hazard) instead of 1000 (classes).
  • Dropout applied at 50 % to fully-connected layers (configurable).
  • New FC-layer weights initialised with Xavier uniform (variance scaling).
  • Pre-trained ImageNet weights are kept for the convolutional backbone.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Variance-Scaling Initialiser (Xavier Uniform ≈ TF variance_scaling fan_avg)
# ──────────────────────────────────────────────────────────────────────────────

def _variance_scaling_init(module: nn.Module) -> None:
    """Apply Xavier uniform initialisation to Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class VGG16Survival(nn.Module):
    """
    Modified VGG-16 for survival risk prediction.

    Forward pass returns a 1-D tensor of log-hazard scores (one per tile).

    Args:
        dropout_rate: Dropout probability on the FC layers (paper: 0.50).
        pretrained:   Load ImageNet weights for the conv backbone.
        freeze_features: Freeze the conv backbone during training (fine-tuning only).
    """

    def __init__(
        self,
        dropout_rate:     float = 0.50,
        pretrained:       bool  = True,
        freeze_features:  bool  = False,
    ):
        super().__init__()

        # ── Convolutional backbone ───────────────────────────────────────────
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        base    = vgg16(weights=weights)
        if pretrained:
            logger.info("VGG-16: loaded ImageNet pre-trained weights.")
        else:
            logger.info("VGG-16: training from scratch.")

        self.features = base.features          # 5 conv blocks → (B, 512, H/32, W/32)
        self.avgpool  = base.avgpool           # AdaptiveAvgPool2d → (B, 512, 7, 7)

        if freeze_features:
            for p in self.features.parameters():
                p.requires_grad = False
            logger.info("VGG-16: conv backbone frozen.")

        # ── Survival classifier head ─────────────────────────────────────────
        # 512 × 7 × 7 = 25 088 input features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 1),               # log-hazard output (no activation)
        )

        # Apply variance-scaling init to the new head
        self.classifier.apply(_variance_scaling_init)

        # ── Parameter count ──────────────────────────────────────────────────
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"VGG16Survival: {total:,} params total, {trainable:,} trainable."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised tile images.
        Returns:
            risk: (B,) log-hazard scores.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)          # (B, 25088)
        x = self.classifier(x)            # (B, 1)
        return x.squeeze(1)               # (B,)

    # ── Convenience ───────────────────────────────────────────────────────────

    def get_feature_extractor(self) -> nn.Sequential:
        """Return the conv backbone (useful for transfer learning)."""
        return self.features

    def unfreeze_features(self) -> None:
        """Unfreeze the conv backbone for fine-tuning."""
        for p in self.features.parameters():
            p.requires_grad = True
        logger.info("VGG-16: conv backbone unfrozen.")
