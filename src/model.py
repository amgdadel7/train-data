"""بناء النموذج: شبكة مدربة مسبقًا (transfer learning) مع طبقة تصنيف جديدة."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHS = ("efficientnet_b0", "resnet50", "mobilenet_v3_large")


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_large(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"architecture غير مدعومة: {arch}. المدعومة: {SUPPORTED_ARCHS}")
    return m
