import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor

from importlib import import_module


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str,
        layers: list[str],
        patch_size: int,
        image_size: tuple[int, int],
    ):
        super().__init__()
        self.layers = layers

        # tro to get model from torchvision
        try:
            models = import_module("torchvision.models")
            backbone_class = getattr(models, backbone)
            model = backbone_class(weights="IMAGENET1K_V1")
        except AttributeError as e:
            print(f"Backbone {backbone} not found in torchvision.models.")
            raise AttributeError from e

        self.feature_extractor = create_feature_extractor(model, return_nodes=layers)
        self.pooler = nn.AvgPool2d(
            kernel_size=patch_size, stride=1, padding=patch_size // 2
        )
        self.feature_dim = self.get_feature_dim(image_size)

    def forward(self, images: Tensor) -> Tensor:
        # extract features
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(images)

        features = list(features.values())

        _, _, h, w = features[0].shape
        feature_map = []
        for layer in features:
            # upscale all to 2x the size of the first (largest)
            resized = F.interpolate(
                layer, size=(h * 2, w * 2), mode="bilinear", align_corners=True
            )
            feature_map.append(resized)
        # channel-wise concat
        feature_map = torch.cat(feature_map, dim=1)

        # neighboring patch aggregation
        feature_map = self.pooler(feature_map)

        return feature_map

    def get_feature_dim(self, image_shape: tuple[int, int]) -> tuple[int, int, int]:
        # dryrun
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(torch.rand(1, 3, *image_shape))
        # sum channels
        channels = sum(feature.shape[1] for feature in features.values())
        _, _, h, w = next(iter(features.values())).shape
        return channels, h * 2, w * 2


def build_feature_extractor(config: dict, image_size: tuple[int, int]) -> nn.Module:
    """Build the CNN or ViT feature extractor based on the backbone name.

    ViT / foundation-model backbones (dinov2*, dinov3*, radio*, clip*, siglip2*,
    tipsv2*) are dispatched by name prefix; anything else is treated as a
    torchvision CNN, keeping all existing configs working unchanged.
    """
    from .vit_feature_extractor import ViTFeatureExtractor, is_vit_backbone

    backbone = config.get("backbone", "wide_resnet50_2")
    if is_vit_backbone(backbone):
        return ViTFeatureExtractor(
            backbone=backbone,
            layers=config.get("vit_layers"),
            patch_size=config.get("patch_size", 3),
            image_size=image_size,
            feat_scale=config.get("feat_scale", 1),
            dinov3_path=config.get("dinov3_path"),
            dinov3_weights=config.get("dinov3_weights"),
        )
    return FeatureExtractor(
        backbone=backbone,
        layers=config.get("layers", ["layer2", "layer3"]),
        patch_size=config.get("patch_size", 3),
        image_size=image_size,
    )
