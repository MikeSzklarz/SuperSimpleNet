"""ViT / vision-foundation-model feature extractors for SuperSimpleNet.

Backbone wrappers adapted from AnomalyVFM (https://github.com/vicoslab/AnomalyVFM,
../AnomalyVFM/models/*.py), made self-contained: no PEFT hooks, no hardcoded
.cuda() calls, HuggingFace imports are lazy so the base install stays untouched.

Each extractor matches the CNN FeatureExtractor contract:
    forward(images [B,3,H,W]) -> [B, C, fh, fw]
    self.feature_dim == (C, fh, fw)
The backbone is always frozen (requires_grad=False + eval() + no_grad()).

Inputs are expected to be ImageNet-normalized (as produced by the datamodules);
backbones with different input statistics are handled by a fused renormalization
inside the extractor.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_log = logging.getLogger("ssn")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# backbone-name prefix -> handled by ViTFeatureExtractor
VIT_BACKBONE_PREFIXES = ("dinov2", "dinov3", "radio", "clip", "siglip2", "tipsv2")

# native / recommended square input sizes per patch size
RECOMMENDED_SIZES = {14: (252, 448, 518), 16: (256, 512)}


def is_vit_backbone(backbone: str) -> bool:
    return backbone.lower().startswith(VIT_BACKBONE_PREFIXES)


def expected_patch_size(backbone: str) -> int:
    """Patch size of a ViT backbone by name, without loading any weights."""
    name = backbone.lower()
    if name.startswith(("dinov2", "clip", "tipsv2")):
        return 14
    return 16  # dinov3, radio, siglip2


def _require_transformers(backbone: str):
    try:
        from transformers import AutoModel  # noqa: F401

        return AutoModel
    except ImportError as e:
        raise RuntimeError(
            f"Backbone '{backbone}' requires the 'transformers' package "
            "(pip install transformers). It is an optional dependency and is "
            "only needed for the clip/siglip2/tipsv2 backbones."
        ) from e


# ---------------------------------------------------------------------------
# Backbone wrappers
# every wrapper: .net (nn.Module), .patch_size (int),
# .mean/.std (input stats the net expects; None -> raw [0,1] images),
# .extract(x, layers) -> list of [B, C, h, w] feature maps
# ---------------------------------------------------------------------------


class _DINOBackbone(nn.Module):
    """DINOv2 (torch.hub facebookresearch/dinov2) and DINOv3 (local hub repo).

    Supports multi-layer extraction via get_intermediate_layers.
    """

    def __init__(self, backbone: str, dinov3_path=None, dinov3_weights=None):
        super().__init__()
        if backbone.startswith("dinov3"):
            if not dinov3_path:
                raise RuntimeError(
                    "DINOv3 weights are gated. Clone https://github.com/facebookresearch/dinov3, "
                    "request the checkpoint from Meta, then pass "
                    "--dinov3-path /path/to/dinov3 [--dinov3-weights /path/to/weights.pth]."
                )
            kwargs = {"source": "local"}
            if dinov3_weights:
                kwargs["weights"] = dinov3_weights
            self.net = torch.hub.load(dinov3_path, backbone, **kwargs)
        else:
            self.net = torch.hub.load(
                "facebookresearch/dinov2", backbone, trust_repo=True
            )
        self.patch_size = self.net.patch_size
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD
        self.num_blocks = len(self.net.blocks)

    def extract(self, x: Tensor, layers: list[int] | None) -> list[Tensor]:
        if layers is None:
            layers = [self.num_blocks - 1]
        return list(
            self.net.get_intermediate_layers(x, n=layers, reshape=True, norm=True)
        )


def _tokens_to_map(tokens: Tensor, h: int, w: int) -> Tensor:
    # [B, N, C] row-major patch tokens -> [B, C, h, w]
    b, n, c = tokens.shape
    assert n == h * w, f"token count {n} does not match grid {h}x{w}"
    return tokens.permute(0, 2, 1).reshape(b, c, h, w)


class _RADIOBackbone(nn.Module):
    """NVIDIA RADIO via torch.hub (NVlabs/RADIO). Expects raw [0,1] input."""

    def __init__(self, backbone: str):
        super().__init__()
        try:
            self.net = torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                version=backbone,
                skip_validation=True,
                trust_repo=True,
            )
        except ImportError as e:
            raise RuntimeError(
                f"Backbone '{backbone}' requires the 'timm' package (pip install timm)."
            ) from e
        self.patch_size = 16
        self.mean = None  # raw [0,1]
        self.std = None

    def extract(self, x: Tensor, layers) -> list[Tensor]:
        _, _, ih, iw = x.shape
        output = self.net(x)
        return [_tokens_to_map(output.features, ih // self.patch_size, iw // self.patch_size)]


class _CLIPBackbone(nn.Module):
    """OpenAI CLIP ViT-L/14-336 vision tower via HuggingFace transformers."""

    HF_NAME = "openai/clip-vit-large-patch14-336"

    def __init__(self, backbone: str):
        super().__init__()
        AutoModel = _require_transformers(backbone)
        self.net = AutoModel.from_pretrained(self.HF_NAME).vision_model
        self.patch_size = 14
        self.mean = CLIP_MEAN
        self.std = CLIP_STD

    def extract(self, x: Tensor, layers) -> list[Tensor]:
        _, _, ih, iw = x.shape
        tokens = self.net(x, interpolate_pos_encoding=True).last_hidden_state
        # drop CLS token
        return [_tokens_to_map(tokens[:, 1:, :], ih // self.patch_size, iw // self.patch_size)]


class _SigLIP2Backbone(nn.Module):
    """SigLIP2 so400m NaFlex vision tower via HuggingFace transformers.

    NaFlex takes pre-patchified input, so the image is re-packed into a
    patch sequence with attention mask + spatial shapes (device-aware, unlike
    the AnomalyVFM original which hardcoded .cuda()).
    """

    HF_NAME = "google/siglip2-so400m-patch16-naflex"

    def __init__(self, backbone: str):
        super().__init__()
        AutoModel = _require_transformers(backbone)
        self.net = AutoModel.from_pretrained(self.HF_NAME).vision_model
        self.patch_size = 16
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def extract(self, x: Tensor, layers) -> list[Tensor]:
        b, _, ih, iw = x.shape
        p = self.patch_size
        nph, npw = ih // p, iw // p
        patches = (
            x.reshape(b, 3, nph, p, npw, p)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(b, nph * npw, p * p * 3)
        )
        inputs = {
            "pixel_values": patches,
            "attention_mask": torch.ones(b, nph * npw, device=x.device),
            "spatial_shapes": torch.tensor([[nph, npw]], device=x.device).repeat(b, 1),
        }
        tokens = self.net(**inputs).last_hidden_state
        return [_tokens_to_map(tokens, nph, npw)]


class _TIPSv2Backbone(nn.Module):
    """Google TIPSv2-L/14 vision encoder via HuggingFace (trust_remote_code)."""

    HF_NAME = "google/tipsv2-l14"

    def __init__(self, backbone: str):
        super().__init__()
        AutoModel = _require_transformers(backbone)
        self.net = AutoModel.from_pretrained(
            self.HF_NAME, trust_remote_code=True
        ).vision_encoder
        self.patch_size = 14
        self.mean = None  # raw [0,1]
        self.std = None

    def extract(self, x: Tensor, layers) -> list[Tensor]:
        _, _, ih, iw = x.shape
        _, _, tokens = self.net(x)
        return [_tokens_to_map(tokens, ih // self.patch_size, iw // self.patch_size)]


def _build_backbone(backbone: str, dinov3_path, dinov3_weights) -> nn.Module:
    name = backbone.lower()
    if name.startswith(("dinov2", "dinov3")):
        return _DINOBackbone(backbone, dinov3_path, dinov3_weights)
    if name.startswith("radio"):
        return _RADIOBackbone(backbone)
    if name.startswith("clip"):
        return _CLIPBackbone(backbone)
    if name.startswith("siglip2"):
        return _SigLIP2Backbone(backbone)
    if name.startswith("tipsv2"):
        return _TIPSv2Backbone(backbone)
    raise ValueError(f"Unknown ViT backbone: {backbone}")


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class ViTFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str,
        layers: list[int] | None,
        patch_size: int,
        image_size: tuple[int, int],
        feat_scale: int = 1,
        dinov3_path=None,
        dinov3_weights=None,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers
        self.feat_scale = feat_scale

        if layers is not None and not backbone.lower().startswith(("dinov2", "dinov3")):
            raise ValueError(
                "--vit-layers (intermediate-layer selection) is only supported for "
                f"dinov2/dinov3 backbones, not '{backbone}' (uses last layer only)."
            )

        self.backbone = _build_backbone(backbone, dinov3_path, dinov3_weights)

        p = self.backbone.patch_size
        bad = [s for s in image_size if s % p != 0]
        if bad:
            recommended = RECOMMENDED_SIZES.get(p, ())
            raise ValueError(
                f"Image size {tuple(image_size)} is not divisible by the patch size "
                f"({p}) of backbone '{backbone}'. Use multiples of {p}, "
                f"e.g. --image-size {' '.join(str(s) for s in recommended)} "
                f"(square) or the nearest multiples "
                f"{tuple(round(s / p) * p for s in image_size)}."
            )

        # frozen backbone: no grads ever (forward additionally runs under no_grad)
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        # fused renormalization: input arrives ImageNet-normalized, convert to the
        # stats this backbone expects:  x' = x * (in_std/bb_std) + (in_mean-bb_mean)/bb_std
        bb_mean = self.backbone.mean if self.backbone.mean is not None else (0.0, 0.0, 0.0)
        bb_std = self.backbone.std if self.backbone.std is not None else (1.0, 1.0, 1.0)
        in_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        in_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        t_mean = torch.tensor(bb_mean).view(1, 3, 1, 1)
        t_std = torch.tensor(bb_std).view(1, 3, 1, 1)
        self.renorm_identity = bool(
            torch.allclose(in_mean, t_mean) and torch.allclose(in_std, t_std)
        )
        self.register_buffer("renorm_scale", in_std / t_std)
        self.register_buffer("renorm_shift", (in_mean - t_mean) / t_std)

        # neighbouring patch aggregation (same convention as the CNN extractor);
        # patch_size 1 disables it
        self.pooler = nn.AvgPool2d(
            kernel_size=patch_size, stride=1, padding=patch_size // 2
        )

        self.feature_dim = self.get_feature_dim(image_size)

    def _renormalize(self, images: Tensor) -> Tensor:
        if self.renorm_identity:
            return images
        return images * self.renorm_scale + self.renorm_shift

    def forward(self, images: Tensor) -> Tensor:
        # extract features (backbone always frozen)
        self.backbone.eval()
        with torch.no_grad():
            feature_maps = self.backbone.extract(
                self._renormalize(images), self.layers
            )

        # channel-wise concat (single map unless multi-layer dinov2/v3)
        feature_map = torch.cat(feature_maps, dim=1)

        if self.feat_scale > 1:
            feature_map = F.interpolate(
                feature_map,
                scale_factor=self.feat_scale,
                mode="bilinear",
                align_corners=True,
            )

        # neighboring patch aggregation
        feature_map = self.pooler(feature_map)

        return feature_map

    def get_feature_dim(self, image_shape: tuple[int, int]) -> tuple[int, int, int]:
        # dryrun (also validates the backbone end-to-end at init)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            feature_map = self.forward(torch.rand(1, 3, *image_shape))
        self.train(was_training)
        _, c, h, w = feature_map.shape
        return c, h, w
