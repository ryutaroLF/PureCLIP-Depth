"""
MidasNet-small (MiDaS v3.1) — encoder / decoder 分割版
Adapted from the original MiDaS implementation:
  https://github.com/isl-org/MiDaS
"""

import torch
import torch.nn as nn
from typing import Optional

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
)


class MidasNet_small(BaseModel):
    """
    Lightweight monocular depth-estimation network (MiDaS-small)  
    ・forward_encoder  : RGB → tuple(4) of scratch features  
    ・forward_decoder  : scratch features → depth map  
    ・forward          : encoder → decoder ラッパ
    """

    def __init__(
        self,
        path: Optional[str] = None,
        features: int = 64,
        backbone: str = "efficientnet_lite3",
        non_negative: bool = True,
        exportable: bool = True,
        channels_last: bool = False,
        align_corners: bool = True,
        blocks: dict = {"expand": True},
    ):
        super().__init__()

        # ------------------------------------------------------------------ #
        #               1.  Encoder  (pretrained + scratch)                  #
        # ------------------------------------------------------------------ #
        use_pretrained = False if path else True
        self.channels_last = channels_last
        self.backbone = backbone
        self.blocks = blocks
        self.groups = 1

        # チャンネル数 (expand=True ならピラミッド的に増やす)
        self.expand = bool(blocks.get("expand", False))
        ch1 = features
        ch2 = features * (2 if self.expand else 1)
        ch3 = features * (4 if self.expand else 1)
        ch4 = features * (8 if self.expand else 1)

        # pretrained  : encoder (EfficientNet-Lite-3 等)
        # scratch     : channel projection + refinenet blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            use_pretrained,
            groups=self.groups,
            expand=self.expand,
            exportable=exportable,
        )

        self.scratch.activation = nn.ReLU(inplace=False)

        # Refine blocks (= decoder 内で使用)
        self.scratch.refinenet4 = FeatureFusionBlock_custom(
            ch4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners
        )
        self.scratch.refinenet3 = FeatureFusionBlock_custom(
            ch3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners
        )
        self.scratch.refinenet2 = FeatureFusionBlock_custom(
            ch2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners
        )
        self.scratch.refinenet1 = FeatureFusionBlock_custom(
            ch1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners
        )

        # Output head
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=align_corners),
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(True) if non_negative else nn.Identity(),  # 深度を負にしない
        )

        # 重みロード（ファインチューニング／推論用）
        if path is not None:
            print("Loading MiDaS-small weights from:", path)
            self.load(path)

    # ------------------------------------------------------------------ #
    #  Encoder : RGB → scratch features                                  #
    # ------------------------------------------------------------------ #
    def forward_encoder(self, x: torch.Tensor):
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        # EfficientNetLite3 (例) のステージ出力
        layer1 = self.pretrained.layer1(x)
        layer2 = self.pretrained.layer2(layer1)
        layer3 = self.pretrained.layer3(layer2)
        layer4 = self.pretrained.layer4(layer3)

        # scratch に通してチャンネル数を統一
        layer1_rn = self.scratch.layer1_rn(layer1)
        layer2_rn = self.scratch.layer2_rn(layer2)
        layer3_rn = self.scratch.layer3_rn(layer3)
        layer4_rn = self.scratch.layer4_rn(layer4)

        return (layer1_rn, layer2_rn, layer3_rn, layer4_rn)

    # ------------------------------------------------------------------ #
    #  Decoder : scratch features → depth map                            #
    # ------------------------------------------------------------------ #
    def forward_decoder(self, feats):
        layer1_rn, layer2_rn, layer3_rn, layer4_rn = feats

        path4 = self.scratch.refinenet4(layer4_rn)                        # 最深層
        path3 = self.scratch.refinenet3(path4, layer3_rn)                # + skip3
        path2 = self.scratch.refinenet2(path3, layer2_rn)                # + skip2
        path1 = self.scratch.refinenet1(path2, layer1_rn)                # + skip1

        depth = self.scratch.output_conv(path1)                          # (B,1,H,W)
        return depth.squeeze(1)                                          # → (B,H,W)

    # ------------------------------------------------------------------ #
    #  Wrapper : encoder → decoder                                       #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor):
        feats = self.forward_encoder(x)
        return self.forward_decoder(feats)


# ---------------------------------------------------------------------- #
#   オプション: Conv-BN-ReLU の fusing も元実装そのまま使用可能            #
# ---------------------------------------------------------------------- #
def fuse_model(model: nn.Module):
    """
    Conv2d + BatchNorm2d (+ ReLU) を量子化/高速化のために fuse する。
    """
    prev_prev_type, prev_prev_name = nn.Identity, ""
    prev_type,      prev_name      = nn.Identity, ""
    for name, module in model.named_modules():
        if (
            prev_prev_type is nn.Conv2d
            and prev_type is nn.BatchNorm2d
            and isinstance(module, nn.ReLU)
        ):
            torch.quantization.fuse_modules(model, [prev_prev_name, prev_name, name], inplace=True)
        elif prev_prev_type is nn.Conv2d and prev_type is nn.BatchNorm2d:
            torch.quantization.fuse_modules(model, [prev_prev_name, prev_name], inplace=True)

        prev_prev_type, prev_prev_name = prev_type, prev_name
        prev_type,      prev_name      = type(module), name