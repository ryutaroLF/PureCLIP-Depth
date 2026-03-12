import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_beit,
    forward_swin,
    forward_levit,
    forward_vit,
)
from .backbones.levit import stem_b4_transpose
from timm.models.layers import get_act_layer


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPT(BaseModel):
    """
    Vision-Transformer 系バックボーンを用いた深度推定ネットワーク。
    forward_encoder / forward_decoder に分割。
    """

    def __init__(
        self,
        head,
        features: int = 256,
        backbone: str = "vitb_rn50_384",
        readout: str = "project",
        channels_last: bool = False,
        use_bn: bool = False,
        use_imagenet_pretrained: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.channels_last = channels_last

        # ------------------------------ #
        #   1.  Transformer  backbone   #
        # ------------------------------ #
        hooks = {
            # 4-scale backbones
            "beitl16_512": [5, 11, 17, 23],
            "beitl16_384": [5, 11, 17, 23],
            "beitb16_384": [2, 5, 8, 11],
            "swin2l24_384": [1, 1, 17, 1],
            "swin2b24_384": [1, 1, 17, 1],
            "swin2t16_256": [1, 1, 5, 1],
            "swinl12_384":  [1, 1, 17, 1],
            "next_vit_large_6m": [2, 6, 36, 39],
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
            # 3-scale backbone
            "levit_384": [3, 11, 21],
        }[backbone]

        in_features = None
        if "next_vit" in backbone:
            in_features = {"next_vit_large_6m": [96, 256, 512, 1024]}[backbone]

        self.pretrained, self.scratch = _make_encoder(
            backbone=backbone,
            features=features,
            use_pretrained=use_imagenet_pretrained,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks,
            use_readout=readout,
            in_features=in_features,
        )

        self.number_layers = len(hooks)  # 3 または 4
        size_refinenet3 = None
        self.scratch.stem_transpose = None

        # forward 関数切替
        if "beit" in backbone:
            self.forward_transformer = forward_beit
        elif "swin" in backbone:
            self.forward_transformer = forward_swin
        elif "next_vit" in backbone:
            from .backbones.next_vit import forward_next_vit
            self.forward_transformer = forward_next_vit
        elif "levit" in backbone:
            self.forward_transformer = forward_levit
            size_refinenet3 = 7  # LeViT 固有
            self.scratch.stem_transpose = stem_b4_transpose(256, 128, get_act_layer("hard_swish"))
        else:
            self.forward_transformer = forward_vit

        # ------------------------------ #
        #   2.  Decoder / Refine blocks  #
        # ------------------------------ #
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, size_refinenet3)
        if self.number_layers >= 4:
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    # ------------------------------------------------------------------ #
    #  Encoder: Transformer → Scratch でチャンネル整形した中間特徴を返す   #
    # ------------------------------------------------------------------ #
    def forward_encoder(self, x):
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        layers = self.forward_transformer(self.pretrained, x)

        if self.number_layers == 3:
            layer_1, layer_2, layer_3 = layers
            layer_4_rn = None
        else:  # 4-scale
            layer_1, layer_2, layer_3, layer_4 = layers
            layer_4_rn = self.scratch.layer4_rn(layer_4)

        # scratch でチャンネル数統一
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)

        return (layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn)

    # ------------------------------------------------------------------ #
    #  Decoder: FeatureFusionBlock でアップサンプリングして深度を出力     #
    # ------------------------------------------------------------------ #
    def forward_decoder(self, features):
        layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn = features

        if self.number_layers == 3:
            path_3 = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        else:
            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])

        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if self.scratch.stem_transpose is not None:
            path_1 = self.scratch.stem_transpose(path_1)

        out = self.scratch.output_conv(path_1)
        return out  # (B, 1, H, W)

    # ------------------------------------------------------------------ #
    #  Wrapper: encoder → decoder                                        #
    # ------------------------------------------------------------------ #
    def forward(self, x):
        feats = self.forward_encoder(x)
        depth = self.forward_decoder(feats)
        return depth


class DPTDepthModel(DPT):
    """
    DPT ベース深度モデル（出力を squeeze(1) して (B,H,W) にする）
    """

    def __init__(
        self,
        path: str = None,
        non_negative: bool = True,
        use_imagenet_pretrained: bool = False,  # ← 追加
        **kwargs,
    ):
        features = kwargs.pop("features", 256)
        head_features_1 = kwargs.pop("head_features_1", features)
        head_features_2 = kwargs.pop("head_features_2", 32)

        head = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        super().__init__(
            head=head,
            features=features,
            **kwargs,
            use_imagenet_pretrained=use_imagenet_pretrained,
        )

        if path is not None:
            self.load(path)

    # デフォルト forward で (B, H, W) に squeeze
    def forward(self, x):
        return super().forward(x).squeeze(1)

    # 親クラスの forward_encoder / forward_decoder をそのまま継承