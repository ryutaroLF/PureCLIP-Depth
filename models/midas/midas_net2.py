import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import FeatureFusionBlock, Interpolate, _make_encoder


class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
        """
        print("Loading weights: ", path)

        super(MidasNet, self).__init__()

        use_pretrained = False if path is None else True

        # エンコーダとスケッチ部分を生成
        self.pretrained, self.scratch = _make_encoder(
            backbone="resnext101_wsl",
            features=features,
            use_pretrained=use_pretrained,
        )

        # デコーダの FeatureFusion ブロック
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        # 最終出力用畳み込み
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        if path:
            self.load(path)

    def forward_encoder(self, x):
        """Encoder 部分のみを実行し，中間特徴量を返す.

        Args:
            x (Tensor): 入力画像 tensor (B, C, H, W)

        Returns:
            tuple of Tensors: (layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn)
        """
        # Backbone の各レイヤー
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        # スクラッチ部分でチャネル数を揃える
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        return layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn

    def forward_decoder(self, features):
        """Decoder 部分のみを実行し，最終深度マップを返す.

        Args:
            features (tuple): forward_encoder が返す中間特徴量

        Returns:
            Tensor: 深度マップ (B, H_out, W_out)
        """
        layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn = features

        # FeatureFusionBlock でアップサンプリング＋融合
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # 最終畳み込みで 1 チャンネルに
        out = self.scratch.output_conv(path_1)

        return torch.squeeze(out, dim=1)

    def forward(self, x):
        """Full forward: encoder → decoder."""
        mid_feats = self.forward_encoder(x)
        depth    = self.forward_decoder(mid_feats)
        return depth