# example_run_midas.py

import os
import torch
import numpy as np
import cv2

from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

def main():
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルと重みのロード
    weights_path = r"F:\MDE\Patch_Lang_ver1\models\midas\weights\midas_v21_384.pt"
    model = MidasNet(path=weights_path, features=256, non_negative=True)
    model.to(device)
    model.eval()

    # MiDaS v2.1 用の前処理パイプライン
    transform = Compose([
        Resize(
            384, 384,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])

    # 入力画像の読み込み
    img_bgr = cv2.imread(r"F:\MDE\Patch_Lang_ver1\models\midas\test_img.jpg")
    if img_bgr is None:
        raise FileNotFoundError("test_img.jpg が見つかりません")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 前処理 → NumPy 配列取得
    sample = {"image": img_rgb}
    img_input = transform(sample)["image"]   # -> NumPy ndarray, shape (C, H, W)

    # Torch テンソル化・バッチ次元追加・デバイス転送
    input_tensor = torch.from_numpy(img_input).unsqueeze(0).to(device)  # -> (1, C, H, W)

    # 推論
    with torch.no_grad():
        prediction = model(input_tensor)      # -> torch.Tensor (1, H_out, W_out)
    depth_map = prediction.squeeze().cpu().numpy()  # -> ndarray (H_out, W_out)

    # 正規化して 0–255 のグレースケールに
    d_min, d_max = depth_map.min(), depth_map.max()
    normalized = (depth_map - d_min) / (d_max - d_min)
    depth_uint8 = (normalized * 255).astype(np.uint8)

    # カラーマップ適用（Inferno）
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    # 出力フォルダ作成＆保存
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/output.png", depth_colored)

    print("Depth map saved to output/output.png")

if __name__ == "__main__":
    main()