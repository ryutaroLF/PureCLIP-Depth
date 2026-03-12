
# split_midas_weights.py

import torch
import os

def main():
    # 元の重みファイル
    src = r"F:\MDE\Patch_Lang_ver1\models\midas\weights\midas_v21_384.pt"
    # 保存先ディレクトリ（なければ作る）
    os.makedirs("weights", exist_ok=True)

    # state_dict をロード
    state = torch.load(src, map_location="cpu")
    # midas_v21_384.pt はおそらく model.state_dict() そのものなので、
    # state が { "pretrained.layer1.weight": ..., "scratch.refinenet4.conv.weight": ..., ... } の辞書

    encoder_sd = {}
    decoder_sd = {}

    for k, v in state.items():
        if k.startswith("pretrained."):
            # エンコーダ部分：キーから "pretrained." を取り除く
            new_key = k[len("pretrained."):]
            encoder_sd[new_key] = v
        elif k.startswith("scratch."):
            # デコーダ部分：キーから "scratch." を取り除く
            new_key = k[len("scratch."):]
            decoder_sd[new_key] = v
        else:
            # もし他のキーがあれば（例：バッファなど）、必要に応じて振り分け
            pass

    # 分割した state_dict を保存
    torch.save(encoder_sd, "weights/midas_v21_encoder.pt")
    torch.save(decoder_sd, "weights/midas_v21_decoder.pt")

    print("Saved:")
    print("  Encoder weights → weights/midas_v21_encoder.pt")
    print("  Decoder weights → weights/midas_v21_decoder.pt")

if __name__ == "__main__":
    main()