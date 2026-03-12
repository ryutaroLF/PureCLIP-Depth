# example_run_dpt_loader_debug.py

"""
Model weight mapping (model_type -> weight file path) and backbones:
"dpt_beit_large_512":         "weights/dpt_beit_large_512.pt"          (BEiT)
"dpt_beit_large_384":         "weights/dpt_beit_large_384.pt"          (BEiT)
"dpt_beit_base_384":          "weights/dpt_beit_base_384.pt"           (BEiT)
"dpt_swin2_large_384":        "weights/dpt_swin2_large_384.pt"         (Swin Transformer V2)
"dpt_swin2_base_384":         "weights/dpt_swin2_base_384.pt"          (Swin Transformer V2)
"dpt_swin2_tiny_256":         "weights/dpt_swin2_tiny_256.pt"          (Swin Transformer V2)
"dpt_swin_large_384":         "weights/dpt_swin_large_384.pt"          (Swin Transformer V1)
"dpt_next_vit_large_384":     "weights/dpt_next_vit_large_384.pt"      (Next-ViT)
"dpt_levit_224":              "weights/dpt_levit_224.pt"               (LeViT)
"dpt_large_384":              "weights/dpt_large_384.pt"               (ViT)
"dpt_hybrid_384":             "weights/dpt_hybrid_384.pt"              (ViT+ResNet Hybrid)
"midas_v21_384":              "weights/midas_v21_384.pt"               (ResNet-50)
"midas_v21_small_256":        "weights/midas_v21_small_256.pt"         (ResNet-50)
"openvino_midas_v21_small_256":"weights/openvino_midas_v21_small_256.xml"(ResNet-50, OpenVINO)
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from midas.model_loader import load_model

def main():
    # --- model selection ---
    model_type = "dpt_swin2_base_384"
    model_path = r"F:/MDE/Patch_Lang_ver1/models/midas/weights/dpt_swin2_base_384.pt"

    model, transform, net_w, net_h = load_model(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model_path=model_path,
        model_type=model_type,
        optimize=False,
        height=None,
        square=False,
    )
    model.eval()

    # --- load and inspect input image ---
    image_path = r"F:/MDE/Patch_Lang_ver1/models/midas/demo.jpg"
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) / 255.0
    print("Loaded image shape:", image_rgb.shape)

    # --- transform (resize/normalize) ---
    processed = transform({"image": image_rgb})["image"]
    print("After transform shape:", processed.shape)

    # --- to tensor ---
    if isinstance(processed, np.ndarray):
        tensor = torch.from_numpy(processed)
    else:
        tensor = processed
    print("Tensor shape before permute:", tensor.shape)

    # permute if needed
    if tensor.ndim == 3 and tensor.shape[0] != 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
    print("Tensor shape after permute:", tensor.shape)

    # add batch dim
    sample = tensor.unsqueeze(0).to(next(model.parameters()).device)
    print("Input batch shape:", sample.shape)

    # --- inference ---
    with torch.no_grad():
        out = model(sample)
    print("Output shape:", out.shape)

    # --- visualize depth map ---
    depth = out.squeeze().cpu().numpy()
    plt.imshow(depth, cmap="inferno")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()