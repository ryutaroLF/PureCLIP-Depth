# -*- coding: utf-8 -*-
"""
Unified NYU/KITTI Dataset (Eigen crop + KITTI 3-tiles, no overlap)

- NYU:
    - RGB -> [0,1] にスケーリング
    - Depth: mm -> m
    - list_file は root_dir/train|test/.../xxx.txt を想定
    - (Eigen crop → リサイズ(height,width) → テンソル)

- KITTI:
    - 入力は /255, /1000 でスケール（RGB: [0,1], Depth: [m]）
    - list_file は root_dir 直下 (train_9.txt / validation_1.txt / test.txt)
    - 実データは root_dir/rgb/<rel_path>, root_dir/depth/<rel_path>
    - (Eigen crop → **バイリニア**で (tile_w*3, tile_h) にリサイズ → 横に3分割)
      - train は 3枚のうちランダム1枚を返す
      - validation/test は 3枚リスト (rgb_list, depth_list) を返す

このファイルは、そのまま実行すれば簡易テストを行います。
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List

# ---------------------------
# KITTI helper: 1行を分解
# ---------------------------
def parse_kitti_line(line: str) -> Tuple[str, str]:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"行の形式が不正です: {line}")
    rgb_rel = os.path.normpath(parts[0])   # 例: 2011_09_26/.../image_02/data/000000.png
    depth_rel = os.path.normpath(parts[1]) # 例: 2011_09_26_drive_xxxx_sync/proj_depth/groundtruth/image_02/000000.png
    return rgb_rel, depth_rel


# ---------------------------
# Dataset
# ---------------------------
class CustomDatasetUnified(Dataset):
    """
    NYU / KITTI を1クラスで扱うデータセット。

    Args:
        root_dir (str): データセットのルートディレクトリ。
        mode (str): 'train', 'validation', 'test'
        height (int): 出力タイル高さ（KITTIでは各タイルの高さ）
        width (int): 出力タイル幅（KITTIでは各タイルの幅）
        dataset_type (str): 'NYU' または 'KITTI'
        eigen_crop (Optional[Tuple[int,int,int,int]]): (top, bottom, left, right)
    """
    def __init__(self,
                 root_dir: str,
                 mode: str,
                 height: int = 336,
                 width: int = 336,
                 dataset_type: str = "NYU",
                 eigen_crop: Optional[Tuple[int, int, int, int]] = None):
        super().__init__()
        assert mode in ("train", "validation", "test")
        assert dataset_type in ("NYU", "KITTI")
        self.root_dir = root_dir
        self.mode = mode
        self.height = height
        self.width = width
        self.dataset_type = dataset_type
        self.eigen_crop = eigen_crop

        # list_file と prefix を決定
        if mode == 'train':
            list_file, self.prefix = 'train_9.txt', 'train/'
        elif mode == 'validation':
            list_file, self.prefix = 'validation_1.txt', 'train/'
        else:
            list_file, self.prefix = 'test.txt', 'test/'

        # list_file の場所
        if self.dataset_type == "KITTI":
            # KITTI は root_dir 直下に txt がある
            txt_path = os.path.join(root_dir, list_file)
        else:
            # NYU は prefix 下に txt がある
            txt_path = os.path.join(root_dir, list_file)

        # サンプル読み込み
        self.samples: List[Tuple[str, str]] = []
        with open(txt_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                if self.dataset_type == "NYU":
                    # NYU: 「rgb depth」の相対パスが書かれている想定
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    rgb_rel = os.path.normpath(parts[0].lstrip('/'))
                    depth_rel = os.path.normpath(parts[1].lstrip('/'))
                    self.samples.append((rgb_rel, depth_rel))
                else:
                    # KITTI: 1行から (rgb_rel, depth_rel) を生成（深度に日付prefixは足さない）
                    rgb_rel, depth_rel = parse_kitti_line(line)
                    self.samples.append((rgb_rel, depth_rel))

    def __len__(self):
        return len(self.samples)

    def _paths_for_index(self, idx: int) -> Tuple[str, str]:
        """ 実ファイルの絶対パスを返す """
        rgb_rel, depth_rel = self.samples[idx]
        if self.dataset_type == "KITTI":
            # KITTI は root_dir/rgb/, root_dir/depth/ 配下に格納
            rgb_path = os.path.join(self.root_dir, "rgb", rgb_rel)
            depth_path = os.path.join(self.root_dir, "depth", depth_rel)
        else:
            # NYU は相対をそのまま連結
            rgb_path = os.path.join(self.root_dir, self.prefix, rgb_rel)
            depth_path = os.path.join(self.root_dir, self.prefix, depth_rel)
        return rgb_path, depth_path

    def _load_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        rgb_path, depth_path = self._paths_for_index(idx)

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise FileNotFoundError(f"ファイルが見つかりません: {rgb_path} または {depth_path}")

        # BGR -> RGB は共通で実施
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # NYU/KITTI ともにスケールを統一: RGB [0,1], Depth [m]
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        rgb = rgb.astype(np.float32) / 255.0
        depth = depth.astype(np.float32) / 1000.0

        return rgb, depth

    def _apply_eigen_crop(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.eigen_crop is None:
            return rgb, depth
        top, bottom, left, right = self.eigen_crop
        return rgb[top:bottom, left:right], depth[top:bottom, left:right]

    def _to_tensors(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # RGBは (H, W, C) -> (C, H, W)
        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        # Depthは (H, W) -> (1, H, W)
        depth_tensor = torch.from_numpy(depth[None, :, :]).float()
        return rgb_tensor, depth_tensor

    # -------- KITTI 用：4分割(Resize→Tile) --------
    def _kitti_resize_and_tile(self, rgb: np.ndarray, depth: np.ndarray
                               ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        KITTI画像を (tile_w*4, tile_h) にリサイズ後、横に4分割してタイルを返す。
        - RGB: バイリニア
        - Depth: マスク付きバイリニア（欠損は保持）
        欠損条件: depth <= th または非有限値
        """
        import cv2
        import numpy as np
        import torch

        tile_h, tile_w = self.height, self.width
        comp_w, comp_h = tile_w * 4, tile_h  # 例: (1152, 384)
        th = getattr(self, "depth_valid_threshold", 1e-3)  # 好みで変更可
        eps = 1e-8

        # --- RGB: 標準の線形補間 ---
        rgb_resized = cv2.resize(rgb, (comp_w, comp_h), interpolation=cv2.INTER_LINEAR)

        # --- Depth: マスク付きバイリニア ---
        depth_f = depth.astype(np.float32, copy=False)
        valid = np.isfinite(depth_f) & (depth_f > th)  # True=有効
        valid_f = valid.astype(np.float32)

        # 有効値のみを線形補間
        num = cv2.resize(depth_f * valid_f, (comp_w, comp_h), interpolation=cv2.INTER_LINEAR)
        den = cv2.resize(valid_f,           (comp_w, comp_h), interpolation=cv2.INTER_LINEAR)

        dep_resized = num / (den + eps)
        # 有効度が十分でない所は欠損へ（0 でも np.nan でもOK。下は 0 に統一）
        dep_resized[den < 0.5] = 0.0

        # --- タイル化 ---
        rgb_tiles: List[torch.Tensor] = []
        dep_tiles: List[torch.Tensor] = []
        for k in range(4):
            l = k * tile_w
            r = l + tile_w
            rgb_k = rgb_resized[:, l:r]              # (H, W, 3)
            dep_k = dep_resized[:, l:r]              # (H, W)

            rt, dt = self._to_tensors(rgb_k, dep_k)  # 既存のテンソル化(正規化/CHW化など)
            rgb_tiles.append(rt)
            dep_tiles.append(dt)

        return rgb_tiles, dep_tiles

    def __getitem__(self, idx):
        rgb, depth = self._load_pair(idx)

        # --- Eigen crop 共通 ---
        rgb, depth = self._apply_eigen_crop(rgb, depth)

        if self.dataset_type == "NYU":
            # RGB: 標準どおりバイリニア
            rgb_resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            # Depth: マスク付きバイリニア（欠損は保持）
            th  = getattr(self, "depth_valid_threshold", 1e-3)  # mm→m済みなので 1e-3[m] が妥当
            eps = 1e-8
            depth_f = depth.astype(np.float32, copy=False)
            valid   = np.isfinite(depth_f) & (depth_f > th)
            valid_f = valid.astype(np.float32)

            num = cv2.resize(depth_f * valid_f, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            den = cv2.resize(valid_f,           (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            dep_resized = num / (den + eps)
            # 欠損復元（0でもnp.nanでもOK。学習/評価の下流に合わせて選択）
            dep_resized[den < 0.5] = 0.0

            rgb_t, dep_t = self._to_tensors(rgb_resized, dep_resized)
            return rgb_t, dep_t

        # --- 以下 KITTI 専用処理（オーバーラップ無し：Resize→3分割） ---
        rgb_tiles, dep_tiles = self._kitti_resize_and_tile(rgb, depth)

        if self.mode == "train":
            # 4つのうちランダム1枚
            k = np.random.randint(0, 4)
            return rgb_tiles[k], dep_tiles[k], k
        
        else:
            # validation / test: 4タイルをリストで返す
            return rgb_tiles, dep_tiles


# =========================================================
# ここから下は「直書きの簡易テスト」コードです（編集して使ってください）
# =========================================================
if __name__ == "__main__":
    # ---- ここを環境に合わせて編集してください ----
    # 例1: KITTI の検証（3タイル方式）
    root_dir      = r"G:\KITTI"
    dataset_type  = "KITTI"                     # "KITTI" or "NYU"
    mode          = "train"                # "train" / "validation" / "test"
    height, width = 384, 384                    # タイル1枚のサイズ（→ composite は (width*3, height)）
    eigen_crop    = (40, 470, 38, 603)           # 必要に応じて（None可）
    sample_index  = 0

    # 例2: NYU の学習（従来どおり）
    # root_dir      = r"G:\datasets\nyu_v2"
    # dataset_type  = "NYU"
    # mode          = "train"
    # height, width = 384, 384
    # eigen_crop    = (45, 471, 41, 601)
    # sample_index  = 0

    # -----------------------------------------------

    ds = CustomDatasetUnified(
        root_dir=root_dir,
        mode=mode,
        height=height,
        width=width,
        dataset_type=dataset_type,
        eigen_crop=eigen_crop
    )

    print(f"[INFO] Samples: {len(ds)} (type={dataset_type}, mode={mode})")

    sample = ds[sample_index]

    if dataset_type == "KITTI" and mode in ("validation", "test"):
        rgb_list, depth_list = sample
        print(f"[INFO] Tiles: {len(rgb_list)} (expect 3)")
        for i in range(len(rgb_list)):
            print(f"  Tile {i}: RGB {tuple(rgb_list[i].shape)}, DEP {tuple(depth_list[i].shape)}")
        # 値域確認（共通化：RGB[0,1], Depth[m]）
        rgb0 = rgb_list[0].numpy()
        dep0 = depth_list[0].numpy()
        print(f"  RGB[0] min/max: {rgb0.min():.3f}/{rgb0.max():.3f}")
        print(f"  DEP[0] min/max: {dep0.min():.3f}/{dep0.max():.3f} (m)")
    else:
        rgb_t, depth_t = sample
        print(f"RGB shape: {tuple(rgb_t.shape)} (dtype={rgb_t.dtype})")
        print(f"DEP shape: {tuple(depth_t.shape)} (dtype={depth_t.dtype})")
        arr_rgb = rgb_t.numpy()
        arr_dep = depth_t.numpy()
        print(f"RGB min/max: {arr_rgb.min():.3f}/{arr_rgb.max():.3f}")
        print(f"DEP min/max: {arr_dep.min():.3f}/{arr_dep.max():.3f} (m)")

    print("[OK] Quick test finished.")