# util_eval.py
import torch
import torch.nn.functional as F
from utils.final_model import extract_vit_patches  # ここはあなたの既存構成に合わせる
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Metrics (NYU/Eigen-style) --------------------
# I only use GT > min_d(1e-3) 
def compute_errors_eigen_style(gt: torch.Tensor,
                               pred: torch.Tensor,
                               min_d: float,
                               max_d: float):
    names = ['abs_diff','a1','a2','a3','abs_rel','log10','rmse_tot']
    sums = {k: 0.0 for k in names}
    Btot = 0
    for sparse_gt, pred_map in zip(gt, pred):
        s = sparse_gt[0]; p = pred_map[0]
        valid = (s < max_d) & (s > min_d) & (p > min_d)
        if not torch.any(valid):
            continue
        vg = s[valid].clamp(min_d, max_d)
        vp = p[valid].clamp(min_d, max_d)
        thresh = torch.max(vg / vp, vp / vg)
        sums['a1']      += (thresh < 1.25      ).float().mean().item()
        sums['a2']      += (thresh < 1.25**2   ).float().mean().item()
        sums['a3']      += (thresh < 1.25**3   ).float().mean().item()
        sums['rmse_tot']+= torch.sqrt(((vg - vp) ** 2).mean()).item()
        sums['abs_diff']+= torch.abs(vg - vp).mean().item()
        sums['abs_rel'] += (torch.abs(vg - vp) / vg).mean().item()
        sums['log10']   += torch.abs(torch.log10(vg) - torch.log10(vp)).mean().item()
        Btot += 1
    Btot = max(Btot, 1)
    return {k: v / Btot for k, v in sums.items()}

def _compute_viz_limits_from_lists(dep_list, pred_list, min_depth, max_depth, mode="both", percentile=99.5):
    """
    dep_list, pred_list: list of 2D numpy arrays (同じスケールの値)
    mode: "gt" | "pred" | "both"
    """
    import numpy as np
    def pctl(arrs):
        if len(arrs) == 0:
            return min_depth
        flat = np.concatenate([a.ravel() for a in arrs])
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return min_depth
        return float(np.percentile(flat, percentile))
    if mode == "gt":
        vmax = pctl(dep_list)
    elif mode == "pred":
        vmax = pctl(pred_list)
    else:  # both
        vmax = max(pctl(dep_list), pctl(pred_list))
    vmax = max(vmax, min_depth)          # 逆転防止
    vmax = min(vmax, max_depth)          # データセット上限は尊重
    return min_depth, vmax


def _compute_viz_limits_from_arrays(dep_np, pred_np, min_depth, max_depth, mode="both", percentile=99.5):
    """
    dep_np, pred_np: それぞれ (B,H,W) の numpy
    """
    dep_list = [dep_np[b] for b in range(dep_np.shape[0])]
    pred_list = [pred_np[b] for b in range(pred_np.shape[0])]
    return _compute_viz_limits_from_lists(dep_list, pred_list, min_depth, max_depth, mode, percentile)

def plot_dual_heatmap(
    gt_vals,
    pred_vals,
    title_suffix="",
    num_bins=10,
    bin_min=1,
    bin_max=10,
    annotate_counts=False,
    save_path: str = None,
    show: bool = True,
    return_fig_ax: bool = False,
):
    """
    2Dヒストから
      1) 生カウント（GT:横軸, Pred:縦軸）
      2) 行正規化（GTごとのPred分布）
    を並べて描画するユーティリティ。

    Args:
        gt_vals, pred_vals: 1D配列（numpy または torch）。NaN/Infは自動で除去。
        title_suffix      : タイトルに付ける文字列（例: "(No-TTA)").
        num_bins          : ビン数（モデルのbin数と合わせると見やすい）
        bin_min, bin_max  : ヒストの最小/最大（評価レンジ）
        annotate_counts   : 左パネルにカウント数字を表示するか（多いと読みにくくなる）
        save_path         : 画像を保存するパス（Noneなら保存しない）
        show              : plt.show() を呼ぶか
        return_fig_ax     : (fig, (ax_raw, ax_norm)) を返すか

    Returns:
        None もしくは (fig, (ax_raw, ax_norm))
    """
    # ---- 入力をnumpy 1Dに整形・フィルタ ----
    if torch.is_tensor(gt_vals):
        gt_vals = gt_vals.detach().cpu().numpy()
    if torch.is_tensor(pred_vals):
        pred_vals = pred_vals.detach().cpu().numpy()
    gt_vals = np.asarray(gt_vals).ravel()
    pred_vals = np.asarray(pred_vals).ravel()

    # 有効レンジ＆有限値でフィルタ
    finite = np.isfinite(gt_vals) & np.isfinite(pred_vals)
    inrange = (gt_vals >= bin_min) & (gt_vals <= bin_max) & (pred_vals >= bin_min) & (pred_vals <= bin_max)
    mask = finite & inrange
    gt_vals = gt_vals[mask]
    pred_vals = pred_vals[mask]

    # ---- 2Dヒスト ----
    edges = np.linspace(bin_min, bin_max, num_bins + 1)
    Hmat, xedges, yedges = np.histogram2d(gt_vals, pred_vals, bins=[edges, edges])  # H[gt_bin, pred_bin]
    centers = edges[:-1] + (edges[1] - edges[0]) / 2.0

    # 行正規化（GTごとに）
    Hrownorm = Hmat / (Hmat.sum(axis=1, keepdims=True) + 1e-6)

    # ---- 描画 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax_raw, ax_norm = axes

    # 1) 生カウント（GT:横, Pred:縦）→ 可視化は転置して (pred, gt)
    sns.heatmap(
        Hmat.T,
        annot=annotate_counts,
        fmt='d' if annotate_counts else '',
        cmap='YlGnBu',
        xticklabels=np.round(centers, 2),
        yticklabels=np.round(centers, 2),
        ax=ax_raw
    )
    ax_raw.set_title(f'Raw Count {title_suffix}')
    ax_raw.set_xlabel('Ground Truth (m)')
    ax_raw.set_ylabel('Predicted (m)')

    # 2) 行正規化（確率）
    sns.heatmap(
        Hrownorm.T,
        cmap='YlGnBu',
        vmin=0, vmax=1,
        xticklabels=np.round(centers, 2),
        yticklabels=np.round(centers, 2),
        cbar_kws={'label': 'Probability'},
        ax=ax_norm
    )
    ax_norm.set_title(f'Normalized per-GT {title_suffix}')
    ax_norm.set_xlabel('Ground Truth (m)')
    ax_norm.set_ylabel('')

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig_ax:
        return fig, (ax_raw, ax_norm)

def compute_errors_eigen_style_kitti(gt: torch.Tensor,
                                     pred: torch.Tensor,
                                     min_d: float,
                                     max_d: float,
                                     garg_crop_mask: torch.Tensor = None):
    """
    KITTI評価用:
      - 有効画素: garg_crop_mask==1 (あれば) AND gt>0 AND [min_d,max_d] AND 有限
      - その上で Eigen-style 指標を pixel-wise に計算

    Args:
        gt   : (B,1,H,W)  ground-truth depth (meters)
        pred : (B,1,H,W)  predicted depth (meters)
        garg_crop_mask: (B,1,H,W) or (1,1,H,W) or (H,W)  0/1(又はbool)
    """
    assert gt.shape == pred.shape, f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}"
    B, C, H, W = gt.shape
    device = gt.device

    
    s = gt.float() # GTは生で
    p = pred.float()

    # 有効画素の初期マスク（KITTIはLiDAR欠損が多いので gt>0 を明示）
    valid = (gt > min_d) & torch.isfinite(s) & torch.isfinite(p) & (s > min_d) & (s <= max_d) & (p > min_d) & (p <= max_d)

    # garg crop マスクがあれば AND
    if garg_crop_mask is not None:
        m = garg_crop_mask
        # 形状を (B,1,H,W) にブロードキャスト可能に整える
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)        # (1,1,H,W)
        elif m.dim() == 3:
            m = m.unsqueeze(1)                      # (B,1,H,W)
        m = m.to(device)
        if m.dtype != torch.bool:
            m = (m > 0.5)
        # 形が合わない場合は明示的にエラー（静かに失敗しないように）
        if m.shape[-2:] != (H, W):
            raise ValueError(f"garg_crop_mask spatial size {m.shape[-2:]} != gt/pred {(H,W)}")
        if m.shape[0] not in (1, B):
            raise ValueError(f"garg_crop_mask batch size {m.shape[0]} incompatible with B={B}")
        valid = valid & m

    names = ['abs_diff','a1','a2','a3','abs_rel','log10','rmse_tot']
    sums  = {k: 0.0 for k in names}
    Btot  = 0

    for bi in range(B):
        si = s[bi, 0]
        pi = p[bi, 0]
        vi = valid[bi, 0]
        if not torch.any(vi):
            continue

        vg = si[vi].clamp(min_d, max_d)
        vp = pi[vi].clamp(min_d, max_d)

        thresh = torch.maximum(vg / vp, vp / vg)
        sums['a1']      += (thresh < 1.25      ).float().mean().item()
        sums['a2']      += (thresh < 1.25**2   ).float().mean().item()
        sums['a3']      += (thresh < 1.25**3   ).float().mean().item()
        sums['rmse_tot']+= torch.sqrt(((vg - vp) ** 2).mean()).item()
        sums['abs_diff']+= torch.abs(vg - vp).mean().item()
        sums['abs_rel'] += (torch.abs(vg - vp) / vg).mean().item()
        sums['log10']   += torch.abs(torch.log10(vg) - torch.log10(vp)).mean().item()
        Btot += 1

    Btot = max(Btot, 1)
    return {k: v / Btot for k, v in sums.items()}



@torch.no_grad()
def predict_depth_maps(model, rgb: torch.Tensor, tau: float, input_size: int):
    """
    rgb: (B,3,H,W) -> depth_pred: (B,1,H,W)  (nearest upsampling)
    model 側で以下の属性/メソッドを想定:
      - model.clip.visual / model.clip.encode_image
      - model._prep_rgb
      - model.rgb_adapter, model.fusion_type, model.fusion_mlp or model.film_gen
      - model.d, model.P, model.patch
      - model.depth_table.weight, model.bin_centers, model.num_bins
    """
    model.eval()
    B, _, H, W = rgb.shape
    assert H == input_size and W == input_size, "Input size mismatch with config."
    patches, _ = extract_vit_patches(model.clip.visual, model._prep_rgb(rgb))
    z_rgb = model.rgb_adapter(patches.view(-1, model.d)).view(B, model.P, model.d)

    cls_feat = F.normalize(model.clip.encode_image(model._prep_rgb(rgb)).float(), dim=-1)

    if model.fusion_type == 'concat':
        cls_exp = cls_feat.unsqueeze(1).expand(-1, model.P, -1)
        fused = model.fusion_mlp(torch.cat([z_rgb, cls_exp], dim=-1).view(-1, 2 * model.d)).view(B, model.P, model.d)
    else:
        gamma, beta = model.film_gen(cls_feat).chunk(2, dim=-1)
        fused = gamma.unsqueeze(1) * z_rgb + beta.unsqueeze(1)

    z = F.normalize(fused, dim=-1)
    W_norm = F.normalize(model.depth_table.weight, dim=-1)  # (C,d)
    logits = torch.matmul(z, W_norm.T) / tau
    probs  = F.softmax(logits, dim=-1)
    centers = model.bin_centers.view(1, 1, model.num_bins)
    pred_patch = (probs * centers).sum(dim=-1)  # (B,P)

    ph = pw = input_size // model.patch
    depth_pred = F.interpolate(pred_patch.view(B, 1, ph, pw), size=(input_size, input_size), mode='nearest')
    return depth_pred


# ===== helper: valid masks =====
def _valid_mask_nyu(depth, pred, min_d, max_d):
    s = depth.float()
    p = pred.float()
    return (s < max_d) & (s > min_d) & (p > min_d) & torch.isfinite(s) & torch.isfinite(p)

def _valid_mask_kitti(depth, pred, min_d, max_d, garg_crop_mask, device):
    s = depth.float()
    p = pred.float()
    valid = (depth > 0) & torch.isfinite(s) & torch.isfinite(p) \
            & (s > min_d) & (s <= max_d) & (p > min_d) & (p <= max_d)
    if garg_crop_mask is not None:
        m = garg_crop_mask
        if m.dim() == 2: m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3: m = m.unsqueeze(1)
        m = m.to(device)
        if m.dtype != torch.bool: m = (m > 0.5)
        valid = valid & m
    return valid

# ===== helper: debug viz for valid masks =====
### [ADD] save gray mask (0/1) as png
def _save_bool_mask_png(mask_t: torch.Tensor, out_path):
    """
    mask_t : (H,W) bool/0-1 tensor on CPU
    """
    import numpy as np, matplotlib.pyplot as plt
    m = mask_t.detach().cpu().numpy().astype(np.float32)
    plt.imsave(out_path, m, cmap="gray", vmin=0.0, vmax=1.0)

### [ADD] overlay mask on rgb (invalid=red)
def _save_overlay_invalid_on_rgb(rgb_t: torch.Tensor, valid_t: torch.Tensor, out_path, invalid_color=(1.0,0.0,0.0), alpha=0.6):
    """
    rgb_t   : (3,H,W) in [0,1] or [0,255]
    valid_t : (H,W) bool/0-1
    invalid : ~valid を半透明で重ねる
    """
    import numpy as np, matplotlib.pyplot as plt
    rgb = rgb_t.detach().cpu()
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rgb = rgb.clamp(0,1).permute(1,2,0).numpy()  # (H,W,3)
    valid = valid_t.detach().cpu().numpy().astype(bool)
    inv = ~valid
    overlay = rgb.copy()
    col = np.array(invalid_color, dtype=np.float32).reshape(1,1,3)
    overlay[inv] = (1-alpha)*overlay[inv] + alpha*col
    plt.imsave(out_path, overlay)
# ===== helper: metrics accumulator =====
def _accumulate(sums, res, B):
    for k in sums.keys():
        sums[k] += float(res[k]) * B

# ===== helper: collect heatmap values =====
def _collect_heatmap(all_gt_vals, all_pred_vals, depth, pred, valid, min_d, max_d):
    s = depth.float().clamp(min_d, max_d)
    p = pred.float().clamp(min_d, max_d)
    B = s.size(0)
    for bi in range(B):
        vi = valid[bi, 0]
        if torch.any(vi):
            all_gt_vals.append(s[bi, 0][vi].detach().cpu().view(-1))
            all_pred_vals.append(p[bi, 0][vi].detach().cpu().view(-1))

# ===== helper: visualization (keep original fixed range) =====
def _viz_kitti_tiles_save(rgbs_np, deps_np, preds_np, out_path, min_depth, max_depth):
    import numpy as np, matplotlib.pyplot as plt
    rgb_cat  = np.concatenate(rgbs_np, axis=1)
    dep_cat  = np.concatenate(deps_np, axis=1)
    pred_cat = np.concatenate(preds_np, axis=1)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(rgb_cat); axs[0].set_title("RGB"); axs[0].axis("off")
    axs[1].imshow(dep_cat, cmap="viridis", vmin=min_depth, vmax=max_depth)
    axs[1].set_title("Depth"); axs[1].axis("off")
    axs[2].imshow(pred_cat, cmap="viridis", vmin=min_depth, vmax=max_depth)
    axs[2].set_title("Prediction"); axs[2].axis("off")
    plt.tight_layout(); plt.savefig(out_path.as_posix(), dpi=150); plt.close(fig)

def _viz_nyu_batch_save(rgb_np, dep_np, pred_np, out_dir, save_idx, min_depth, max_depth):
    import matplotlib.pyplot as plt
    for b in range(rgb_np.shape[0]):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(rgb_np[b]); axs[0].set_title("RGB"); axs[0].axis("off")
        axs[1].imshow(dep_np[b], cmap="viridis", vmin=min_depth, vmax=max_depth)
        axs[1].set_title("Depth"); axs[1].axis("off")
        axs[2].imshow(pred_np[b], cmap="viridis", vmin=min_depth, vmax=max_depth)
        axs[2].set_title("Prediction"); axs[2].axis("off")
        out_path = out_dir / f"sample_{save_idx:06d}.png"
        plt.tight_layout(); plt.savefig(out_path.as_posix(), dpi=150); plt.close(fig)
        save_idx += 1
    return save_idx



@torch.no_grad()
def predict_depth_maps_tta(model, rgb: torch.Tensor, tau: float, input_size: int):
    """
    水平反転TTA: (素の予測 + 反転→推論→元に戻す) / 2
    入出力のshape/レンジは predict_depth_maps と同じ (B,1,H,W)。
    """
    pred_no = predict_depth_maps(model, rgb, tau=tau, input_size=input_size)
    rgb_flip = torch.flip(rgb, dims=[-1])                 # 水平反転
    pred_flip = predict_depth_maps(model, rgb_flip, tau=tau, input_size=input_size)
    pred_flip = torch.flip(pred_flip, dims=[-1])          # 元に戻す
    return 0.5 * (pred_no + pred_flip)

# ====== main ======
@torch.no_grad()
def eval_metrics(model, loader, device, min_depth, max_depth, tau, input_size,
                 plot_prediction: bool = False, save_dir=None, dataset_type: str = 'nyu',
                 garg_crop_mask=None,
                 plot_dual_heatmap_flag: bool = False,
                 heatmap_num_bins: int = None,
                 use_TTA: bool = False,
                 debug_valid: bool = False,
                 save_valid_masks: bool = False):
    """
    Evaluate metrics over a dataloader.
    Optionally save prediction plots and/or draw dual heatmap of GT vs Pred.
    """
    import numpy as np
    from pathlib import Path

    # --- I/O setup ---
    if plot_prediction or plot_dual_heatmap_flag:
        if save_dir is None:
            raise ValueError("plot_prediction/plot_dual_heatmap_flag=True requires `save_dir` to be specified.")


    save_idx = 0

    if plot_prediction:
        out_dir = Path(save_dir) / "all_predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_idx = 0

    if plot_dual_heatmap_flag:
        heatmap_dir = Path(save_dir) / "heat_map"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        all_gt_vals, all_pred_vals = [], []
        if heatmap_num_bins is None:
            heatmap_num_bins = getattr(model, 'num_bins', 15)
    
    if (debug_valid or save_valid_masks) and save_dir is not None:
        debug_dir = Path(save_dir) / "debug_valid"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
    # --- metric setup ---
    model.eval()
    names = ['abs_diff', 'a1', 'a2', 'a3', 'abs_rel', 'log10', 'rmse_tot']
    sums  = {k: 0.0 for k in names}
    nimg  = 0
    is_kitti = (dataset_type.lower() == 'kitti')

    # --- iterate loader ---
    for batch in loader:
        # ========== KITTI: list-of-tiles ==========
        if isinstance(batch[0], list):
            rgb_list, depth_list = batch[0], batch[1]

            # gather for visualization
            rgbs_np, deps_np, preds_np = [], [], []

            for tile_idx, (rgb_t, depth_t) in enumerate(zip(rgb_list, depth_list)):
                # shape normalize
                if rgb_t.dim() == 3:
                    rgb   = rgb_t.unsqueeze(0).to(device)
                    depth = depth_t.unsqueeze(0).to(device)
                elif rgb_t.dim() == 4:
                    rgb   = rgb_t.to(device)
                    depth = depth_t.to(device)
                else:
                    raise ValueError(f"Unexpected rgb_t dim: {rgb_t.dim()}")

                # inference
                #depth = depth.clamp(min_depth, max_depth)
                # GT depthのclampはしない

                pred  = (predict_depth_maps_tta(model, rgb, tau=tau, input_size=input_size)
                         if use_TTA else
                         predict_depth_maps(model, rgb, tau=tau, input_size=input_size))

                # --- Garg マスクをタイルごとに選ぶ ---
                m_tile = None
                if garg_crop_mask is not None:
                    if isinstance(garg_crop_mask, (list, tuple)):
                        m_tile = garg_crop_mask[tile_idx]  # このタイル用の (H,W) マスク
                    else:
                        m_tile = garg_crop_mask           # 単一Tensorの互換ケース

                # --- debug/valid mask ---
                if is_kitti and (debug_valid or save_valid_masks):
                    s = depth.float()
                    p = pred.float()
                    base_valid = (depth > 0) & torch.isfinite(s) & torch.isfinite(p) \
                                 & (s >= min_depth) & (s <= max_depth) & (p >= min_depth) & (p <= max_depth)

                    if m_tile is not None:
                        m = m_tile
                        if m.dim() == 2: m = m.unsqueeze(0).unsqueeze(0)
                        elif m.dim() == 3: m = m.unsqueeze(1)
                        m = m.to(device)
                        if m.dtype != torch.bool: m = (m > 0.5)
                        if m.shape[-2:] != depth.shape[-2:]:
                            raise ValueError(f"Garg tile mask size {m.shape[-2:]} != tile {depth.shape[-2:]}")
                        combined_valid = base_valid & m
                        garg_m = m
                    else:
                        combined_valid = base_valid
                        garg_m = None

                    vi0 = combined_valid[0,0]
                    tot0 = vi0.numel()
                    vcount0 = int(vi0.sum().item())
                    ratio0 = (vcount0 / tot0) if tot0 > 0 else 0.0

                    if debug_valid:
                        print(f"[KITTI][tile={tile_idx}] valid={vcount0}/{tot0} ({ratio0*100:.2f}%)")
                        bcount0 = int(base_valid[0,0].sum().item())
                        bratio0 = (bcount0 / tot0) if tot0 > 0 else 0.0
                        print(f"[KITTI][tile={tile_idx}] base={bcount0}/{tot0} ({bratio0*100:.2f}%)")
                        if garg_m is not None:
                            gmcount0 = int(garg_m[0,0].sum().item())
                            gmr0 = (gmcount0 / tot0) if tot0 > 0 else 0.0
                            print(f"[KITTI][tile={tile_idx}] garg-only={gmcount0}/{tot0} ({gmr0*100:.2f}%)")

                        if save_valid_masks:
                            base_name = f"sample_{save_idx:06d}_tile{tile_idx}"  # ← タイル番号を付ける
                            _save_bool_mask_png(vi0.cpu(), (debug_dir / f"{base_name}_valid.png").as_posix())
                            _save_bool_mask_png(base_valid[0,0].cpu(), (debug_dir / f"{base_name}_baseValid.png").as_posix())
                            if garg_m is not None:
                                _save_bool_mask_png(garg_m[0,0].cpu(), (debug_dir / f"{base_name}_gargMask.png").as_posix())
                            _save_overlay_invalid_on_rgb(
                                rgb[0].detach().cpu(), vi0.cpu(),
                                (debug_dir / f"{base_name}_invalidOverlay.png").as_posix()
                            )

                # --- metrics ---
                if is_kitti:
                    res = compute_errors_eigen_style_kitti(
                        depth, pred, min_depth, max_depth,
                        garg_crop_mask=m_tile   # ← 各タイル用マスクを渡す
                    )
                else:
                    res = compute_errors_eigen_style(depth, pred, min_depth, max_depth)
                B = rgb.size(0)
                _accumulate(sums, res, B)
                nimg += B

                # collect heatmap values
                # collect heatmap values
                if plot_dual_heatmap_flag:
                    if is_kitti:
                        valid = _valid_mask_kitti(depth, pred, min_depth, max_depth, m_tile, device)  # ← m_tile を渡す
                    else:
                        valid = _valid_mask_nyu(depth, pred, min_depth, max_depth)
                    _collect_heatmap(all_gt_vals, all_pred_vals, depth, pred, valid, min_depth, max_depth)

                # collect for visualization (first item in B when saving tiles)
                if plot_prediction:
                    r = rgb[0].detach().cpu().permute(1, 2, 0).numpy()
                    d = depth[0, 0].detach().cpu().numpy()
                    p = pred[0, 0].detach().cpu().numpy()
                    rgbs_np.append(r); deps_np.append(d); preds_np.append(p)

            # save one concat figure per sample (tiles)
            if plot_prediction and len(rgbs_np) > 0:
                # 動的な可視化レンジを計算（両方に合わせる）
                viz_vmin, viz_vmax = _compute_viz_limits_from_lists(
                    deps_np, preds_np, min_depth, max_depth, mode="both", percentile=99.5
                )
                out_path = out_dir / f"sample_{save_idx:06d}.png"
                # vmin/vmax を差し替えて渡す
                _viz_kitti_tiles_save(rgbs_np, deps_np, preds_np, out_path, viz_vmin, viz_vmax)
                save_idx += 1

        # ========== NYU: regular batched tensors ==========
        else:
            if len(batch) == 3:
                rgb, depth, _ = batch
            else:
                rgb, depth = batch

            rgb   = rgb.to(device)
            depth = depth.to(device).clamp(min_depth, max_depth)
            pred  = (predict_depth_maps_tta(model, rgb, tau=tau, input_size=input_size)
                     if use_TTA else
                     predict_depth_maps(model, rgb, tau=tau, input_size=input_size))

            # metrics
            res = compute_errors_eigen_style(depth, pred, min_depth, max_depth)
            B = rgb.size(0)
            _accumulate(sums, res, B)
            nimg += B

            # heatmap collect
            if plot_dual_heatmap_flag:
                valid = _valid_mask_nyu(depth, pred, min_depth, max_depth)
                _collect_heatmap(all_gt_vals, all_pred_vals, depth, pred, valid, min_depth, max_depth)

            # visualization (one image per item)
            if plot_prediction:
                rgb_np  = rgb.detach().cpu().permute(0, 2, 3, 1).numpy()
                dep_np  = depth.detach().cpu().squeeze(1).numpy()
                pred_np = pred.detach().cpu().squeeze(1).numpy()

                print("depth: min", dep_np.min(), "max", dep_np.max(), "mean", dep_np.mean())
                print("pred : min", pred_np.min(), "max", pred_np.max(), "mean", pred_np.mean())

                print("valid pixels:", np.sum(np.isfinite(dep_np)))
                save_idx = _viz_nyu_batch_save(rgb_np, dep_np, pred_np, out_dir, save_idx,
                                               min_depth, max_depth)

    # --- heatmap at the end (optional) ---
    if plot_dual_heatmap_flag and len(all_gt_vals) > 0:
        gt_concat   = np.concatenate([x.numpy() for x in all_gt_vals], axis=0)
        pred_concat = np.concatenate([x.numpy() for x in all_pred_vals], axis=0)
        heatmap_dir = Path(save_dir) / "heat_map"
        heatmap_path = heatmap_dir / "heatmap.png"
        plot_dual_heatmap(gt_concat, pred_concat,
                          title_suffix="Heatmap",
                          num_bins=heatmap_num_bins if heatmap_num_bins is not None else getattr(model, 'num_bins', 15),
                          bin_min=min_depth, bin_max=max_depth,
                          save_path=heatmap_path, show=True)

    nimg = max(nimg, 1)
    return {k: sums[k] / nimg for k in names}