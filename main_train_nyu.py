#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v300 : v249 のコピー, kitti
"""


##############################
version = 'v301'
##############################


import os, configparser
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import clip
from PIL import Image
import numpy as np

from utils.dataset7_NYU_KITTI import CustomDatasetUnified
from utils.final_model import PatchAligner
from utils.util_checkpoint import save_checkpoint_keep_prev, load_ckpt, save_metric_tag_and_prune  
from utils.util_eval import compute_errors_eigen_style, predict_depth_maps, eval_metrics

# ----------------------- Data -----------------------
def load_cfg(path: str):
    cfg = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    if not cfg.read(path):
        raise FileNotFoundError(f"INI not found: {path}")
    return cfg


def build_loaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    H = W = 336

    root_key = 'DATASET_WIN' if os.name == 'nt' else 'DATASET_TSUBAME'
    if cfg.get('DATASET','dataset_type',fallback='NYU').upper() == 'KITTI':
        root_dir = cfg.get(root_key, 'kitti_dataset_root_dir_path')
        eigen_crop = None
        dataset_type = "KITTI"
    else:
        root_dir = cfg.get(root_key, 'nyu_dataset_root_dir_path')
        eigen_crop = (40, 470, 38, 603)
        dataset_type = "NYU"

    bs = cfg.getint('TRAINING', 'batch_size', fallback=8)
    ds_train = CustomDatasetUnified(root_dir, "train",      H, W, dataset_type, eigen_crop)
    ds_val   = CustomDatasetUnified(root_dir, "validation", H, W, dataset_type, eigen_crop)
    ds_test  = CustomDatasetUnified(root_dir, "test",       H, W, dataset_type, eigen_crop)
    kw_train = dict(batch_size=bs, num_workers=0, pin_memory=True, drop_last=True)
    
    if dataset_type == "KITTI":
        kw_eval = dict(batch_size=1, num_workers=0, pin_memory=True, drop_last=False)
    else:
        kw_eval = dict(batch_size=bs, num_workers=0, pin_memory=True, drop_last=True)
    return (
        DataLoader(ds_train, shuffle=True,  **kw_train),
        DataLoader(ds_val,   shuffle=False, **kw_eval),
        DataLoader(ds_test,  shuffle=False, **kw_eval)
    )

def load_garg_masks(cfg, out_h: int = 336, out_w: int = 336):
    from torchvision import transforms  # ← 忘れずに

    root_key = 'DATASET_WIN' if os.name == 'nt' else 'DATASET_TSUBAME'
    path = cfg.get(root_key, 'kitti_Garg_crop_mask_path', fallback=None)
    if not path:
        raise KeyError("cfg[DATASET][kitti_Garg_crop_mask_path] is not set.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask file not found: {path}")

    # 0/1想定のマスク（0–255を0–1に）
    arr = np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.0  # (H, W)

    # (1,1,H,W) → (1,H,W) で Resize (NEAREST)
    t = torch.from_numpy(arr).to(dtype=torch.float32)[None, None, ...]  # (1,1,H,W)
    resize = transforms.Resize((out_h, out_w * 4), interpolation=transforms.InterpolationMode.NEAREST)
    resized = resize(t.squeeze(0))  # (1,H,W) → (1,out_h,out_w*4)
    resized = resized.unsqueeze(0).contiguous()  # (1,1,out_h,out_w*4)

    # 4分割（幅方向）
    chunks = torch.chunk(resized, chunks=4, dim=-1)
    garg_mask_list = [c.squeeze(0).squeeze(0).contiguous() for c in chunks]   # 各 (out_h,out_w)
    garg_mask_all  = resized.squeeze(0).squeeze(0).contiguous()               # (out_h, out_w*4)

    return garg_mask_list, garg_mask_all

#######################
# For metrics improvement and save
#######################
_IMPROVE_UP   = {'a1','a2','a3'}
_IMPROVE_DOWN = {'abs_diff','abs_rel','log10','rmse_tot'}

_SHORT = {'a1':'d1','a2':'d2','a3':'d3',
          'rmse_tot':'rmse','abs_rel':'rel','abs_diff':'abs','log10':'log10'}

def _init_best_metrics():
    best = {}
    for k in _IMPROVE_UP:   best[k] = float('-inf')
    for k in _IMPROVE_DOWN: best[k] = float('inf')
    return best

def _check_improvements(new, best):
    """
    new: dict, best: dict
    戻り: (improved_keys:list, updated_best:dict)
    """
    improved = []
    updated = dict(best)
    for k, v in new.items():
        if k in _IMPROVE_UP:
            if v > best[k]:
                updated[k] = v; improved.append(k)
        elif k in _IMPROVE_DOWN:
            if v < best[k]:
                updated[k] = v; improved.append(k)
    return improved, updated

# ------------------ Training helpers ----------------
def set_phase(model, phase):
    """phase: 'info' or 'center'"""
    req = True
    for p in model.rgb_adapter.parameters(): p.requires_grad_(req)
    if model.fusion_type=='concat':
        for p in model.fusion_mlp.parameters(): p.requires_grad_(req)
    else:
        for p in model.film_gen.parameters(): p.requires_grad_(req)
    for p in model.depth_table.parameters(): p.requires_grad_(req)
    model.bin_centers.requires_grad_(False)
    model.train()

def _apply_lr(optimizer: optim.Optimizer, base_lr: float, factor: float):
    """optimizer の全param_groupへ base_lr * factor を適用"""
    new_lr = base_lr * max(0.1, factor)  # 念のため下限0.1*base_lrを維持
    for pg in optimizer.param_groups:
        pg['lr'] = new_lr
    return new_lr



def visualize_garg_masks(garg_mask_list, save_dir, dataset_type):
    """
    KITTI の Garg マスクリストを横並びでサブプロットして PNG 保存する。
    保存先: save_dir / "garg_mask_list" / "garg_masks.png"
    """
    if dataset_type != 'KITTI':
        return
    if not isinstance(garg_mask_list, (list, tuple)) or len(garg_mask_list) == 0:
        return

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    out_dir = save_dir / "garg_mask_list"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(garg_mask_list)
    fig_w = max(4 * n, 8)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4), squeeze=False)

    for i, m in enumerate(garg_mask_list):
        ax = axes[0, i]
        # Tensor → numpy（0/1のマスク想定）
        m_np = m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
        ax.imshow(m_np, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"mask[{i}]", fontsize=10)
        ax.axis("off")

    fig.suptitle("KITTI Garg crop masks (resized & split into 4 tiles)", fontsize=12)
    fig.tight_layout()
    fig.savefig((out_dir / "garg_masks.png").as_posix(), dpi=150)
    plt.close(fig)

# ------------------------- Main ----------------------
def main():
    cfg = load_cfg(f"./config/config_{version}.ini")
    base_sd = Path(cfg.get('TRAINING','model_save_dir_path',fallback='./ckpt'))
    save_dir = base_sd / cfg.get('IDENTIFICATION','model_identifier',fallback='v145_bin_center')
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get('TRAINING','device',fallback='cuda'))
    fusion = cfg.get('TRAINING','fusion_type',fallback='concat')
    max_epoch = cfg.getint('TRAINING','num_epochs',fallback=20)
    lr_mlp = cfg.getfloat('TRAINING','lr_mlp',fallback=3e-4)
    lr_bin_center = cfg.getfloat('TRAINING','lr_bin_center',fallback=1e-3)
    wd = cfg.getfloat('TRAINING','weight_decay',fallback=1e-2)
    patience = cfg.getint('TRAINING','patience',fallback=50)
    early = cfg.getboolean('TRAINING','early_stopping',fallback=True)
    num_bins   = cfg.getint('TRAINING', 'num_bins', fallback=15)
    input_size = cfg.getint('TRAINING', 'input_size', fallback=336)
    switch_steps = cfg.getint('TRAINING', 'switch_steps', fallback=100)
    counter_start_decay = cfg.getint('TRAINING', 'counter_start_decay', fallback=20)
    dataset_type = cfg.get('DATASET','dataset_type',fallback='NYU').upper()
    garg_mask_list, garg_mask_all = load_garg_masks(cfg) if dataset_type == 'KITTI' else (None, None)

    tqdm_disable = cfg.getboolean('TRAINING','tqdm_disable', fallback=False)
    plot_prediction = cfg.getboolean('TEST','plot_prediction', fallback=False)
    test_only = cfg.getboolean('TEST','test_only', fallback=False)
    plot_dual_heatmap = cfg.getboolean('TEST','plot_dual_heatmap', fallback=False)
    use_TTA = cfg.getboolean('TEST','use_TTA', fallback=False)
    debug_valid = cfg.getboolean('TEST','debug_valid', fallback=False)
    save_valid_masks = cfg.getboolean('TEST','save_valid_masks', fallback=True)
    
    min_depth_nyu  = cfg.getfloat('TRAINING', 'min_depth_nyu', fallback=1e-3)
    max_depth_nyu  = cfg.getfloat('TRAINING', 'max_depth_nyu', fallback=10.0)
    min_depth_kitti = cfg.getfloat('TRAINING', 'min_depth_kitti', fallback=1e-3)
    max_depth_kitti = cfg.getfloat('TRAINING', 'max_depth_kitti', fallback=80.0)
    if dataset_type == 'NYU':
        min_depth = min_depth_nyu
        max_depth = max_depth_nyu
    else:
        min_depth = min_depth_kitti
        max_depth = max_depth_kitti
    print("~"*40)
    print(f"[Info] Dataset: {dataset_type}, min_depth: {min_depth}, max_depth: {max_depth}")
    print(f"[Info] Model version: {version}, fusion: {fusion}"
          f", input_size: {input_size}, garg_mask: {'Yes' if dataset_type == 'KITTI' else 'No'}"
          f", device: {device}, save_dir: {save_dir}")
    print(f"Depth min: {min_depth}, max: {max_depth}, num_bins: {num_bins}")
    print("~"*40)
    ########################
    # Model
    ########################

    clip_model,_ = clip.load(
        cfg.get('CLIP','clip_model_name',fallback='ViT-L/14@336px'),
        device=device,jit=False)
    
    model = PatchAligner(
        clip_model, device,
        fusion_type=cfg.get('TRAINING','fusion_type',fallback='concat'),
        tau=cfg.getfloat('TRAINING','tau',fallback=0.07),
        lam_nce=cfg.getfloat('TRAINING','lam_nce',fallback=1.0),
        dataset_type=dataset_type,
        garg_mask=garg_mask_list,
        num_bins=num_bins,
        min_depth=min_depth,
        max_depth=max_depth,
        input_size=input_size,
    ).to(device)

    ######################
    # Optimizer Parameters
    ######################

    mlp_params = list(model.rgb_adapter.parameters())
    mlp_params += (list(model.fusion_mlp.parameters())
                   if fusion=='concat' else
                   list(model.film_gen.parameters()))
    mlp_params += list(model.depth_table.parameters())

    opt_info   = optim.AdamW(mlp_params, lr=lr_mlp,   weight_decay=wd)
    opt_center = optim.AdamW(mlp_params, lr=lr_bin_center, weight_decay=0)
    opt_dict = {"info": opt_info, "center": opt_center}

    wait,global_step = 0,0
    

    #######################
    # Test Only Mode
    #######################
    if test_only:
        visualize_garg_masks(garg_mask_list, save_dir, dataset_type)
        print("[Mode] test_only=True → training/validation skipped.")
        try:
            model, _, loaded_epoch, extra = load_ckpt(save_dir, model, optimizer=None, epoch=None)
            print(f"=> Loaded latest epoch checkpoint: epoch={loaded_epoch}")
        except Exception as e:
            print(f"[Warn] Failed to load latest epoch ckpt via load_ckpt: {e}")
            # epoch_*.pth または epoch_*_tags.pth から最大epochを拾う
            cands = list(save_dir.glob("epoch_*.pth"))
            if not cands:
                raise FileNotFoundError(f"No checkpoint files found in {save_dir}")
            import re
            _E = re.compile(r"^epoch_(\d+)(?:_.+)?\.pth$")
            def _epoch(p):
                m = _E.match(p.name)
                return int(m.group(1)) if m else -1
            cands.sort(key=_epoch)
            ckpt_path = cands[-1]
            ck = torch.load(ckpt_path, map_location=device)
            sd = ck.get('model', ck.get('model_state', ck)) if isinstance(ck, dict) else ck
            model.load_state_dict(sd, strict=True)
            print(f"=> Fallback loaded: {ckpt_path.name}")

        _, _, test_loader = build_loaders(cfg)
        test_metrics = eval_metrics(model, tqdm(test_loader,desc="Testing",unit="batch",disable=tqdm_disable), device,
                                    min_depth, max_depth,
                                    tau=cfg.getfloat('TRAINING','tau',fallback=0.07),
                                    input_size=input_size,plot_prediction=plot_prediction, \
                                    save_dir = save_dir,
                                    plot_dual_heatmap_flag=plot_dual_heatmap,
                                    heatmap_num_bins=num_bins,
                                    garg_crop_mask=garg_mask_list, # ! Give list
                                    use_TTA=use_TTA,
                                    dataset_type=dataset_type,
                                    debug_valid=debug_valid,
                                    save_valid_masks=save_valid_masks)
        print("=== Test-only Results ===")
        for k, v in test_metrics.items():
            print(f"{k:8s}: {v:.4f}")
        return  


    #####################
    # Resume Training
    #####################

    resume_epoch = cfg.getint('TRAINING', 'resume_epoch', fallback=-1)

    start_epoch = 1
    if resume_epoch != 0:
        try:
            target = None if resume_epoch < 0 else resume_epoch
            model, opt_dict, loaded_epoch, extra = load_ckpt(
                save_dir, model, optimizer=opt_dict, epoch=target, strict=True
            )
            global_step = int(extra.get("global_step", 0))
            print(f"[Resume] loaded epoch {loaded_epoch} from {save_dir}")
            start_epoch = loaded_epoch + 1
        except FileNotFoundError as e:
            print(f"[Resume skipped] {e}")

    #####################
    # Training Loop
    #####################
    best_metrics = _init_best_metrics()
    train_loader,val_loader,test_loader = build_loaders(cfg)

    for ep in range(start_epoch, max_epoch+1):
        pbar = tqdm(train_loader, desc=f"Train E{ep}", disable=tqdm_disable)
        run_info = run_center = n_info = n_center = 0.0

        for rgb, depth, *maybe_idx in pbar:
            rgb, depth = rgb.to(device), depth.to(device)
            rgb_idx = maybe_idx[0] if len(maybe_idx) > 0 else None  # NYUはNone、KITTIはk(0..3)

            if rgb_idx is not None:
                if isinstance(rgb_idx, torch.Tensor):
                    if rgb_idx.ndim == 0:
                        rgb_idx = torch.full((rgb.shape[0],), int(rgb_idx.item()),
                                             dtype=torch.long, device=device)
                    else:
                        rgb_idx = rgb_idx.to(device=device, dtype=torch.long)
                elif isinstance(rgb_idx, int):
                    rgb_idx = torch.full((rgb.shape[0],), rgb_idx,
                                         dtype=torch.long, device=device)
                else:
                    rgb_idx = torch.as_tensor(rgb_idx, dtype=torch.long, device=device)
            # ================================================================

            phase = 'info' if (global_step // switch_steps) % 2 == 0 else 'center'
            if phase == 'info':
                set_phase(model, 'info')
                opt_info.zero_grad(set_to_none=True)
                loss = model.info_loss(rgb, depth, rgb_idx=rgb_idx)
                loss.backward(); opt_info.step()
                run_info += loss.item(); n_info += 1
            else:
                set_phase(model, 'center')
                opt_center.zero_grad(set_to_none=True)
                rmse = model.rmse_loss(rgb, depth)
                rmse.backward(); opt_center.step()
                run_center += rmse.item(); n_center += 1

            global_step += 1
            pbar.set_postfix(info=f"{run_info/n_info:.3f}" if n_info else "-",
                             center=f"{run_center/n_center:.3f}" if n_center else "-")

        #######################
        # Validation metrics calculation
        #######################
        val_metrics = eval_metrics(model, val_loader, device,
                                   min_depth, max_depth,
                                   tau=cfg.getfloat('TRAINING','tau',fallback=0.07),
                                   input_size=input_size,
                                    garg_crop_mask=garg_mask_list, # ! Give list
                                   use_TTA=use_TTA,
                                   dataset_type=dataset_type)
        mstr = " | ".join([f"{k}:{val_metrics[k]:.4f}" for k in ['a1','a2','a3','rmse_tot','abs_rel','log10','abs_diff']])
        print(f"[Epoch {ep:03d}] {mstr}")

        improved, best_metrics = _check_improvements(val_metrics, best_metrics)

        if improved:
            wait = 0

            save_checkpoint_keep_prev(
                model, epoch=ep, save_dir=save_dir, optimizer=opt_dict,
                extra={"global_step": global_step}
            )

            common_state = {
                'epoch': ep,
                'val_metrics': val_metrics,
                'model': model.state_dict(),
            }
            for k in improved:
                save_metric_tag_and_prune(
                    save_dir=save_dir,
                    epoch=ep,
                    metric_key=k,
                    metric_value=val_metrics[k],
                    state=common_state,
                    short_map=_SHORT,   # {'a1': 'd1', ...}
                    ndigits=4
                )

            tags_shown = ", ".join([_SHORT.get(k, k) for k in improved])
            print(f"  -> saved metric tags for: {tags_shown}")
        else:
            wait += 1
            print(f"  -> no metric improved (wait={wait}/{patience})")
            if early and wait >= patience:
                print(f"Early stop @ epoch {ep} | best (so far): " +
                      " ".join([f"{k}:{best_metrics[k]:.4f}" for k in ['a1','a2','a3','rmse_tot','abs_rel','log10','abs_diff']]))
                break
        

        #########################
        # Learning Rate Decay
        #########################
        if early:
            if wait < counter_start_decay:
                factor = 1.0
            else:
                # progress: counter_start_decay → patience  0 → 1
                denom = max(1, patience - counter_start_decay)
                progress = min(1.0, (wait - counter_start_decay) / denom)
                # 1.0 → 0.1 (linear)
                factor = 1.0 - 0.9 * progress

            cur_lr_mlp = _apply_lr(opt_info,   lr_mlp,       factor)
            cur_lr_ctr = _apply_lr(opt_center, lr_bin_center, factor)
            print(f"    LR(updated): info={cur_lr_mlp:.6g}, center={cur_lr_ctr:.6g}, wait={wait}, factor={factor:.3f}")


    print("Training finished | best (so far): " +
          " ".join([f"{k}:{best_metrics[k]:.4f}" for k in ['a1','a2','a3','rmse_tot','abs_rel','log10','abs_diff']]))

    test_metrics = eval_metrics(model, test_loader, device,
                                min_depth, max_depth,
                                tau=cfg.getfloat('TRAINING','tau',fallback=0.07),
                                input_size=input_size)
    print("=== Test Results ===")
    for k, v in test_metrics.items():
        print(f"{k:8s}: {v:.4f}")


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    torch.backends.cuda.matmul.allow_tf32 = True
    main()