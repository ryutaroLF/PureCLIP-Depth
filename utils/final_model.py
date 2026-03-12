import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import matplotlib.pyplot as plt


# ---------------- ViT patch extractor ---------------
@torch.no_grad()
def extract_vit_patches(visual: nn.Module, x: torch.Tensor):
    dtype = visual.conv1.weight.dtype
    x = x.to(dtype)
    B = x.size(0)
    x = visual.conv1(x)
    x = x.reshape(B, visual.conv1.out_channels, -1).permute(0,2,1)
    cls = visual.class_embedding.to(dtype).repeat(B,1,1)
    x = torch.cat([cls, x], 1)
    x = x + visual.positional_embedding.to(dtype)
    x = visual.ln_pre(x)
    x = visual.transformer(x.permute(1,0,2)).permute(1,0,2)
    x = visual.ln_post(x)
    if visual.proj is not None: x = x @ visual.proj
    return x[:,1:,:].float(), x[:,0,:].float()  # patches, cls




class PatchAligner(nn.Module):
    def __init__(self, clip_model, device, *,
                 fusion_type='concat',
                 tau=0.07, lam_nce=1.0,
                 dataset_type='NYU',
                 garg_mask=None,
                 num_bins: int = 15,
                 min_depth: float = 1e-3,
                 max_depth: float = 10.0,
                 input_size: int = 336):
        super().__init__()
        self.num_bins = num_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.input_size = input_size
        
        assert fusion_type in ['concat', 'film']
        self.fusion_type, self.tau, self.lam_nce = fusion_type, tau, lam_nce
        self.clip = clip_model.eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)

        # CLIP normalization buffers
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073],
                         device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711],
                         device=device).view(1, 3, 1, 1)
        )

        # Infer patch count P and embedding dim d
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.input_size, self.input_size, device=device)
            patches, _ = extract_vit_patches(self.clip.visual, self._prep_rgb(dummy))
            self.P, self.d = patches.shape[1:]
        print(f"[INFO] P={self.P}, d={self.d}, fusion={fusion_type}")

        # Depth-bin table
        self.depth_table = nn.Embedding(self.num_bins, self.d)
        bw = (self.max_depth - self.min_depth) / self.num_bins
        init_c = torch.linspace(self.min_depth + 0.5 * bw,
                                self.max_depth - 0.5 * bw,
                                self.num_bins, device=device)
        self.bin_centers = nn.Parameter(init_c)

        prompts = [f"{c.item():.2f} meter" for c in init_c]
        tok = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            zt = self.clip.encode_text(tok).float()
            self.depth_table.weight.copy_(F.normalize(zt, dim=-1))

        # RGB -> depth adapter
        self.rgb_adapter = nn.Sequential(
            nn.LayerNorm(self.d),
            nn.Linear(self.d, 4 * self.d, bias=False), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(4 * self.d, self.d, bias=False),
            nn.LayerNorm(self.d)
        )
        if fusion_type == 'concat':
            self.fusion_mlp = nn.Sequential(
                nn.LayerNorm(2 * self.d),
                nn.Linear(2 * self.d, 4 * self.d, bias=False), nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(4 * self.d, self.d, bias=False),
                nn.LayerNorm(self.d)
            )
        else:
            self.film_gen = nn.Sequential(
                nn.Linear(self.d, 2 * self.d), nn.GELU(),
                nn.Linear(2 * self.d, 2 * self.d)
            )
        self.patch = self.clip.visual.conv1.kernel_size[0]

        # Dataset/mask config
        self.dataset_type = dataset_type
        self.garg_mask = garg_mask  # list of 4 tensors (336x336), or None for NYU        
        if self.dataset_type == 'KITTI' and self.garg_mask is None:
            raise ValueError("KITTI requires `garg_mask` to be provided.")

    # ---------- utils ----------
    def _prep_rgb(self, x):
        return (x - self.clip_mean) / self.clip_std

    #@torch.no_grad()
    def _encode_rgb_patches(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Shared feature extractor: RGB -> normalized patch embeddings z.
        Returns: z (B, P, d)
        """
        B = rgb.size(0)
        patches, _ = extract_vit_patches(self.clip.visual, self._prep_rgb(rgb))
        z_rgb = self.rgb_adapter(patches.view(-1, self.d)).view(B, self.P, self.d)
        cls = F.normalize(self.clip.encode_image(self._prep_rgb(rgb)).float(), dim=-1)
        if self.fusion_type == 'concat':
            z_cat = torch.cat([z_rgb, cls.unsqueeze(1).expand(-1, self.P, -1)], -1)
            z_fused = self.fusion_mlp(z_cat.view(-1, 2 * self.d)).view(B, self.P, self.d)
        else:
            g, b = self.film_gen(cls).chunk(2, -1)
            z_fused = g.unsqueeze(1) * z_rgb + b.unsqueeze(1)
        z = F.normalize(z_fused, dim=-1)
        return z

    def _patchify_depth(self, depth, thr=0.1, rgb_idx=None):
        """
        Compute per-patch ground-truth bin index and a validity mask.
        Returns:
          - true_depth_bin_idx: (B, L) long; -1 means invalid patch
          - patch_valid_mask  : (B, L) float in {0,1}
        NYU:
          - valid pixels: depth > thr (0.1m)
          - bins from mean of valid pixels; mask = (idx >= 0)
        KITTI:
          - same mean depth logic with thr
          - additionally invalidate by Garg crop mask (per-patch mean <= 0.5)
          - NOTE: rgb_idx can be scalar or a (B,) tensor (tile id in {0,1,2,3})
        """
        if self.dataset_type == 'NYU':
            # --- unchanged ---
            p = F.unfold(depth, kernel_size=self.patch, stride=self.patch).squeeze(1)
            valid = p > thr
            cnt = valid.sum(1)
            mean = (p * valid).sum(1) / cnt.clamp(min=1e-6)
            bw = (self.max_depth - self.min_depth) / self.num_bins
            edges = torch.arange(self.min_depth, self.max_depth + 1e-6, bw,
                                 device=depth.device, dtype=mean.dtype)
            true_depth_bin_idx = torch.bucketize(mean, edges) - 1
            true_depth_bin_idx = true_depth_bin_idx.clamp(-1, self.num_bins - 1)
            patch_valid_mask = (true_depth_bin_idx >= 0).float()
            return true_depth_bin_idx, patch_valid_mask

        # -------- KITTI --------
        # depth: (B,1,H,W), assuming H=W=input_size
        B, _, H, W = depth.shape

        # 1) Unfold depth to patches and compute per-patch mean depth
        p = F.unfold(depth, kernel_size=self.patch, stride=self.patch)   # (B, A, L)
        valid_pix = p > thr                                              # (B, A, L)
        cnt = valid_pix.sum(dim=1)                                       # (B, L)
        mean = (p * valid_pix).sum(dim=1) / cnt.clamp(min=1e-6)          # (B, L)

        # 2) Basic validity from depth (at least one valid pixel and mean >= thr)
        depth_valid = (cnt > 0) & (mean >= thr)                          # (B, L) bool

        # 3) Garg crop mask gating (per-sample, supports scalar or (B,) rgb_idx)
        if rgb_idx is None:
            raise ValueError("KITTI requires rgb_idx to be specified.")

        # Convert self.garg_mask (list of 4 tensors (H,W)) into a bank tensor (4,H,W)
        # Do this lazily here to keep the patch minimal (no __init__ changes required).
        bank = torch.stack(self.garg_mask, dim=0).to(depth.device).contiguous()  # (4,H,W)

        # Normalize rgb_idx to a (B,) LongTensor on the correct device
        if isinstance(rgb_idx, torch.Tensor):
            if rgb_idx.ndim == 0:
                idx = rgb_idx.view(1).to(torch.long).expand(B).to(depth.device)   # (B,)
            else:
                idx = rgb_idx.to(torch.long).to(depth.device)                     # (B,)
        elif isinstance(rgb_idx, int):
            idx = torch.full((B,), rgb_idx, dtype=torch.long, device=depth.device)
        else:
            idx = torch.as_tensor(rgb_idx, dtype=torch.long, device=depth.device) # (B,)

        # Clamp just in case (should already be in {0,1,2,3})
        idx = idx.clamp_(0, bank.shape[0] - 1)

        # Gather per-sample masks: (B,H,W) -> (B,1,H,W)
        gmask = bank.index_select(0, idx)                     # (B,H,W)
        gmask = gmask.unsqueeze(1)                            # (B,1,H,W)

        # Unfold per-sample masks and compute per-patch mean (same spatial tiling)
        g_unfold = F.unfold(gmask, kernel_size=self.patch, stride=self.patch)  # (B, A, L)
        g_mean = g_unfold.mean(dim=1)                                          # (B, L)
        garg_valid = (g_mean > 0.5)                                            # (B, L) bool

        # 4) Assign depth bins from mean depth
        bw = (self.max_depth - self.min_depth) / self.num_bins
        edges = torch.arange(self.min_depth, self.max_depth + 1e-6, bw,
                             device=depth.device, dtype=mean.dtype)
        true_depth_bin_idx = torch.bucketize(mean, edges) - 1                  # (B, L)
        true_depth_bin_idx = true_depth_bin_idx.clamp(-1, self.num_bins - 1)

        # 5) Final validity = depth_valid AND garg_valid AND (idx >= 0)
        patch_valid_mask = (true_depth_bin_idx >= 0) & depth_valid & garg_valid  # (B, L) bool
        patch_valid_mask = patch_valid_mask.float()

        # Force invalid patches to idx = -1
        true_depth_bin_idx = torch.where(
            patch_valid_mask > 0,
            true_depth_bin_idx,
            torch.full_like(true_depth_bin_idx, -1)
        )

        return true_depth_bin_idx, patch_valid_mask

    def info_loss(self, rgb, depth, rgb_idx=None):
        """
        Compute Info (Align + NCE) loss. Backward-compatible with original logic.
        """
        z = self._encode_rgb_patches(rgb)                     # (B, P, d)
        idx, mask = self._patchify_depth(depth, rgb_idx=rgb_idx)  # (B, L), (B, L)
        #self.plot_tensors(rgb, depth, mask, self.patch)
        z_d = self.depth_table(idx.clamp(0))
        z_d[idx == -1] = 0
        z_d = F.normalize(z_d, dim=-1)

        L_align = ((1 - (z * z_d).sum(-1)) * mask).sum() / mask.sum().clamp(1)
        W = F.normalize(self.depth_table.weight, dim=-1)
        logits = (z @ W.T) / self.tau
        valid = mask.bool()
        L_nce = F.cross_entropy(logits[valid], idx[valid])
        return L_align + self.lam_nce * L_nce

    def pred_map_from_rgb(self, rgb):
        """
        Predict dense depth map from RGB via patch-wise depth expectation.
        Returns: (B,1,336,336)
        """
        z = self._encode_rgb_patches(rgb)
        pred_patch = self.predict_patch_depth(z)                               # (B, P)
        ph = pw = int(self.input_size / self.patch)
        pred_map = F.interpolate(
            pred_patch.view(rgb.size(0), 1, ph, pw),
            size=(self.input_size, self.input_size), mode='nearest'
        )
        return pred_map

    def rmse_loss(self, rgb, depth):
        """
        RMSE loss on valid pixel range only (kept as in original code for compatibility).
        """
        pred_map = self.pred_map_from_rgb(rgb)
        mask = (depth > self.min_depth) & (depth < self.max_depth)
        rmse = torch.sqrt(((pred_map[mask] - depth[mask]) ** 2).mean())
        return rmse

    # ---------- depth expectation (unchanged) ----------
    def predict_patch_depth(self, z):
        W = F.normalize(self.depth_table.weight, dim=-1)
        logits = (z @ W.T) / self.tau
        probs = F.softmax(logits, -1)
        return (probs * self.bin_centers.view(1, 1, -1)).sum(-1)

    # ---------- forward (compat shim) ----------
    def forward(self, rgb, depth, rgb_idx=None, plot_debug=True):
        """
        Backward-compatible forward: returns (info_loss, z).
        Existing training code that calls `loss, _ = model(rgb, depth)` will keep working.
        """
        z = self._encode_rgb_patches(rgb)
        idx, mask = self._patchify_depth(depth, rgb_idx=rgb_idx)
        z_d = self.depth_table(idx.clamp(0))
        z_d[idx == -1] = 0
        z_d = F.normalize(z_d, dim=-1)

        L_align = ((1 - (z * z_d).sum(-1)) * mask).sum() / mask.sum().clamp(1)
        W = F.normalize(self.depth_table.weight, dim=-1)
        logits = (z @ W.T) / self.tau
        valid = mask.bool()
        L_nce = F.cross_entropy(logits[valid], idx[valid])
        loss = L_align + self.lam_nce * L_nce
        return loss, z
    

    def plot_tensors(self, rgb, depth, mask, patch, out_path="debug_rgb_depth_mask.png", b=0):
        """
        Plot RGB, Depth, and Mask side by side and save as PNG.

        Args:
            rgb   : (B,3,H,W) tensor
            depth : (B,1,H,W) tensor
            mask  : (B,L) tensor (patch-level validity mask)
            patch : int, patch size used in unfolding
            out_path : str, output file path
            b     : int, which batch index to plot
        """
        rgb_img = rgb[b].permute(1,2,0).detach().cpu().numpy()
        depth_img = depth[b,0].detach().cpu().numpy()

        # mask (B,L) -> (B,1,H,W)
        H = W = rgb.shape[-1]
        patch = self.patch  # use same patch size as in model

        # (L,) → (1, patch*patch, L)
        mask_2d = mask[b].unsqueeze(0).unsqueeze(0)   # (1,1,L)
        mask_2d = mask_2d.expand(1, patch*patch, -1)  # (1, patch*patch, L)

        mask_2d = F.fold(mask_2d, output_size=(H,W),
                         kernel_size=patch, stride=patch)    # (1,1,H,W)
        mask_img = mask_2d[0,0].cpu().numpy()

        fig, axs = plt.subplots(1,3,figsize=(12,4))
        axs[0].imshow(rgb_img);   axs[0].set_title("RGB")
        axs[1].imshow(depth_img, cmap="viridis"); axs[1].set_title("Depth")
        axs[2].imshow(mask_img, cmap="gray");     axs[2].set_title("Mask")
        for ax in axs: ax.axis("off")

        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[DEBUG] Saved visualization to {out_path}")