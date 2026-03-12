# checkpoint_io_multiopt.py
import re
from pathlib import Path
import torch
# utils/util_metric_tags.py
from typing import Dict, Any, Optional

_EPOCH_RE = re.compile(r"^epoch_(\d+)\.pth$")


def _list_epochs(save_dir):
    """Return sorted list of available epochs in the directory (ascending)."""
    save_dir = Path(save_dir)
    epochs = []
    for p in save_dir.glob("epoch_*.pth"):
        m = _EPOCH_RE.match(p.name)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)


def _ckpt_path(save_dir, epoch):
    return Path(save_dir) / f"epoch_{int(epoch)}.pth"


def _is_optimizer_dict(opt):
    """Heuristic: dict-like of optimizers => True, single optimizer => False."""
    return isinstance(opt, dict)


def _state_dict_from_optimizer(opt):
    """Return state dict payload for checkpoint from optimizer(s)."""
    if _is_optimizer_dict(opt):
        return {k: v.state_dict() for k, v in opt.items()}
    else:
        return opt.state_dict()


def _load_optimizer_state(opt, state):
    """
    Load optimizer state for either a single optimizer or a dict of optimizers.
    - If opt is dict and state is dict: match by keys.
    - If opt is dict and state is single state_dict (backward compatibility):
      try to load into each optimizer (best-effort).
    - If opt is single optimizer:
      - If state is dict-of-dicts and key 'default' exists, try state['default'].
      - Else try state directly.
    """
    try:
        if _is_optimizer_dict(opt):
            if isinstance(state, dict):
                # dict of states -> match keys when possible,
                # else try best-effort load into each optimizer.
                all_values_are_dicts = all(isinstance(v, dict) for v in state.values())
                if all_values_are_dicts:
                    # likely the new format {name: opt_state}
                    matched = False
                    for k, o in opt.items():
                        if k in state:
                            o.load_state_dict(state[k])
                            matched = True
                    if not matched:
                        # Fallback: try load the first state into all
                        try:
                            any_state = next(iter(state.values()))
                            for o in opt.values():
                                o.load_state_dict(any_state)
                        except Exception:
                            pass
                else:
                    # state is a single optimizer-like dict: try load into all
                    for o in opt.values():
                        o.load_state_dict(state)
            else:
                # Unknown structure: ignore
                pass
        else:
            # single optimizer
            if isinstance(state, dict) and all(isinstance(v, dict) for v in state.values()):
                # dict of dicts (multi) – try 'default' or the first available
                if 'default' in state:
                    opt.load_state_dict(state['default'])
                else:
                    any_state = next(iter(state.values()))
                    opt.load_state_dict(any_state)
            else:
                # plain single state dict
                opt.load_state_dict(state)
    except Exception as ex:
        print(f"Warning: failed to load optimizer state: {ex}")


def save_checkpoint_keep_prev(model, epoch, save_dir, optimizer, extra=None):
    """
    Save checkpoint as 'epoch_{epoch}.pth' and prune older ones.
    `extra` は任意メタ（例: {"global_step": 12345}）
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": _state_dict_from_optimizer(optimizer),
        "torch_version": torch.__version__,
    }
    # 任意メタを格納（存在すれば）
    if extra is not None:
        ckpt["extra"] = dict(extra)
        # 旧実装との互換を少し確保（top-levelにも置いておくと読み捨て互換が効く）
        if "global_step" in extra:
            ckpt["global_step"] = int(extra["global_step"])

    # Save atomically-ish: write temp, then replace
    tmp_path = save_dir / f".epoch_{int(epoch)}.pth.tmp"
    final_path = _ckpt_path(save_dir, epoch)
    torch.save(ckpt, tmp_path)
    try:
        tmp_path.replace(final_path)  # atomic on most platforms
    except Exception:
        # fallback
        final_path.unlink(missing_ok=True)
        tmp_path.replace(final_path)

    # Prune: among epochs < current, keep only the max (previous), delete the rest
    epochs = _list_epochs(save_dir)
    smaller = [e for e in epochs if e < epoch]
    if len(smaller) >= 2:
        prev = max(smaller)
        for e in smaller:
            if e != prev:
                p = _ckpt_path(save_dir, e)
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
                except Exception as ex:
                    print(f"Warning: failed to delete {p}: {ex}")


def load_ckpt(save_dir, model, optimizer=None, epoch=None, map_location=None, strict=True):
    """
    Load a checkpoint into the given model and optimizer(s).
    Returns: (model, optimizer, loaded_epoch:int, extra:dict)
    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        raise FileNotFoundError(f"Directory not found: {save_dir}")

    epochs = _list_epochs(save_dir)
    if not epochs:
        raise FileNotFoundError(f"No checkpoints found in: {save_dir}")

    if epoch is None:
        target_epoch = epochs[-1]
    else:
        if epoch in epochs:
            target_epoch = epoch
        else:
            latest = epochs[-1]
            print(f"Requested epoch {epoch} not found. Loaded epoch {latest} instead.")
            target_epoch = latest

    ckpt_path = _ckpt_path(save_dir, target_epoch)
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # Load model (strict by default, configurable)
    model.load_state_dict(ckpt["model_state"], strict=strict)

    # Load optimizer(s), if provided and present in ckpt
    if optimizer is not None and "optimizer_state" in ckpt and ckpt["optimizer_state"]:
        _load_optimizer_state(optimizer, ckpt["optimizer_state"])

    loaded_epoch = int(ckpt.get("epoch", target_epoch))

    # extra を返す（互換のため top-level 'global_step' も拾う）
    extra = ckpt.get("extra", {}) if isinstance(ckpt, dict) else {}
    if "global_step" in ckpt and "global_step" not in extra:
        try:
            extra = dict(extra)
            extra["global_step"] = int(ckpt["global_step"])
        except Exception:
            pass

    return model, optimizer, loaded_epoch, extra



__all__ = ["save_metric_tag_and_prune", "fmt_metric_value"]

# epoch_07_d1_0p8123.pth のような名前を読むため
_METRIC_FILE_RE = re.compile(r"^epoch_(\d+)_([a-zA-Z0-9]+)_([0-9p\-]+)\.pth$")

def fmt_metric_value(v: float, ndigits: int = 4) -> str:
    """
    0.812345 -> '0p8123' のように小数点を 'p' へ置換して固定小数に整形
    """
    s = f"{v:.{ndigits}f}"
    return s.replace('.', 'p')

def save_metric_tag_and_prune(
    save_dir: Path,
    epoch: int,
    metric_key: str,
    metric_value: float,
    state: Dict[str, Any],
    short_map: Optional[Dict[str, str]] = None,
    ndigits: int = 4,
) -> Path:
    """
    metric_key（a1, a2, ...）ごとに 'epoch_<ep>_<short>_<val>.pth' を保存。
    同一メトリクスで epoch_old < epoch の既存ファイルを削除して 1 本だけ残す。

    Returns:
        Path: 保存したファイルパス
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    short = (short_map or {}).get(metric_key, metric_key)
    val_str = fmt_metric_value(metric_value, ndigits=ndigits)
    out_path = save_dir / f"epoch_{epoch:02d}_{short}_{val_str}.pth"

    torch.save(state, out_path)

    # prune: 同じ short を持ち、かつ旧 epoch のファイルを削除
    for p in save_dir.glob(f"epoch_*_{short}_*.pth"):
        m = _METRIC_FILE_RE.match(p.name)
        if not m:
            continue
        ep_old = int(m.group(1))
        short_old = m.group(2)
        if short_old != short:
            continue
        if ep_old < epoch:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
            except Exception as ex:
                print(f"Warning: failed to delete {p}: {ex}")

    return out_path




