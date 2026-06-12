"""
Shared helpers for the per-modality embedding extractors
(``extract_text.py``, ``extract_audio.py``, ``extract_video.py``).

Each extractor produces an *independent* cache file:

    <cache_dir>/text_embs_<split>.pt
    <cache_dir>/audio_embs_<split>.pt
    <cache_dir>/video_embs_<split>.pt

with the schema::

    {
        "embeddings": FloatTensor (N, K, dim),
        "labels":     list[int],
        "video_ids":  list[str],
        "weights_fingerprint": {"path", "size", "mtime"} | None,
        "extraction_config":   {modality-local config dict},
    }

The fusion-side dataset (``MultimodalCachedFusionDataset`` in
``utils/load_dataset.py``) loads the three files, intersects their
``video_ids``, and re-orders to a common index set so that ``__getitem__``
returns aligned ``(text, audio, video, label)`` triples.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch


# Project root and sys.path setup — keep identical to the legacy script so
# imports of ``utils`` and ``multimodal/src`` modules work from any cwd.
REPO_ROOT = Path('/home3/iasarantsev').resolve()
if str(REPO_ROOT / "utils") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "utils"))
if str(REPO_ROOT / "multimodal" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "multimodal" / "src"))


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def weights_fingerprint_one(weights_path) -> dict:
    """Capture (path, size, mtime) for one backbone checkpoint.

    Returns a dict whose entries are ``None`` if the file does not exist —
    callers may still want the path for diagnostics.
    """
    if weights_path is None:
        return {"path": "", "size": None, "mtime": None}
    wp = Path(weights_path)
    if wp.exists():
        st = wp.stat()
        return {"path": str(wp), "size": st.st_size, "mtime": int(st.st_mtime)}
    return {"path": str(wp), "size": None, "mtime": None}


def check_existing_cache(
    cache_path: Path,
    expected_fp: dict,
    expected_ext_cfg: dict,
    force: bool,
) -> bool:
    """Return True iff the on-disk cache can be reused.

    A cache is reused when:
    - the file exists,
    - its ``weights_fingerprint`` matches the current backbone checkpoint,
    - its ``extraction_config`` matches the requested config, and
    - the embeddings tensor has shape (N, K, dim).
    """
    if force or not cache_path.exists():
        return False
    try:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [Cache] Failed to load {cache_path}: {e!r} — recomputing")
        return False
    if cached.get("weights_fingerprint") != expected_fp:
        print(f"  [Cache] Fingerprint mismatch in {cache_path.name} — recomputing")
        return False
    if cached.get("extraction_config") != expected_ext_cfg:
        print(f"  [Cache] extraction_config mismatch in {cache_path.name} — recomputing")
        return False
    t = cached.get("embeddings")
    if not (isinstance(t, torch.Tensor) and t.ndim == 3):
        print(
            f"  [Cache] {cache_path.name}: 'embeddings' has unexpected shape "
            f"{getattr(t, 'shape', None)} — recomputing"
        )
        return False
    return True


def load_extraction_settings(cfg: dict, modality: str):
    """Pull the per-modality extraction settings from the multimodal config.

    Returns
    -------
    (K_train, K_eval, augment_flag, batch_size, seed, splits)
    """
    ext_cfg = cfg.get("extraction", {})
    K_train = int(ext_cfg.get("num_views_train", 5))
    K_eval = int(ext_cfg.get("num_views_eval", 1))

    # Per-modality augmentation toggle, with backward-compat fallback to the
    # legacy single ``augment_train`` flag.
    legacy = bool(ext_cfg.get("augment_train", True))
    key = f"augment_{modality}"
    augment = bool(ext_cfg.get(key, legacy))

    batch = int(ext_cfg.get("batch_size_embed", 8))
    seed = int(cfg.get("training", {}).get("seed", 42))
    splits = list(ext_cfg.get("splits", ["train", "val", "test"]))

    return K_train, K_eval, augment, batch, seed, splits


def cache_path_for(
    cache_dir: Path,
    modality: str,
    split: str,
    variant: Optional[str] = None,
) -> Path:
    """Per-modality cache filename.

    Examples
    --------
    >>> cache_path_for(d, "text", "train")
    .../text_embs_train.pt
    >>> cache_path_for(d, "video", "train", variant="swin")
    .../video_swin_embs_train.pt
    >>> cache_path_for(d, "video", "train", variant="videomae")
    .../video_videomae_embs_train.pt

    The ``variant`` slot is used by the video modality to keep separate
    caches for each backbone (Swin vs VideoMAE) without overwriting one
    another.  Modalities that omit ``variant`` keep the legacy filename.
    """
    if variant:
        return Path(cache_dir) / f"{modality}_{variant}_embs_{split}.pt"
    return Path(cache_dir) / f"{modality}_embs_{split}.pt"
