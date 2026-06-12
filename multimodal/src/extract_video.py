"""
Video Embedding Extraction (stage 1 — video only)
==================================================

Pre-computes K augmented Swin embeddings per sample, then saves them to::

    <cache_dir>/video_embs_<split>.pt

Usage
-----
    cd multimodal/src
    python extract_video.py
    FORCE_EXTRACT=1 python extract_video.py
    VIDEO_WEIGHTS=/path/to/swin.pth python extract_video.py
"""

import os
import time
from pathlib import Path

import numpy as np
import toml
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from _extract_common import (
    REPO_ROOT,
    resolve_device,
    weights_fingerprint_one,
    check_existing_cache,
    load_extraction_settings,
    cache_path_for,
)


def augment_clip(
    clip: torch.Tensor, rng: np.random.Generator, cfg: dict
) -> torch.Tensor:
    """
    Video augmentation on a (T, C, H, W) clip in [0,1].
      - random horizontal flip (consistent across the clip's frames)
      - random brightness / contrast jitter (consistent per clip)
      - random frame subsampling (replace K random frames with neighbours)
    """
    out = clip.clone()
    T, C, H, W = out.shape

    if rng.random() < float(cfg.get("video_hflip_prob", 0.5)):
        out = torch.flip(out, dims=[3])

    bj = float(cfg.get("video_brightness", 0.2))
    cj = float(cfg.get("video_contrast", 0.2))
    if bj > 0:
        b = 1.0 + rng.uniform(-bj, bj)
        out = (out * b).clamp(0.0, 1.0)
    if cj > 0:
        c = 1.0 + rng.uniform(-cj, cj)
        mean = out.mean(dim=(2, 3), keepdim=True)
        out = ((out - mean) * c + mean).clamp(0.0, 1.0)

    n_frame_jitter = int(cfg.get("video_n_frame_jitter", 2))
    if n_frame_jitter > 0 and T > 2:
        for _ in range(min(n_frame_jitter, T // 2)):
            idx = int(rng.integers(0, T))
            neighbour = max(0, min(T - 1, idx + int(rng.choice([-1, 1]))))
            out[idx] = out[neighbour]

    return out


def _extract_one_split(
    split: str,
    K: int,
    augment_flag: bool,
    video_embedder,
    video_processor,
    cfg: dict,
    aug_cfg: dict,
    device,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> dict:
    """Run multi-view video extraction for one split.

    Performance notes
    -----------------
    Earlier revisions created one ``DataLoader`` per video with
    ``num_workers=0`` and a tiny ``batch_size`` — this serialised disk I/O
    against GPU compute and left the GPU largely idle (≈8 % SM utilisation,
    ≈7 GB VRAM on an 80 GB card).  The current loop instead drives a single
    ``DataLoader`` over the entire ``video_ds`` with workers + pin_memory +
    persistent_workers so that JPEG decoding for the next clip batch
    overlaps with the previous batch's Swin forward pass, and scatters the
    resulting per-clip embeddings into per-video sums afterwards.
    """
    from load_dataset import get_segmented_video_clip_splits

    video_ds = get_segmented_video_clip_splits(
        image_size=cfg["video"]["image_size"],
        clip_len=cfg["video"]["clip_len"],
        clip_stride=cfg["video"]["clip_stride"],
        pad_last=True,
        label_agg=cfg["video"]["label_agg"],
    )[split]

    # Build {video_id → output position} and {clip_idx → output position}.
    vid_label = {}
    seen_order = []
    for vfolder, label, _paths in video_ds.clips:
        stem = Path(vfolder).stem
        if stem not in vid_label:
            vid_label[stem] = int(label)
            seen_order.append(stem)

    video_ids = sorted(vid_label.keys())
    vid_to_pos = {v: i for i, v in enumerate(video_ids)}
    labels = [vid_label[v] for v in video_ids]
    N = len(video_ids)

    clip_to_pos = np.empty(len(video_ds.clips), dtype=np.int64)
    for ci, (vfolder, _label, _paths) in enumerate(video_ds.clips):
        clip_to_pos[ci] = vid_to_pos[Path(vfolder).stem]
    clip_to_pos_t = torch.from_numpy(clip_to_pos)

    print(
        f"  [Split {split}] {N} videos  |  K={K}  |  augment={augment_flag}  "
        f"|  clips={len(video_ds.clips):,}  batch={batch_size}  workers={num_workers}"
    )

    out = torch.zeros(N, K, video_embedder.dim, dtype=torch.float32)

    # ── Bake the HF image processor's ImageNet normalization into tensors ─────
    # The dataset already yields (T, C, H, W) in [0,1] at the target image size
    # (see SegmentedVideoClipDataset which cv2-resizes to image_size).  All the
    # AutoImageProcessor adds on top of that is ImageNet mean/std normalisation
    # — but it does so by routing every frame through a Python list of numpy
    # arrays + PIL conversion, which would dominate cost.  Fused on device.
    mean = torch.tensor(
        list(video_processor.image_mean), dtype=torch.float32
    ).view(1, 1, 3, 1, 1).to(device)
    std = torch.tensor(
        list(video_processor.image_std), dtype=torch.float32
    ).view(1, 1, 3, 1, 1).to(device)

    def _stack_collate(batch):
        # batch = list of (clip_tensor (T,C,H,W), label) — stack & drop labels.
        clips = torch.stack([c for (c, _) in batch], dim=0)  # (B, T, C, H, W)
        return clips

    # One DataLoader for the entire split — workers prefetch the next clip
    # batch while the GPU is busy.  ``shuffle=False`` keeps clip order stable
    # so a running index counter can map each batch back to ``clip_to_pos``.
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_stack_collate,
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    for view in range(K):
        view_seed = seed * 9973 + view * 17 + (0 if split == "train" else 1)
        rng = np.random.default_rng(view_seed)
        apply_aug = augment_flag and (view > 0)
        print(
            f"  [Split {split}] Video view {view + 1}/{K}  "
            f"[{'aug' if apply_aug else 'clean'}]"
        )

        sums = torch.zeros(N, video_embedder.dim, dtype=torch.float32)
        counts = torch.zeros(N, dtype=torch.float32)

        loader = DataLoader(video_ds, **loader_kwargs)

        clip_offset = 0
        for batch in tqdm(
            loader,
            desc=f"    video [{split} v{view}]",
            leave=False,
            total=len(loader),
        ):
            B = batch.shape[0]
            if apply_aug:
                # CPU-side per-clip augmentation; cheap relative to disk I/O.
                for b in range(B):
                    batch[b] = augment_clip(batch[b], rng, aug_cfg)
            pv = batch.to(device, non_blocking=True)
            pv = (pv - mean) / std
            emb = video_embedder.embed(pv).detach().cpu()  # (B, dim)

            positions = clip_to_pos_t[clip_offset : clip_offset + B]
            sums.index_add_(0, positions, emb)
            counts.index_add_(0, positions, torch.ones(B, dtype=torch.float32))
            clip_offset += B

        # Per-video mean = sum / count.  All videos should have count>0
        # because every video contributes at least one clip; clamp for safety.
        out[:, view, :] = sums / counts.unsqueeze(1).clamp(min=1.0)

    return {
        "embeddings": out,
        "labels": labels,
        "video_ids": video_ids,
    }


def _resolve_video_cfg(CFG: dict) -> dict:
    """
    Layer the shared ``[video]`` block with the active per-backbone block.

    The resolved dict carries everything :func:`_extract_one_split` needs:
    ``backbone``, ``model_name``, ``image_size``, ``clip_len``,
    ``clip_stride``, ``label_agg``, ``weights_path``.  Per-backbone keys
    override shared ones; shared keys provide fallbacks.
    """
    v = dict(CFG.get("video", {}))
    backbone = str(v.get("backbone", "swin")).lower()
    sub = v.get(backbone, {}) or {}
    if not sub:
        # Backward-compat: treat the flat legacy block (model_name/clip_len
        # at the top level of [video]) as the swin sub-block.
        sub = {
            k: v[k] for k in ("model_name", "clip_len", "clip_stride", "weights_path")
            if k in v
        }
    resolved = {**v, **sub, "backbone": backbone}
    # Strip nested sub-blocks from the flat view to avoid confusion later.
    for k in ("swin", "videomae"):
        resolved.pop(k, None)
    return resolved


def main():
    from embedders import build_video_embedder
    from transformers import AutoImageProcessor

    CFG = toml.load("config.toml")
    aug_cfg = CFG.get("augmentation", {})
    K_TRAIN, K_EVAL, AUGMENT, BATCH, SEED, SPLITS = load_extraction_settings(
        CFG, modality="video"
    )

    vcfg = _resolve_video_cfg(CFG)
    BACKBONE = vcfg["backbone"]
    MODEL_NAME = vcfg["model_name"]
    IMAGE_SIZE = int(vcfg["image_size"])
    CLIP_LEN = int(vcfg["clip_len"])
    CLIP_STRIDE = int(vcfg["clip_stride"])
    LABEL_AGG = vcfg["label_agg"]

    # Make the resolved values available to ``_extract_one_split`` through
    # the legacy ``cfg["video"][...]`` access pattern without rewriting it.
    CFG["video"] = {
        **CFG.get("video", {}),
        "model_name": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "clip_len": CLIP_LEN,
        "clip_stride": CLIP_STRIDE,
        "label_agg": LABEL_AGG,
    }

    CACHE_DIR_RAW = CFG["fusion"].get("cache_dir", "multimodal/cache")
    CACHE_DIR = REPO_ROOT / CACHE_DIR_RAW
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = resolve_device()
    FORCE = bool(int(os.environ.get("FORCE_EXTRACT", "0")))
    WEIGHTS = Path(
        os.environ.get(
            "VIDEO_WEIGHTS", str(REPO_ROOT / vcfg["weights_path"])
        )
    )

    print("=" * 70)
    print(f"Video Embedding Extraction — BAH A/H Multimodal  (backbone={BACKBONE})")
    print("=" * 70)
    print(f"Device          : {DEVICE}")
    print(f"Cache dir       : {CACHE_DIR}")
    print(f"Backbone        : {BACKBONE}")
    print(f"K train         : {K_TRAIN}")
    print(f"K eval          : {K_EVAL}")
    print(f"Augment train   : {AUGMENT}")
    print(f"Splits          : {SPLITS}")
    print(f"Force re-extract: {FORCE}")
    print(f"Batch size      : {BATCH}")
    print(f"Seed            : {SEED}")
    print(f"Model           : {MODEL_NAME}")
    print(f"Weights         : {WEIGHTS}")
    print(f"Clip len/stride : {CLIP_LEN}/{CLIP_STRIDE}")
    print("=" * 70)

    fp = weights_fingerprint_one(WEIGHTS)

    def expected_cfg(split: str) -> dict:
        K = K_TRAIN if split == "train" else K_EVAL
        do_aug = AUGMENT and (split == "train")
        return {
            "modality": "video",
            "backbone": BACKBONE,
            "num_views": K,
            "augment": do_aug,
            "augmentation_cfg": aug_cfg if do_aug else None,
            "model_name": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "clip_len": CLIP_LEN,
            "clip_stride": CLIP_STRIDE,
            "label_agg": LABEL_AGG,
            "seed": SEED,
        }

    splits_to_do = []
    for split in SPLITS:
        cache_path = cache_path_for(CACHE_DIR, "video", split, variant=BACKBONE)
        if check_existing_cache(cache_path, fp, expected_cfg(split), FORCE):
            print(
                f"[{split:5s}] Cache OK at {cache_path} — skipping (use "
                f"FORCE_EXTRACT=1 to override)"
            )
        else:
            splits_to_do.append(split)

    if not splits_to_do:
        print("\nAll video caches up to date — nothing to do.")
        return

    print(f"\n[Backbone] Loading {BACKBONE} video encoder ...")
    video_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    video_embedder = build_video_embedder(
        backbone=BACKBONE,
        model_name=MODEL_NAME,
        weights_path=WEIGHTS,
        device=DEVICE,
    )

    for split in splits_to_do:
        K = K_TRAIN if split == "train" else K_EVAL
        cache_path = cache_path_for(CACHE_DIR, "video", split, variant=BACKBONE)
        print(f"\n{'=' * 70}\n[{split:5s}] Video ({BACKBONE}) → {cache_path}\n{'=' * 70}")
        t0 = time.time()
        result = _extract_one_split(
            split=split,
            K=K,
            augment_flag=AUGMENT and (split == "train"),
            video_embedder=video_embedder,
            video_processor=video_processor,
            cfg=CFG,
            aug_cfg=aug_cfg,
            device=DEVICE,
            batch_size=BATCH,
            seed=SEED,
            num_workers=int(CFG.get("extraction", {}).get("video_num_workers", 0)),
        )
        dt = time.time() - t0
        print(
            f"[{split:5s}] Done in {dt / 60:.1f} min  "
            f"embeddings={tuple(result['embeddings'].shape)}"
        )
        payload = {
            **result,
            "weights_fingerprint": fp,
            "extraction_config": expected_cfg(split),
        }
        torch.save(payload, cache_path)
        print(f"[{split:5s}] Saved → {cache_path}")

    print(f"\nAll requested video splits processed (backbone={BACKBONE}).")


if __name__ == "__main__":
    main()
