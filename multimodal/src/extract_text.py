"""
Text Embedding Extraction (stage 1 — text only)
================================================

Pre-computes K augmented [CLS] embeddings per sample using the fine-tuned
text backbone, then saves them to::

    <cache_dir>/text_embs_<split>.pt

Cache schema
------------
::

    {
        "embeddings": FloatTensor (N, K_t, dim_text),
        "labels":     list[int],
        "video_ids":  list[str],
        "weights_fingerprint": {"path", "size", "mtime"},
        "extraction_config": dict (text-local settings),
    }

For ``val`` and ``test`` only K=1 view is extracted with no augmentation,
to keep evaluation deterministic.

Usage
-----
    FORCE_EXTRACT=1 python extract_text.py
    TEXT_WEIGHTS=/path/to/text.pth python extract_text.py
"""

import os
import time
from pathlib import Path

import numpy as np
import toml
import torch
from tqdm.auto import tqdm

from _extract_common import (
    REPO_ROOT,
    resolve_device,
    weights_fingerprint_one,
    check_existing_cache,
    load_extraction_settings,
    cache_path_for,
)
from augmentation import augment_text  # noqa: E402  (sys.path tweaked in _extract_common)


def _extract_one_split(
    split: str,
    K: int,
    augment_flag: bool,
    text_embedder,
    cfg: dict,
    aug_cfg: dict,
    batch_size: int,
    seed: int,
) -> dict:
    """Run multi-view text extraction for one split and return a payload dict."""
    from load_dataset import get_text_splits

    text_ds = get_text_splits(
        max_length=cfg["text"]["max_length"], tokenizer=None
    )[split]

    video_ids = sorted(text_ds.data["video_id"].astype(str).tolist())
    vid_to_idx = {
        row["video_id"]: i for i, (_, row) in enumerate(text_ds.data.iterrows())
    }
    labels = [int(text_ds.data.iloc[vid_to_idx[v]]["label"]) for v in video_ids]
    N = len(video_ids)
    print(
        f"  [Split {split}] {N} text samples  |  K={K}  |  augment={augment_flag}"
    )

    out = torch.zeros(N, K, text_embedder.dim, dtype=torch.float32)

    for view in range(K):
        view_seed = seed * 9973 + view * 17 + (0 if split == "train" else 1)
        rng = np.random.default_rng(view_seed)
        apply_aug = augment_flag and (view > 0)
        print(
            f"  [Split {split}] Text view {view + 1}/{K}  "
            f"[{'aug' if apply_aug else 'clean'}]"
        )
        for i in tqdm(
            range(0, N, batch_size),
            desc=f"    text [{split} v{view}]",
            leave=False,
        ):
            batch_vids = video_ids[i : i + batch_size]
            raw_texts = [
                text_ds.data.iloc[vid_to_idx[v]]["transcript"]
                for v in batch_vids
            ]
            if apply_aug:
                texts = [augment_text(t, rng, aug_cfg) for t in raw_texts]
            else:
                texts = raw_texts
            emb = text_embedder.embed(texts).detach().cpu()
            out[i : i + len(batch_vids), view, :] = emb

    return {
        "embeddings": out,
        "labels": labels,
        "video_ids": video_ids,
    }


def main():
    from embedders import TextEmbedder

    CFG = toml.load("config.toml")
    aug_cfg = CFG.get("augmentation", {})
    K_TRAIN, K_EVAL, AUGMENT, BATCH, SEED, SPLITS = load_extraction_settings(
        CFG, modality="text"
    )

    CACHE_DIR_RAW = CFG["fusion"].get("cache_dir", "multimodal/cache")
    CACHE_DIR = REPO_ROOT / CACHE_DIR_RAW
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = resolve_device()
    FORCE = bool(int(os.environ.get("FORCE_EXTRACT", "0")))
    WEIGHTS = Path(
        os.environ.get("TEXT_WEIGHTS", str(REPO_ROOT / CFG["text"]["weights_path"]))
    )

    print("=" * 70)
    print("Text Embedding Extraction — BAH A/H Multimodal")
    print("=" * 70)
    print(f"Device          : {DEVICE}")
    print(f"Cache dir       : {CACHE_DIR}")
    print(f"K train         : {K_TRAIN}")
    print(f"K eval          : {K_EVAL}")
    print(f"Augment train   : {AUGMENT}")
    print(f"Splits          : {SPLITS}")
    print(f"Force re-extract: {FORCE}")
    print(f"Batch size      : {BATCH}")
    print(f"Seed            : {SEED}")
    print(f"Model           : {CFG['text']['model_name']}")
    print(f"Weights         : {WEIGHTS}")
    print("=" * 70)

    fp = weights_fingerprint_one(WEIGHTS)

    def expected_cfg(split: str) -> dict:
        K = K_TRAIN if split == "train" else K_EVAL
        do_aug = AUGMENT and (split == "train")
        return {
            "modality": "text",
            "num_views": K,
            "augment": do_aug,
            "augmentation_cfg": aug_cfg if do_aug else None,
            "model_name": CFG["text"]["model_name"],
            "max_length": CFG["text"]["max_length"],
            "seed": SEED,
        }

    splits_to_do = []
    for split in SPLITS:
        cache_path = cache_path_for(CACHE_DIR, "text", split)
        if check_existing_cache(cache_path, fp, expected_cfg(split), FORCE):
            print(
                f"[{split:5s}] Cache OK at {cache_path} — skipping (use "
                f"FORCE_EXTRACT=1 to override)"
            )
        else:
            splits_to_do.append(split)

    if not splits_to_do:
        print("\nAll text caches up to date — nothing to do.")
        return

    print("\n[Backbone] Loading text encoder ...")
    text_embedder = TextEmbedder(
        model_name=CFG["text"]["model_name"],
        weights_path=WEIGHTS,
        device=DEVICE,
        max_length=CFG["text"]["max_length"],
    )

    for split in splits_to_do:
        K = K_TRAIN if split == "train" else K_EVAL
        cache_path = cache_path_for(CACHE_DIR, "text", split)
        print(f"\n{'=' * 70}\n[{split:5s}] Text → {cache_path}\n{'=' * 70}")
        t0 = time.time()
        result = _extract_one_split(
            split=split,
            K=K,
            augment_flag=AUGMENT and (split == "train"),
            text_embedder=text_embedder,
            cfg=CFG,
            aug_cfg=aug_cfg,
            batch_size=BATCH,
            seed=SEED,
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

    print("\nAll requested text splits processed.")


if __name__ == "__main__":
    main()
