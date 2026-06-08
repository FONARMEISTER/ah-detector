"""
Audio Embedding Extraction (stage 1 — audio only)
==================================================

Pre-computes K augmented embeddings per sample using the fine-tuned audio
backbone (Wav2Vec2 / Wav2Vec2-Emotional), then saves them to::

    <cache_dir>/audio_embs_<split>.pt

Usage
-----
    FORCE_EXTRACT=1 python extract_audio.py
    AUDIO_WEIGHTS=/path/to/audio.pth python extract_audio.py
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
from augmentation import augment_waveform  # noqa: E402


def _extract_one_split(
    split: str,
    K: int,
    augment_flag: bool,
    audio_embedder,
    cfg: dict,
    aug_cfg: dict,
    batch_size: int,
    seed: int,
) -> dict:
    """Run multi-view audio extraction for one split."""
    from load_dataset import get_audio_splits

    audio_ds = get_audio_splits(
        sample_rate=cfg["audio"]["sample_rate"],
        max_length_sec=cfg["audio"]["max_length_sec"],
        feature_extractor=None,  # we augment raw waveforms manually
    )[split]

    video_ids = sorted(d["video_id"] for d in audio_ds.video_data)
    vid_to_idx = {d["video_id"]: i for i, d in enumerate(audio_ds.video_data)}
    labels = [int(audio_ds.video_data[vid_to_idx[v]]["label"]) for v in video_ids]
    N = len(video_ids)
    print(
        f"  [Split {split}] {N} audio samples  |  K={K}  |  augment={augment_flag}"
    )

    out = torch.zeros(N, K, audio_embedder.dim, dtype=torch.float32)

    sample_rate = int(cfg["audio"]["sample_rate"])
    max_samples = int(sample_rate * cfg["audio"]["max_length_sec"])

    for view in range(K):
        view_seed = seed * 9973 + view * 17 + (0 if split == "train" else 1)
        rng = np.random.default_rng(view_seed)
        apply_aug = augment_flag and (view > 0)
        print(
            f"  [Split {split}] Audio view {view + 1}/{K}  "
            f"[{'aug' if apply_aug else 'clean'}]"
        )
        for i in tqdm(
            range(0, N, batch_size),
            desc=f"    audio [{split} v{view}]",
            leave=False,
        ):
            batch_vids = video_ids[i : i + batch_size]
            wfs = []
            for v in batch_vids:
                aidx = vid_to_idx[v]
                path = audio_ds.video_data[aidx]["audio_path"]
                wf = audio_ds._load_audio(path)
                if len(wf) > max_samples:
                    wf = wf[:max_samples]
                elif len(wf) < max_samples:
                    wf = np.pad(wf, (0, max_samples - len(wf)), mode="constant")
                if apply_aug:
                    wf = augment_waveform(wf, rng, aug_cfg, sample_rate)
                wfs.append(torch.from_numpy(wf.astype(np.float32)))
            batch_wf = torch.stack(wfs, dim=0)
            emb = audio_embedder.embed_waveforms(batch_wf).detach().cpu()
            out[i : i + len(batch_vids), view, :] = emb

    return {
        "embeddings": out,
        "labels": labels,
        "video_ids": video_ids,
    }


def main():
    from embedders import AudioEmbedder

    CFG = toml.load("config.toml")
    aug_cfg = CFG.get("augmentation", {})
    K_TRAIN, K_EVAL, AUGMENT, BATCH, SEED, SPLITS = load_extraction_settings(
        CFG, modality="audio"
    )

    CACHE_DIR_RAW = CFG["fusion"].get("cache_dir", "multimodal/cache")
    CACHE_DIR = REPO_ROOT / CACHE_DIR_RAW
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = resolve_device()
    FORCE = bool(int(os.environ.get("FORCE_EXTRACT", "0")))
    WEIGHTS = Path(
        os.environ.get(
            "AUDIO_WEIGHTS", str(REPO_ROOT / CFG["audio"]["weights_path"])
        )
    )

    print("=" * 70)
    print("Audio Embedding Extraction — BAH A/H Multimodal")
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
    print(f"Model           : {CFG['audio']['model_name']}")
    print(f"Weights         : {WEIGHTS}")
    print("=" * 70)

    fp = weights_fingerprint_one(WEIGHTS)

    def expected_cfg(split: str) -> dict:
        K = K_TRAIN if split == "train" else K_EVAL
        do_aug = AUGMENT and (split == "train")
        return {
            "modality": "audio",
            "num_views": K,
            "augment": do_aug,
            "augmentation_cfg": aug_cfg if do_aug else None,
            "model_name": CFG["audio"]["model_name"],
            "sample_rate": CFG["audio"]["sample_rate"],
            "max_length_sec": CFG["audio"]["max_length_sec"],
            "seed": SEED,
        }

    splits_to_do = []
    for split in SPLITS:
        cache_path = cache_path_for(CACHE_DIR, "audio", split)
        if check_existing_cache(cache_path, fp, expected_cfg(split), FORCE):
            print(
                f"[{split:5s}] Cache OK at {cache_path} — skipping (use "
                f"FORCE_EXTRACT=1 to override)"
            )
        else:
            splits_to_do.append(split)

    if not splits_to_do:
        print("\nAll audio caches up to date — nothing to do.")
        return

    print("\n[Backbone] Loading audio encoder ...")
    audio_embedder = AudioEmbedder(
        model_name=CFG["audio"]["model_name"],
        weights_path=WEIGHTS,
        device=DEVICE,
        sample_rate=CFG["audio"]["sample_rate"],
        max_length_sec=CFG["audio"]["max_length_sec"],
    )

    for split in splits_to_do:
        K = K_TRAIN if split == "train" else K_EVAL
        cache_path = cache_path_for(CACHE_DIR, "audio", split)
        print(f"\n{'=' * 70}\n[{split:5s}] Audio → {cache_path}\n{'=' * 70}")
        t0 = time.time()
        result = _extract_one_split(
            split=split,
            K=K,
            augment_flag=AUGMENT and (split == "train"),
            audio_embedder=audio_embedder,
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

    print("\nAll requested audio splits processed.")


if __name__ == "__main__":
    main()
