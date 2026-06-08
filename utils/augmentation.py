"""
Shared Input-Level Augmentations
==================================

These functions are the single source of truth for raw-input augmentation
used by:
  - ``text/src/distilbert_training.py``      (per-epoch text augmentation)
  - ``audio/src/wav2vec2emotional_training.py`` (per-epoch waveform augmentation)
  - ``multimodal/src/extract_embeddings.py``  (K-view embedding extraction)

All functions are pure: they take the raw input and a NumPy ``Generator`` plus
a config dict, and return a new perturbed input.  They never mutate the input
in place (waveform copy is explicit).

Text augmentation
-----------------
- random word masking (replace with ``[MASK]`` for BERT-family models)
- random word deletion

Audio augmentation
------------------
- random gain ∈ ``[1 - audio_gain, 1 + audio_gain]``
- additive Gaussian noise at the configured SNR (``audio_snr_db``)
- random time shift up to ``audio_time_shift_sec``
- random time-masks (silence windows) of width ``audio_time_mask_frac``

Config keys
-----------
The two functions accept a flat dict.  Missing keys fall back to safe defaults
(documented in each function).  Pass ``{}`` to apply identity augmentation.
"""

from typing import List

import numpy as np


# ── Text ──────────────────────────────────────────────────────────────────────


def augment_text(text: str, rng: np.random.Generator, cfg: dict) -> str:
    """
    Word-level augmentation: mask + delete.  Operates on the raw string before
    tokenisation so the BERT-family tokenizer sees a perturbed sentence.

    Config keys (with defaults)
    ---------------------------
    text_mask_prob   : 0.15  — probability of replacing a word with the mask token
    text_drop_prob   : 0.10  — probability of dropping a word entirely
    text_mask_token  : "[MASK]"  — replacement string for masking

    Guarantees
    ----------
    - Never emits an empty string (would crash some tokenizers); falls back to
      the original text in the degenerate "everything dropped" case.
    - Stable for non-string inputs (returns input unchanged).
    """
    if not isinstance(text, str) or not text:
        return text
    mask_prob = float(cfg.get("text_mask_prob", 0.15))
    drop_prob = float(cfg.get("text_drop_prob", 0.10))
    mask_token = cfg.get("text_mask_token", "[MASK]")

    words = text.split()
    out: List[str] = []
    for w in words:
        u = rng.random()
        if u < drop_prob:
            continue  # drop this word
        if u < drop_prob + mask_prob:
            out.append(mask_token)
        else:
            out.append(w)
    # Ensure we never emit an empty string (DistilBERT would crash)
    if not out:
        return text
    return " ".join(out)


# ── Audio ─────────────────────────────────────────────────────────────────────


def augment_waveform(
    waveform: np.ndarray,
    rng: np.random.Generator,
    cfg: dict,
    sample_rate: int,
) -> np.ndarray:
    """
    Audio augmentation on a 1-D float32 waveform.

    Config keys (with defaults)
    ---------------------------
    audio_gain            : 0.2   — random gain ∈ [1-g, 1+g]
    audio_snr_db          : 25.0  — additive Gaussian noise at this SNR (None or
                                    ≥80 disables noise)
    audio_time_shift_sec  : 0.5   — random shift up to this many seconds
    audio_num_time_masks  : 2     — number of silence windows
    audio_time_mask_frac  : 0.05  — width of each silence window as a fraction
                                    of the total length

    Returns
    -------
    A new ``np.float32`` array of the same length as the input.
    """
    wf = waveform.astype(np.float32, copy=True)
    n = len(wf)

    # Random gain
    gain = float(cfg.get("audio_gain", 0.2))
    if gain > 0:
        g = 1.0 + rng.uniform(-gain, gain)
        wf = wf * g

    # Additive Gaussian noise at requested SNR
    snr_db = cfg.get("audio_snr_db", 25.0)
    if snr_db is not None and float(snr_db) < 80:
        snr_db = float(snr_db)
        sig_pow = float(np.mean(wf ** 2))
        if sig_pow > 1e-10:
            noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
            noise = rng.normal(0.0, np.sqrt(noise_pow), size=n).astype(np.float32)
            wf = wf + noise

    # Random time shift
    shift_sec = float(cfg.get("audio_time_shift_sec", 0.5))
    if shift_sec > 0:
        max_shift = int(shift_sec * sample_rate)
        if max_shift > 0:
            shift = int(rng.integers(-max_shift, max_shift + 1))
            if shift > 0:
                wf = np.concatenate([np.zeros(shift, dtype=np.float32), wf[:-shift]])
            elif shift < 0:
                wf = np.concatenate([wf[-shift:], np.zeros(-shift, dtype=np.float32)])

    # Random time-masks (zero out short windows)
    n_masks = int(cfg.get("audio_num_time_masks", 2))
    mask_frac = float(cfg.get("audio_time_mask_frac", 0.05))
    if n_masks > 0 and mask_frac > 0 and n > 0:
        mask_len = max(1, int(n * mask_frac))
        for _ in range(n_masks):
            start = int(rng.integers(0, max(1, n - mask_len)))
            wf[start : start + mask_len] = 0.0

    return wf
