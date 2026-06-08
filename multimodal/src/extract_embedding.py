"""
Multimodal Embedding Extraction — Orchestrator
================================================

Thin wrapper that runs the three per-modality extractors in sequence:

    multimodal/src/extract_text.py
    multimodal/src/extract_audio.py
    multimodal/src/extract_video.py

Each child script writes its own independent cache file
(``text_embs_<split>.pt`` / ``audio_embs_<split>.pt`` /
``video_embs_<split>.pt``).  The fusion-side ``MultimodalCachedFusionDataset``
loader (see ``utils/load_dataset.py``) reads all three, intersects by
``video_id``, and produces the aligned ``(text, audio, video, label)``
tuples expected by ``fusion_training.py``.

Usage
-----
    cd multimodal/src
    python extract_embeddings.py                 # all three modalities
    python extract_embeddings.py text            # only text
    python extract_embeddings.py text audio      # text + audio
    FORCE_EXTRACT=1 python extract_embeddings.py # rebuild caches
    TEXT_WEIGHTS=/...  AUDIO_WEIGHTS=/... VIDEO_WEIGHTS=/... \\
        python extract_embeddings.py
"""

import sys
import time


def _print_banner(title: str):
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}\n")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    selected = [m.lower() for m in argv] if argv else ["text", "audio", "video"]

    valid = {"text", "audio", "video"}
    bad = [m for m in selected if m not in valid]
    if bad:
        raise SystemExit(
            f"Unknown modality argument(s): {bad}.  "
            f"Choose from {sorted(valid)} or pass no args to run all three."
        )

    print(f"[Orchestrator] Running modalities: {selected}")

    t_start = time.time()
    for modality in selected:
        _print_banner(f"[Orchestrator] Modality: {modality.upper()}")
        if modality == "text":
            import extract_text
            extract_text.main()
        elif modality == "audio":
            import extract_audio
            extract_audio.main()
        elif modality == "video":
            import extract_video
            extract_video.main()
    dt = time.time() - t_start
    print(f"\n[Orchestrator] All requested modalities done in {dt / 60:.1f} min")


if __name__ == "__main__":
    main()
