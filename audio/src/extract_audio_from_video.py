"""
Audio Extraction Script
========================

Extracts audio tracks from all video files in ``data/Videos/`` and saves
them as 16 kHz mono WAV files under ``data/audio/``, mirroring the exact
directory structure expected by ``AudioDataset`` in ``utils/load_dataset.py``.

Output structure
----------------
    data/audio/
        <participant_id>/
            <question_folder>/
                <video_id>.wav

This mirrors the video path structure:
    data/Videos/
        <participant_id>/
            <question_folder>/
                <video_id>.mp4

so that ``AudioDataset`` can locate each file via:
    audio_path = audio_dir / Path(row["video_path"]).with_suffix(".wav")

Usage
-----
    cd audio/src
    python extract_audio.py

    # Dry-run (print what would be done, no files written):
    python extract_audio.py --dry-run

    # Override sample rate:
    python extract_audio.py --sample-rate 22050

    # Skip already-extracted files (default behaviour):
    python extract_audio.py --skip-existing

    # Re-extract everything even if .wav already exists:
    python extract_audio.py --no-skip-existing
"""

import argparse
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract audio from BAH dataset videos → data/audio/"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000 — wav2vec2 native)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-extract even if the .wav file already exists (default: skip)",
    )
    return parser.parse_args()


def extract_with_moviepy(video_path: Path, out_path: Path, sample_rate: int) -> bool:
    """Extract audio using moviepy (requires ffmpeg). Returns True on success."""
    try:
        from moviepy.editor import VideoFileClip

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with VideoFileClip(str(video_path)) as clip:
            if clip.audio is None:
                print(f"    [moviepy] no audio track in {video_path.name}")
                return False
            clip.audio.write_audiofile(
                str(out_path),
                fps=sample_rate,
                nbytes=2,       # 16-bit PCM
                codec="pcm_s16le",
                ffmpeg_params=["-ac", "1"],  # mono
                logger=None,
            )
        return True
    except Exception as e:
        print(f"    [moviepy] failed: {e}")
        return False


def extract_with_ffmpeg(video_path: Path, out_path: Path, sample_rate: int) -> bool:
    """Extract audio by calling ffmpeg directly. Returns True on success."""
    import subprocess

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",                        # overwrite without asking
        "-i", str(video_path),
        "-vn",                       # no video
        "-acodec", "pcm_s16le",      # 16-bit PCM WAV
        "-ar", str(sample_rate),     # sample rate
        "-ac", "1",                  # mono
        str(out_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")[-300:]
            print(f"    [ffmpeg] non-zero exit: {err}")
            return False
        return True
    except FileNotFoundError:
        print("    [ffmpeg] ffmpeg not found on PATH")
        return False
    except subprocess.TimeoutExpired:
        print("    [ffmpeg] timed out")
        return False
    except Exception as e:
        print(f"    [ffmpeg] failed: {e}")
        return False


def main():
    args = parse_args()
    SAMPLE_RATE = args.sample_rate
    DRY_RUN = args.dry_run
    SKIP_EXISTING = not args.no_skip_existing

    # ── Paths ─────────────────────────────────────────────────────────────────
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    VIDEO_DIR = REPO_ROOT / "data" / "Videos"
    AUDIO_DIR = REPO_ROOT / "data" / "audio"

    print("=" * 70)
    print("BAH Dataset — Audio Extraction")
    print("=" * 70)
    print(f"Video source : {VIDEO_DIR}")
    print(f"Audio output : {AUDIO_DIR}")
    print(f"Sample rate  : {SAMPLE_RATE} Hz")
    print(f"Skip existing: {SKIP_EXISTING}")
    print(f"Dry run      : {DRY_RUN}")
    print("=" * 70)

    if not VIDEO_DIR.exists():
        print(f"\n[ERROR] Video directory not found: {VIDEO_DIR}")
        print("  Make sure you are running from the repo root or audio/src/")
        sys.exit(1)

    # ── Discover all video files ──────────────────────────────────────────────
    video_files = sorted(VIDEO_DIR.rglob("*.mp4"))
    if not video_files:
        video_files = sorted(VIDEO_DIR.rglob("*.avi"))
    if not video_files:
        video_files = sorted(VIDEO_DIR.rglob("*.mov"))

    print(f"\n[Scan] Found {len(video_files):,} video files under {VIDEO_DIR}")
    if len(video_files) == 0:
        print("[WARN] No video files found. Nothing to do.")
        sys.exit(0)

    # ── Show a few example paths ──────────────────────────────────────────────
    print("[Scan] Example video paths:")
    for vf in video_files[:3]:
        rel = vf.relative_to(VIDEO_DIR)
        out = AUDIO_DIR / rel.with_suffix(".wav")
        print(f"  {rel}  →  {out.relative_to(REPO_ROOT)}")
    if len(video_files) > 3:
        print(f"  ... and {len(video_files) - 3} more")

    # ── Determine which files need extraction ─────────────────────────────────
    to_process = []
    skipped = 0
    for vf in video_files:
        rel = vf.relative_to(VIDEO_DIR)
        out = AUDIO_DIR / rel.with_suffix(".wav")
        if SKIP_EXISTING and out.exists():
            skipped += 1
        else:
            to_process.append((vf, out))

    print(f"\n[Plan] {len(to_process):,} to extract  |  {skipped:,} already exist (skipped)")

    if DRY_RUN:
        print("\n[DRY RUN] Would extract:")
        for vf, out in to_process[:10]:
            print(f"  {vf.relative_to(VIDEO_DIR)}  →  {out.relative_to(REPO_ROOT)}")
        if len(to_process) > 10:
            print(f"  ... and {len(to_process) - 10} more")
        print("\n[DRY RUN] No files written.")
        return

    if len(to_process) == 0:
        print("\n[INFO] All audio files already exist. Nothing to do.")
        print("       Use --no-skip-existing to force re-extraction.")
        return

    # ── Extract ───────────────────────────────────────────────────────────────
    print(f"\n[Extract] Starting extraction of {len(to_process):,} files ...")
    t_start = time.time()

    ok_count = 0
    fail_count = 0
    fail_list = []

    for i, (vf, out) in enumerate(to_process, 1):
        rel = vf.relative_to(VIDEO_DIR)
        print(f"  [{i:>4}/{len(to_process)}] {rel}", end="  ", flush=True)

        # Try extraction backends in order of preference:
        # ffmpeg first (system-level, confirmed available), then moviepy as fallback
        success = (
            extract_with_ffmpeg(vf, out, SAMPLE_RATE)
            or extract_with_moviepy(vf, out, SAMPLE_RATE)
        )

        if success and out.exists() and out.stat().st_size > 0:
            size_kb = out.stat().st_size / 1024
            print(f"✓  ({size_kb:.0f} KB)")
            ok_count += 1
        else:
            print("✗  FAILED")
            fail_count += 1
            fail_list.append(str(rel))
            # Remove empty/partial file if it was created
            if out.exists() and out.stat().st_size == 0:
                out.unlink()

    elapsed = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"Extraction complete in {elapsed:.1f}s")
    print(f"  ✓ Success : {ok_count:,}")
    print(f"  ✗ Failed  : {fail_count:,}")
    if skipped:
        print(f"  ⏭ Skipped : {skipped:,} (already existed)")
    print(f"  Output dir: {AUDIO_DIR}")
    print(f"{'=' * 70}")

    if fail_list:
        print(f"\n[WARN] Failed files ({len(fail_list)}):")
        for f in fail_list:
            print(f"  {f}")
        print(
            "\nTips to fix failures:\n"
            "  1. Install torchaudio:  pip install torchaudio\n"
            "  2. Install ffmpeg:      brew install ffmpeg  (macOS)\n"
            "                          apt install ffmpeg   (Linux)\n"
            "  3. Install moviepy:     pip install moviepy"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
