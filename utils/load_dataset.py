"""
BAH Dataset Multi-Modal PyTorch Dataset Loaders
================================================

Provides PyTorch Dataset classes for the BAH (Behavioral Ambivalence/Hesitancy)
dataset with proper train/validation/test splits for all modalities.

Dataset Structure
-----------------
- data/split/train.txt, val.txt, test.txt
      Video-level splits.  Format: ``video_path,label,transcript``
- data/split-frames/train.txt, val.txt, test.txt
      Frame-level splits.  Format: ``frame_path,label``
- data/Frames/            — extracted frames from videos
- data/SegmentedFrames/   — YOLO-segmented body frames (.skip sentinels
                            mark frames where no person was detected)
- data/Videos/            — original video files (.mp4)
- data/audio/             — pre-extracted audio files (.wav / .mp3)
- data/cropped-aligned-faces/ — face crops
- data/transcription/     — YAML transcription files

Labels
------
- 0 : No A/H
- 1 : With A/H

Modalities & splitter helpers
-----------------------------
1. ``get_frame_splits()``                  — individual frames
2. ``get_segmented_frame_splits()``        — individual segmented frames
3. ``get_video_clip_splits()``             — clips of N consecutive frames
4. ``get_segmented_video_clip_splits()``   — clips of N consecutive seg. frames
5. ``get_audio_splits()``                  — audio extracted from videos
6. ``get_text_splits()``                   — transcripts
"""

import re
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
import warnings
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================

# Resolve DATA_DIR relative to this file's location (repo root / data/)
# so that load_dataset works regardless of the notebook's working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "data"
SPLIT_DIR = DATA_DIR / "split"
SPLIT_FRAMES_DIR = DATA_DIR / "split-frames"
VIDEO_DIR = DATA_DIR / "Videos"
FRAMES_DIR = DATA_DIR / "Frames"
SEGMENTED_FRAMES_DIR = DATA_DIR / "SegmentedFrames"
FACES_DIR = DATA_DIR / "cropped-aligned-faces"
TRANSCRIPTION_DIR = DATA_DIR / "transcription"
AUDIO_DIR = DATA_DIR / "audio"

CLASS_MAPPING = {"With A-H": 1, "No A-H": 0}
CLASS_NAMES = ["No A-H", "With A-H"]
NUM_CLASSES = 2


# ============================================================================
# Base loading functions
# ============================================================================


def load_video_split_file(split_file: Path) -> pd.DataFrame:
    """
    Load a video-level split file.

    Returns
    -------
    DataFrame with columns:
        video_path, label, transcript, video_id, participant_id, question
    """
    data = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) < 2:
                continue
            video_path = parts[0]
            label = int(parts[1])
            transcript = parts[2] if len(parts) > 2 else ""
            data.append(
                {
                    "video_path": video_path,
                    "label": label,
                    "transcript": transcript,
                    "video_id": Path(video_path).stem,
                    "participant_id": video_path.split("/")[1],
                    "question": video_path.split("/")[2]
                    .replace("_Question_", "Q")
                    .replace("_Video.mp4", ""),
                }
            )
    return pd.DataFrame(data)


def load_frame_split_file(split_file: Path) -> pd.DataFrame:
    """
    Load a frame-level split file.

    Returns
    -------
    DataFrame with columns: frame_path, label, frame_name, video_folder
    """
    print(split_file)
    data = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(",", 1)
            if len(parts) != 2:
                continue
            frame_path = parts[0]
            label = int(parts[1])
            data.append(
                {
                    "frame_path": frame_path,
                    "label": label,
                    "frame_name": Path(frame_path).name,
                    "video_folder": str(Path(frame_path).parent),
                }
            )
    return pd.DataFrame(data)


def _frame_number(name: str) -> int:
    """Extract numeric frame index from ``frame-42.jpg``."""
    m = re.search(r"frame-(\d+)", name)
    return int(m.group(1)) if m else 0


# ============================================================================
# 1.  FrameDataset  —  individual frames from data/Frames/
# ============================================================================


class FrameDataset(Dataset):
    """Individual-frame dataset (reads from ``data/Frames/``)."""

    def __init__(
        self,
        split: str = "train",
        frames_dir: Path = FRAMES_DIR,
        image_size: Union[int, Tuple[int, int]] = (224, 224),
        transform=None,
        subsample_every: int = 1,
    ):
        self.split = split
        self.frames_dir = Path(frames_dir)
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.transform = transform

        raw_df = load_frame_split_file(SPLIT_FRAMES_DIR / f"{split}.txt")

        subsample = subsample_every if split == "train" else 1
        if subsample > 1:
            raw_df = raw_df.sort_values(["video_folder", "frame_name"]).reset_index(
                drop=True
            )
            # Keep ALL minority-class (With A-H, label=1) frames;
            # subsample only majority-class (No A-H, label=0) frames.
            majority = (
                raw_df[raw_df["label"] == 0]
                .groupby("video_folder", group_keys=False)
                .apply(lambda g: g.iloc[::subsample])
            )
            minority = raw_df[raw_df["label"] == 1]
            raw_df = pd.concat([majority, minority]).sort_index().reset_index(drop=True)

        # Store as a plain list of dicts (not a DataFrame) so that
        # __getitem__ is fork-safe when num_workers > 0.
        self.data: List[dict] = raw_df.reset_index(drop=True).to_dict("records")
        print(f"FrameDataset [{split}]: {len(self.data):,} frames loaded")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        import cv2

        row = self.data[idx]
        label = int(row["label"])
        full_path = self.frames_dir / row["frame_path"]

        img = cv2.imread(str(full_path))
        if img is None:
            img = np.zeros((*self.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, label


def get_frame_splits(
    image_size: Union[int, Tuple[int, int]] = (224, 224),
    train_transform=None,
    eval_transform=None,
    subsample_every: int = 1,
    frames_dir: Path = FRAMES_DIR,
) -> Dict[str, FrameDataset]:
    """Return ``{'train': …, 'val': …, 'test': …}`` FrameDataset instances."""
    return {
        "train": FrameDataset(
            "train",
            frames_dir=frames_dir,
            image_size=image_size,
            transform=train_transform,
            subsample_every=subsample_every,
        ),
        "val": FrameDataset(
            "val",
            frames_dir=frames_dir,
            image_size=image_size,
            transform=eval_transform,
        ),
        "test": FrameDataset(
            "test",
            frames_dir=frames_dir,
            image_size=image_size,
            transform=eval_transform,
        ),
    }


# ============================================================================
# 2.  SegmentedFrameDataset  —  data/SegmentedFrames/
# ============================================================================


class SegmentedFrameDataset(Dataset):
    """
    Segmented-frame dataset.

    Reads from ``data/SegmentedFrames/`` using the same split-frames txt
    files.  Frames that are missing or have a ``.skip`` sentinel are silently
    dropped.
    """

    def __init__(
        self,
        split: str = "train",
        seg_dir: Path = SEGMENTED_FRAMES_DIR,
        image_size: Union[int, Tuple[int, int]] = 128,
        transform=None,
        subsample_every: int = 1,
    ):
        self.seg_dir = Path(seg_dir)
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.transform = transform
        self.split = split

        raw_df = load_frame_split_file(SPLIT_FRAMES_DIR / f"{split}.txt")

        subsample = subsample_every if split == "train" else 1
        if subsample > 1:
            raw_df = raw_df.sort_values(["video_folder", "frame_name"]).reset_index(
                drop=True
            )
            # Keep ALL minority-class (With A-H, label=1) frames;
            # subsample only majority-class (No A-H, label=0) frames.
            majority = (
                raw_df[raw_df["label"] == 0]
                .groupby("video_folder", group_keys=False)
                .apply(lambda g: g.iloc[::subsample])
            )
            minority = raw_df[raw_df["label"] == 1]
            raw_df = pd.concat([majority, minority]).sort_index().reset_index(drop=True)

        valid_rows: List[dict] = []
        for _, row in tqdm(
            raw_df.iterrows(),
            total=len(raw_df),
            desc=f"SegFrames [{split}] filtering",
            leave=False,
        ):
            seg_path = self.seg_dir / row["frame_path"]
            skip_path = seg_path.with_suffix(".skip")
            if seg_path.exists() and not skip_path.exists():
                valid_rows.append(
                    {
                        "frame_path": row["frame_path"],
                        "label": row["label"],
                        "seg_path": str(seg_path),
                    }
                )

        # Store as a plain list of dicts (not a DataFrame) so that
        # __getitem__ is fork-safe when num_workers > 0.
        self.data: List[dict] = valid_rows

        total = len(raw_df)
        kept = len(self.data)
        sub_note = f", subsampled 1/{subsample}" if subsample > 1 else ""
        pct = kept / total * 100 if total > 0 else 0
        print(
            f"SegmentedFrameDataset [{split}]{sub_note}: "
            f"{kept:,} / {total:,} frames ({pct:.1f}% coverage)"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        import cv2

        row = self.data[idx]
        label = int(row["label"])

        img = cv2.imread(row["seg_path"])
        if img is None:
            img = np.zeros((*self.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, label


def get_segmented_frame_splits(
    image_size: Union[int, Tuple[int, int]] = 128,
    train_transform=None,
    eval_transform=None,
    subsample_every: int = 1,
    seg_dir: Path = SEGMENTED_FRAMES_DIR,
) -> Dict[str, SegmentedFrameDataset]:
    """Return ``{'train': …, 'val': …, 'test': …}`` SegmentedFrameDataset."""
    print(seg_dir)
    return {
        "train": SegmentedFrameDataset(
            "train",
            seg_dir=seg_dir,
            image_size=image_size,
            transform=train_transform,
            subsample_every=subsample_every,
        ),
        "val": SegmentedFrameDataset(
            "val",
            seg_dir=seg_dir,
            image_size=image_size,
            transform=eval_transform,
        ),
        "test": SegmentedFrameDataset(
            "test",
            seg_dir=seg_dir,
            image_size=image_size,
            transform=eval_transform,
        ),
    }


# ============================================================================
# 3.  VideoClipDataset  —  clips of consecutive frames from data/Frames/
# ============================================================================


class VideoClipDataset(Dataset):
    """
    Groups consecutive frames into fixed-length clips.

    Each sample is ``(clip_tensor, label)`` where ``clip_tensor`` has shape
    ``(clip_len, C, H, W)``.  Videos are split into non-overlapping (or
    overlapping, via ``clip_stride``) windows.  The last partial clip is
    zero-padded when ``pad_last=True``, otherwise dropped.
    """

    def __init__(
        self,
        split: str = "train",
        frames_dir: Path = FRAMES_DIR,
        image_size: Union[int, Tuple[int, int]] = (224, 224),
        clip_len: int = 16,
        clip_stride: int = 0,
        pad_last: bool = True,
        transform=None,
        label_agg: str = "video",
    ):
        self.split = split
        self.frames_dir = Path(frames_dir)
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.clip_len = clip_len
        self.stride = clip_stride if clip_stride > 0 else clip_len
        self.pad_last = pad_last
        self.transform = transform
        self.label_agg = label_agg

        raw_df = load_frame_split_file(SPLIT_FRAMES_DIR / f"{split}.txt")
        raw_df["_fnum"] = raw_df["frame_name"].apply(_frame_number)
        raw_df = raw_df.sort_values(["video_folder", "_fnum"]).reset_index(drop=True)

        self.clips: List[Tuple[str, int, List[str]]] = []
        groups = raw_df.groupby("video_folder", sort=False)
        for vfolder, grp in tqdm(
            groups,
            total=groups.ngroups,
            desc=f"Clips [{split}] windowing",
            leave=False,
        ):
            paths = grp["frame_path"].tolist()
            labels = grp["label"].tolist()
            n = len(paths)
            start = 0
            while start < n:
                end = start + clip_len
                clip_paths = paths[start:end]
                clip_labels = labels[start:end]
                if len(clip_paths) < clip_len and not pad_last:
                    break
                if label_agg == "any":
                    label = int(max(clip_labels))
                else:
                    label = int(clip_labels[0])
                self.clips.append((str(vfolder), label, clip_paths))
                start += self.stride

        print(
            f"VideoClipDataset [{split}]: {len(self.clips):,} clips "
            f"(clip_len={clip_len}, stride={self.stride})"
        )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        import cv2

        _, label, clip_paths = self.clips[idx]
        frames: List[torch.Tensor] = []

        for fp in clip_paths:
            full_path = self.frames_dir / fp
            img = cv2.imread(str(full_path))
            if img is None:
                img = np.zeros((*self.image_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.image_size[1], self.image_size[0]))

            if self.transform:
                img = self.transform(img)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            frames.append(img)

        # zero-pad
        while len(frames) < self.clip_len:
            frames.append(torch.zeros(3, self.image_size[0], self.image_size[1]))

        return torch.stack(frames[: self.clip_len]), label


def get_video_clip_splits(
    image_size: Union[int, Tuple[int, int]] = (224, 224),
    clip_len: int = 16,
    clip_stride: int = 0,
    pad_last: bool = True,
    train_transform=None,
    eval_transform=None,
    frames_dir: Path = FRAMES_DIR,
    label_agg: str = "video",
) -> Dict[str, VideoClipDataset]:
    """Return ``{'train': …, 'val': …, 'test': …}`` VideoClipDataset.

    Parameters
    ----------
    label_agg : str
        ``"video"`` — use the video-level label (first frame's label).
        ``"any"``   — clip is positive (1) if *any* constituent frame is 1.
    """
    kw = dict(
        frames_dir=frames_dir,
        image_size=image_size,
        clip_len=clip_len,
        clip_stride=clip_stride,
        pad_last=pad_last,
        label_agg=label_agg,
    )
    return {
        "train": VideoClipDataset("train", transform=train_transform, **kw),
        "val": VideoClipDataset("val", transform=eval_transform, **kw),
        "test": VideoClipDataset("test", transform=eval_transform, **kw),
    }


# ============================================================================
# 4.  SegmentedVideoClipDataset  —  clips from data/SegmentedFrames/
# ============================================================================


class SegmentedVideoClipDataset(Dataset):
    """
    Same idea as :class:`VideoClipDataset` but reads from
    ``data/SegmentedFrames/``.

    Frames that were skipped during segmentation (``.skip`` sentinel) are
    excluded *before* clip windowing, so clips contain only valid segmented
    frames packed together.
    """

    def __init__(
        self,
        split: str = "train",
        seg_dir: Path = SEGMENTED_FRAMES_DIR,
        image_size: Union[int, Tuple[int, int]] = (224, 224),
        clip_len: int = 16,
        clip_stride: int = 0,
        pad_last: bool = True,
        transform=None,
        label_agg: str = "video",
    ):
        self.split = split
        self.seg_dir = Path(seg_dir)
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.clip_len = clip_len
        self.stride = clip_stride if clip_stride > 0 else clip_len
        self.pad_last = pad_last
        self.transform = transform
        self.label_agg = label_agg

        raw_df = load_frame_split_file(SPLIT_FRAMES_DIR / f"{split}.txt")
        raw_df["_fnum"] = raw_df["frame_name"].apply(_frame_number)
        raw_df = raw_df.sort_values(["video_folder", "_fnum"]).reset_index(drop=True)

        # Filter out skipped / missing segmented frames
        def _seg_ok(fp: str) -> bool:
            p = self.seg_dir / fp
            return p.exists() and not p.with_suffix(".skip").exists()

        tqdm.pandas(desc=f"SegClips [{split}] filtering", leave=False)
        raw_df = raw_df[raw_df["frame_path"].progress_apply(_seg_ok)].reset_index(
            drop=True
        )

        self.clips: List[Tuple[str, int, List[str]]] = []
        groups = raw_df.groupby("video_folder", sort=False)
        for vfolder, grp in tqdm(
            groups,
            total=groups.ngroups,
            desc=f"SegClips [{split}] windowing",
            leave=False,
        ):
            paths = grp["frame_path"].tolist()
            labels = grp["label"].tolist()
            n = len(paths)
            start = 0
            while start < n:
                end = start + clip_len
                clip_paths = paths[start:end]
                clip_labels = labels[start:end]
                if len(clip_paths) < clip_len and not pad_last:
                    break
                if label_agg == "any":
                    label = int(max(clip_labels))
                else:
                    label = int(clip_labels[0])
                self.clips.append((str(vfolder), label, clip_paths))
                start += self.stride

        print(
            f"SegmentedVideoClipDataset [{split}]: {len(self.clips):,} clips "
            f"(clip_len={clip_len}, stride={self.stride})"
        )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        import cv2

        _, label, clip_paths = self.clips[idx]
        frames: List[torch.Tensor] = []

        for fp in clip_paths:
            full_path = self.seg_dir / fp
            img = cv2.imread(str(full_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))

            if self.transform:
                img = self.transform(img)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            frames.append(img)

        while len(frames) < self.clip_len:
            frames.append(torch.zeros(3, self.image_size[0], self.image_size[1]))

        return torch.stack(frames[: self.clip_len]), label


def get_segmented_video_clip_splits(
    image_size: Union[int, Tuple[int, int]] = (224, 224),
    clip_len: int = 16,
    clip_stride: int = 0,
    pad_last: bool = True,
    train_transform=None,
    eval_transform=None,
    seg_dir: Path = SEGMENTED_FRAMES_DIR,
    label_agg: str = "video",
) -> Dict[str, SegmentedVideoClipDataset]:
    """Return ``{'train': …, 'val': …, 'test': …}`` SegmentedVideoClipDataset.

    Parameters
    ----------
    label_agg : str
        ``"video"`` — use the video-level label (first frame's label).
        ``"any"``   — clip is positive (1) if *any* constituent frame is 1.
    """
    kw = dict(
        seg_dir=seg_dir,
        image_size=image_size,
        clip_len=clip_len,
        clip_stride=clip_stride,
        pad_last=pad_last,
        label_agg=label_agg,
    )
    return {
        "train": SegmentedVideoClipDataset("train", transform=train_transform, **kw),
        "val": SegmentedVideoClipDataset("val", transform=eval_transform, **kw),
        "test": SegmentedVideoClipDataset("test", transform=eval_transform, **kw),
    }


# ============================================================================
# 5.  AudioDataset  —  audio extracted from video files
# ============================================================================


class AudioDataset(Dataset):
    """
    Audio dataset.

    Loads pre-extracted audio files from ``data/audio/``.  Run
    ``audio/src/extract_audio.py`` first to populate that directory.

    Pre-extracted audio files are expected at:
        ``data/audio/<participant_id>/<question>/<video_id>.wav``
    mirroring the video path structure but with a ``.wav`` extension.

    Returns a 1-D waveform tensor padded/truncated to ``max_length_sec``
    seconds.  If a ``feature_extractor`` (e.g. HuggingFace
    Wav2Vec2FeatureExtractor) is provided, ``__getitem__`` returns its output
    dict instead of a raw waveform.
    """

    def __init__(
        self,
        split: str = "train",
        audio_dir: Path = AUDIO_DIR,
        sample_rate: int = 16000,
        max_length_sec: float = 30.0,
        feature_extractor=None,
        augment: bool = False,
        aug_cfg: Optional[dict] = None,
        seed: int = 42,
    ):
        self.split = split
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_length_sec = max_length_sec
        self.max_samples = int(sample_rate * max_length_sec)
        self.feature_extractor = feature_extractor
        self.augment = bool(augment)
        self.aug_cfg = dict(aug_cfg) if aug_cfg else {}
        # Each worker gets its own RNG seeded from base seed + worker_id via
        # ``worker_init_fn``; if not set we fall back to this stream.
        self._rng = np.random.default_rng(seed)

        split_file = SPLIT_DIR / f"{split}.txt"
        self.data = load_video_split_file(split_file)

        self.video_data: List[dict] = []
        missing: List[str] = []
        for _, row in tqdm(
            self.data.iterrows(),
            total=len(self.data),
            desc=f"Audio [{split}] scanning",
            leave=False,
        ):
            # video_path is e.g. "Videos/<participant>/<question>/<stem>.mp4"
            # audio files mirror the same structure but with .wav extension
            rel = Path(row["video_path"]).with_suffix(".wav")
            audio_path = self.audio_dir / rel
            if audio_path.exists():
                self.video_data.append(
                    {
                        "audio_path": str(audio_path),
                        "label": row["label"],
                        "video_id": row["video_id"],
                        "source": "audio_dir",
                    }
                )
            else:
                missing.append(str(audio_path))

        if missing:
            warnings.warn(
                f"AudioDataset [{split}]: {len(missing)} audio file(s) not found "
                f"under {self.audio_dir}. Run audio/src/extract_audio.py first.\n"
                f"  First missing: {missing[0]}",
                UserWarning,
                stacklevel=2,
            )

        print(
            f"AudioDataset [{split}]: {len(self.video_data):,} / {len(self.data):,} samples loaded "
            f"(sr={sample_rate}, max={max_length_sec}s)"
        )

    def __len__(self) -> int:
        return len(self.video_data)

    def _load_audio(self, path: str) -> np.ndarray:
        """Load audio waveform from a .wav/.mp3 or .mp4 file."""
        # Try torchaudio first
        try:
            import torchaudio

            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            return waveform.numpy()
        except Exception:
            pass

        # Fallback: librosa
        try:
            import librosa

            audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
            return audio
        except Exception:
            pass

        return np.zeros(self.max_samples, dtype=np.float32)

    def __getitem__(self, idx: int):
        info = self.video_data[idx]
        label = info["label"]

        audio = self._load_audio(info["audio_path"])

        # Pad or truncate
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)), mode="constant")

        # ── Raw-waveform augmentation (train-only) ────────────────────────────
        # Applied BEFORE the feature extractor so the SpecAugment-style
        # perturbations propagate through normalisation.
        if self.augment:
            from augmentation import augment_waveform
            audio = augment_waveform(audio, self._rng, self.aug_cfg, self.sample_rate)

        if self.feature_extractor is not None:
            features = self.feature_extractor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_samples,
                truncation=True,
            )
            # Squeeze batch dim added by return_tensors="pt"
            return {k: v.squeeze(0) for k, v in features.items()}, label

        return torch.from_numpy(audio).float(), label


def get_audio_splits(
    sample_rate: int = 16000,
    max_length_sec: float = 30.0,
    feature_extractor=None,
    audio_dir: Path = AUDIO_DIR,
    augment_train: bool = False,
    aug_cfg: Optional[dict] = None,
    seed: int = 42,
) -> Dict[str, AudioDataset]:
    """Return ``{'train': …, 'val': …, 'test': …}`` AudioDataset.

    Requires pre-extracted ``.wav`` files under ``audio_dir`` (defaults to
    ``data/audio/``).  Run ``audio/src/extract_audio.py`` first.

    If ``augment_train`` is True, the training split applies waveform
    augmentation (see ``utils/augmentation.augment_waveform``) on every
    ``__getitem__`` — val/test are never augmented.
    """
    kw = dict(
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        max_length_sec=max_length_sec,
        feature_extractor=feature_extractor,
    )
    return {
        "train": AudioDataset(
            "train", augment=augment_train, aug_cfg=aug_cfg, seed=seed, **kw
        ),
        "val": AudioDataset("val", augment=False, **kw),
        "test": AudioDataset("test", augment=False, **kw),
    }


# ============================================================================
# 6.  TextDataset  —  transcripts from split files
# ============================================================================


class TextDataset(Dataset):
    """
    Text dataset.

    Returns transcripts with labels.  If a ``tokenizer`` (e.g. HuggingFace
    AutoTokenizer) is provided, ``__getitem__`` returns tokenised tensors;
    otherwise it returns the raw text string.
    """

    def __init__(
        self,
        split: str = "train",
        max_length: int = 512,
        tokenizer=None,
        augment: bool = False,
        aug_cfg: Optional[dict] = None,
        seed: int = 42,
    ):
        self.split = split
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.augment = bool(augment)
        self.aug_cfg = dict(aug_cfg) if aug_cfg else {}
        # Default mask token if user didn't override and a tokenizer is known.
        if (
            self.augment
            and "text_mask_token" not in self.aug_cfg
            and tokenizer is not None
            and getattr(tokenizer, "mask_token", None)
        ):
            self.aug_cfg["text_mask_token"] = tokenizer.mask_token
        self._rng = np.random.default_rng(seed)

        split_file = SPLIT_DIR / f"{split}.txt"
        self.data = load_video_split_file(split_file)

        print(f"TextDataset [{split}]: {len(self.data):,} samples loaded")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        text = row["transcript"]
        label = int(row["label"])

        # ── Word-level text augmentation (train-only) ─────────────────────────
        # Applied BEFORE tokenisation so the tokenizer handles ``[MASK]``.
        if self.augment:
            from augmentation import augment_text
            text = augment_text(text, self._rng, self.aug_cfg)

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return (
                {k: v.squeeze(0) for k, v in encoded.items()},
                label,
            )

        return text, label


def get_text_splits(
    max_length: int = 512,
    tokenizer=None,
    augment_train: bool = False,
    aug_cfg: Optional[dict] = None,
    seed: int = 42,
) -> Dict[str, TextDataset]:
    """Return ``{'train': …, 'val': …, 'test': …}`` TextDataset.

    If ``augment_train`` is True, the training split applies word-level
    augmentation (see ``utils/augmentation.augment_text``) on every
    ``__getitem__`` — val/test are never augmented.
    """
    return {
        "train": TextDataset(
            "train",
            max_length=max_length,
            tokenizer=tokenizer,
            augment=augment_train,
            aug_cfg=aug_cfg,
            seed=seed,
        ),
        "val": TextDataset("val", max_length=max_length, tokenizer=tokenizer),
        "test": TextDataset("test", max_length=max_length, tokenizer=tokenizer),
    }


# ============================================================================
# 7.  MultimodalFusionDataset  —  aligned text + audio + video embeddings
# ============================================================================


class MultimodalFusionDataset(Dataset):
    """
    Aligns text, audio, and video modalities by ``video_id`` and pre-computes
    all embeddings offline (before training begins).

    Pre-computing embeddings is much faster than re-running the frozen
    backbones every epoch.  All embeddings are stored as CPU tensors.

    Embedding Cache
    ---------------
    If ``cache_dir`` is provided, pre-computed embeddings are saved to
    ``<cache_dir>/fusion_embs_<split>.pt`` on first run and loaded from disk
    on subsequent runs.  This avoids the 6-hour recomputation cost.

    Alignment
    ---------
    Text and audio datasets are video-level (one sample per video).
    Video (Swin) is clip-level (multiple clips per video).  This dataset
    aligns all three modalities by ``video_id``:

      1. For each video in the split, one text sample and one audio sample exist.
      2. Multiple Swin clips exist per video; their embeddings are mean-pooled to
         produce a single video-level visual embedding.
      3. The three embeddings are stored separately (text_embs, audio_embs,
         video_embs) for use by the fusion model.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    text_embedder : TextEmbedder
        Object with ``.embed(texts: list) -> Tensor`` and ``.tokenizer``.
    audio_embedder : AudioEmbedder
        Object with ``.embed_from_encoded(encoded: dict) -> Tensor``
        and ``.feature_extractor``.
    video_embedder : VideoEmbedder
        Object with ``.embed(pixel_values: Tensor) -> Tensor``.
    video_processor : AutoImageProcessor
        Used to pre-process raw frames before passing to VideoEmbedder.
    cfg : dict
        Full multimodal config dict (from ``toml.load``).  Expected keys:
        ``cfg["text"]["max_length"]``, ``cfg["audio"]["sample_rate"]``,
        ``cfg["audio"]["max_length_sec"]``, ``cfg["video"]["image_size"]``,
        ``cfg["video"]["clip_len"]``, ``cfg["video"]["clip_stride"]``,
        ``cfg["video"]["label_agg"]``.
    device : torch.device
    batch_size_embed : int
        Batch size used during embedding extraction (not training).
    cache_dir : Path or str or None
        Directory to save/load pre-computed embeddings.  If None, embeddings
        are always recomputed.  If set, a file
        ``<cache_dir>/fusion_embs_<split>.pt`` is created on first run and
        reused on subsequent runs.
    """

    def __init__(
        self,
        split: str,
        text_embedder,
        audio_embedder,
        video_embedder,
        video_processor,
        cfg: dict,
        device,
        batch_size_embed: int = 8,
        cache_dir=None,
    ):
        from torch.utils.data import DataLoader

        self.split = split
        self.device = device

        # ── Try loading from cache ────────────────────────────────────────────
        # Fingerprint backbone weight files (path + mtime + size) so we can
        # detect when a backbone has been retrained and invalidate stale
        # caches automatically — otherwise users silently reuse old embeddings.
        def _fingerprint_weights() -> dict:
            fp = {}
            for name in ("text", "audio", "video"):
                wp = cfg.get(name, {}).get("weights_path", "")
                p = Path(wp) if wp else None
                if p is not None and p.exists():
                    st = p.stat()
                    fp[name] = {
                        "path": str(p),
                        "size": st.st_size,
                        "mtime": int(st.st_mtime),
                    }
                else:
                    fp[name] = {"path": str(wp), "size": None, "mtime": None}
            return fp

        cache_path = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"fusion_embs_{split}.pt"

        current_fp = _fingerprint_weights()

        if cache_path is not None and cache_path.exists():
            print(
                f"\n[FusionDataset:{split}] Loading cached embeddings from "
                f"{cache_path} ..."
            )
            # weights_only=False is required because the cache contains Python
            # lists (labels, video_ids, fingerprint dict) in addition to tensors.
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            cached_fp = cached.get("weights_fingerprint", None)
            if cached_fp != current_fp:
                print(
                    f"[FusionDataset:{split}] Cache fingerprint mismatch — "
                    f"backbone weights changed.  Recomputing embeddings.\n"
                    f"  cached  : {cached_fp}\n"
                    f"  current : {current_fp}"
                )
            else:
                self.text_embs  = cached["text"]       # (N, text_dim)
                self.audio_embs = cached["audio"]      # (N, audio_dim)
                self.video_embs = cached["video"]      # (N, video_dim)
                self.labels     = cached["labels"]     # list[int]
                self.video_ids  = cached["video_ids"]  # list[str]
                print(
                    f"[FusionDataset:{split}] Loaded {len(self.video_ids)} videos "
                    f"from cache.  "
                    f"text={self.text_embs.shape}  "
                    f"audio={self.audio_embs.shape}  "
                    f"video={self.video_embs.shape}"
                )
                return  # skip all embedding computation

        # Stash fingerprint so the writer below can persist it.
        self._weights_fingerprint = current_fp

        # ── Load raw splits ───────────────────────────────────────────────────
        print(f"\n[FusionDataset:{split}] Loading raw modality splits ...")

        # Text (no tokenizer — caller embeds with text_embedder directly)
        text_splits = get_text_splits(
            max_length=cfg["text"]["max_length"],
            tokenizer=None,
        )
        text_ds = text_splits[split]

        # Audio (with feature extractor for waveform pre-processing)
        audio_splits = get_audio_splits(
            sample_rate=cfg["audio"]["sample_rate"],
            max_length_sec=cfg["audio"]["max_length_sec"],
            feature_extractor=audio_embedder.feature_extractor,
        )
        audio_ds = audio_splits[split]

        # Video clips
        video_splits = get_segmented_video_clip_splits(
            image_size=cfg["video"]["image_size"],
            clip_len=cfg["video"]["clip_len"],
            clip_stride=cfg["video"]["clip_stride"],
            pad_last=True,
            label_agg=cfg["video"]["label_agg"],
        )
        video_ds = video_splits[split]

        # ── Build video_id → index maps ───────────────────────────────────────
        # Text: video_id column in text_ds.data
        text_vid_to_idx = {
            row["video_id"]: i
            for i, (_, row) in enumerate(text_ds.data.iterrows())
        }

        # Audio: video_id in audio_ds.video_data[i]["video_id"]
        audio_vid_to_idx = {
            d["video_id"]: i
            for i, d in enumerate(audio_ds.video_data)
        }

        # Video: clips[i][0] is the full relative folder path, e.g.
        #   "Videos/P001/Q1_Question_1/P001_Q1_Video"
        # The last component (Path(vfolder).name) matches the text/audio
        # video_id which is Path(video_path).stem, e.g. "P001_Q1_Video".
        # We key by the stem and store the full vfolder for later lookup.
        video_vid_to_clip_idxs: Dict[str, List[int]] = {}
        video_vid_to_vfolder: Dict[str, str] = {}  # stem → full vfolder path
        for i, (vfolder, _label, _paths) in enumerate(video_ds.clips):
            # vfolder is the parent dir of frames, e.g.
            #   "Videos/P001/Q1/82563_Q1_Video.mp4"
            # Strip any extension so it matches text/audio video_id which uses
            # Path(video_path).stem (no extension).
            vid_stem = Path(vfolder).stem  # removes .mp4 if present
            video_vid_to_clip_idxs.setdefault(vid_stem, []).append(i)
            video_vid_to_vfolder[vid_stem] = vfolder

        # ── Find common video_ids across all three modalities ─────────────────
        common_ids = (
            set(text_vid_to_idx.keys())
            & set(audio_vid_to_idx.keys())
            & set(video_vid_to_clip_idxs.keys())
        )
        print(
            f"[FusionDataset:{split}] Modality coverage: "
            f"text={len(text_vid_to_idx)}, "
            f"audio={len(audio_vid_to_idx)}, "
            f"video={len(video_vid_to_clip_idxs)}  →  "
            f"common={len(common_ids)}"
        )
        if len(common_ids) == 0:
            # Emit diagnostic samples to help debug future mismatches
            text_sample  = sorted(text_vid_to_idx.keys())[:3]
            audio_sample = sorted(audio_vid_to_idx.keys())[:3]
            video_sample = sorted(video_vid_to_clip_idxs.keys())[:3]
            raise RuntimeError(
                f"No common video_ids found across all three modalities for "
                f"split={split}.\n"
                f"  text  sample ids : {text_sample}\n"
                f"  audio sample ids : {audio_sample}\n"
                f"  video sample ids : {video_sample}\n"
                "Check that audio files are extracted (run extract_audio.py) "
                "and SegmentedFrames exist."
            )

        # Stable ordering: sort by video_id for reproducibility
        self.video_ids: List[str] = sorted(common_ids)

        # ── Pre-compute labels ────────────────────────────────────────────────
        # Use text split as the authoritative label source (video-level)
        self.labels: List[int] = [
            int(text_ds.data.iloc[text_vid_to_idx[vid]]["label"])
            for vid in self.video_ids
        ]

        # ── Pre-compute text embeddings ───────────────────────────────────────
        print(f"[FusionDataset:{split}] Extracting text embeddings ...")
        text_embs = []
        for i in range(0, len(self.video_ids), batch_size_embed):
            batch_vids = self.video_ids[i : i + batch_size_embed]
            texts = [
                text_ds.data.iloc[text_vid_to_idx[vid]]["transcript"]
                for vid in batch_vids
            ]
            emb = text_embedder.embed(texts).cpu()  # (B, 768)
            text_embs.append(emb)
        self.text_embs: torch.Tensor = torch.cat(text_embs, dim=0)  # (N, 768)
        print(f"  text_embs shape: {self.text_embs.shape}")

        # ── Pre-compute audio embeddings ──────────────────────────────────────
        print(f"[FusionDataset:{split}] Extracting audio embeddings ...")

        def _audio_collate(batch):
            encoded_list, _ = zip(*batch)
            batch_enc = {
                k: torch.stack([e[k] for e in encoded_list])
                for k in encoded_list[0].keys()
            }
            return batch_enc

        audio_indices = [audio_vid_to_idx[vid] for vid in self.video_ids]
        from torch.utils.data import Subset
        audio_subset = Subset(audio_ds, audio_indices)
        audio_loader = DataLoader(
            audio_subset,
            batch_size=batch_size_embed,
            shuffle=False,
            num_workers=0,
            collate_fn=_audio_collate,
        )
        audio_embs = []
        for encoded in tqdm(audio_loader, desc=f"  audio [{split}]", leave=False):
            encoded = {k: v.to(device) for k, v in encoded.items()}
            emb = audio_embedder.embed_from_encoded(encoded).cpu()  # (B, 256)
            audio_embs.append(emb)
        self.audio_embs: torch.Tensor = torch.cat(audio_embs, dim=0)  # (N, 256)
        print(f"  audio_embs shape: {self.audio_embs.shape}")

        # ── Pre-compute video embeddings (mean-pool clips per video) ──────────
        print(f"[FusionDataset:{split}] Extracting video embeddings ...")

        def _video_collate(batch):
            clips, labels = zip(*batch)
            B = len(clips)
            T = clips[0].shape[0]
            all_frames = []
            for clip in clips:
                np_clip = (clip.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
                for t in range(np_clip.shape[0]):
                    all_frames.append(np_clip[t])
            processed = video_processor(all_frames, return_tensors="pt")
            pv = processed["pixel_values"]
            C_out, H_out, W_out = pv.shape[1:]
            pv = pv.view(B, T, C_out, H_out, W_out)
            return pv, torch.tensor(labels, dtype=torch.long)

        video_embs = []
        for vid in tqdm(self.video_ids, desc=f"  video [{split}]", leave=False):
            clip_idxs = video_vid_to_clip_idxs[vid]
            clip_subset = Subset(video_ds, clip_idxs)
            clip_loader = DataLoader(
                clip_subset,
                batch_size=batch_size_embed,
                shuffle=False,
                num_workers=0,
                collate_fn=_video_collate,
            )
            clip_embs = []
            for pv, _ in clip_loader:
                pv = pv.to(device)
                emb = video_embedder.embed(pv).cpu()  # (B_clips, 768)
                clip_embs.append(emb)
            # Mean-pool all clips for this video → (768,)
            vid_emb = torch.cat(clip_embs, dim=0).mean(dim=0)
            video_embs.append(vid_emb)
        self.video_embs: torch.Tensor = torch.stack(video_embs, dim=0)  # (N, 768)
        print(f"  video_embs shape: {self.video_embs.shape}")

        print(
            f"[FusionDataset:{split}] Embeddings ready: "
            f"text={self.text_embs.shape}  "
            f"audio={self.audio_embs.shape}  "
            f"video={self.video_embs.shape}"
        )

        # ── Save to cache ─────────────────────────────────────────────────────
        if cache_path is not None:
            torch.save(
                {
                    "text":      self.text_embs,
                    "audio":     self.audio_embs,
                    "video":     self.video_embs,
                    "labels":    self.labels,
                    "video_ids": self.video_ids,
                    # Persist a fingerprint of the backbone weight files so a
                    # subsequent run can detect when they have changed and
                    # automatically invalidate this cache.
                    "weights_fingerprint": getattr(
                        self, "_weights_fingerprint", None
                    ),
                },
                cache_path,
            )
            print(f"[FusionDataset:{split}] Embeddings cached → {cache_path}")

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return (text_emb, audio_emb, video_emb, label) for sample idx."""
        return (
            self.text_embs[idx],
            self.audio_embs[idx],
            self.video_embs[idx],
            self.labels[idx],
        )


def get_multimodal_fusion_splits(
    text_embedder,
    audio_embedder,
    video_embedder,
    video_processor,
    cfg: dict,
    device,
    batch_size_embed: int = 8,
    cache_dir=None,
) -> Dict[str, "MultimodalFusionDataset"]:
    """Return ``{'train': …, 'val': …, 'test': …}`` MultimodalFusionDataset.

    All three backbone embedders must be pre-loaded and frozen before calling
    this function.  Embeddings are pre-computed once for all splits.

    Parameters
    ----------
    text_embedder : TextEmbedder
    audio_embedder : AudioEmbedder
    video_embedder : VideoEmbedder
    video_processor : AutoImageProcessor
    cfg : dict
        Full multimodal config dict (from ``toml.load``).
    device : torch.device
    batch_size_embed : int
        Batch size used during embedding extraction.
    cache_dir : Path or str or None
        If provided, embeddings are saved/loaded from this directory.
        Pass ``cfg["fusion"]["cache_dir"]`` to enable caching.
    """
    kw = dict(
        text_embedder=text_embedder,
        audio_embedder=audio_embedder,
        video_embedder=video_embedder,
        video_processor=video_processor,
        cfg=cfg,
        device=device,
        batch_size_embed=batch_size_embed,
        cache_dir=cache_dir,
    )
    return {
        "train": MultimodalFusionDataset("train", **kw),
        "val":   MultimodalFusionDataset("val",   **kw),
        "test":  MultimodalFusionDataset("test",  **kw),
    }


# ============================================================================
# 8.  MultimodalCachedFusionDataset  —  reads multi-view cache from disk
# ============================================================================


def _load_per_modality_cache(cache_path: Path, modality: str) -> dict:
    """Load one per-modality cache file and validate its schema."""
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{modality} embedding cache not found at {cache_path}.  "
            f"Run multimodal/src/extract_{modality}.py first "
            f"(or extract_embeddings.py for all modalities)."
        )
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    for key in ("embeddings", "labels", "video_ids"):
        if key not in cached:
            raise KeyError(
                f"{modality} cache at {cache_path} missing required key '{key}'. "
                f"Re-run extract_{modality}.py."
            )
    emb = cached["embeddings"]
    if not (isinstance(emb, torch.Tensor) and emb.ndim == 3):
        raise ValueError(
            f"{modality} cache at {cache_path}: 'embeddings' must be (N, K, dim) "
            f"but is {tuple(getattr(emb, 'shape', ()))}."
        )
    if emb.shape[0] != len(cached["labels"]) or emb.shape[0] != len(cached["video_ids"]):
        raise ValueError(
            f"{modality} cache at {cache_path}: N mismatch — "
            f"emb.shape[0]={emb.shape[0]}, labels={len(cached['labels'])}, "
            f"video_ids={len(cached['video_ids'])}."
        )
    return cached


class MultimodalCachedFusionDataset(Dataset):
    """
    Reads three independent per-modality embedding caches (text / audio /
    video) produced by ``multimodal/src/extract_{text,audio,video}.py`` and
    serves aligned (text, audio, video, label) tuples per sample.

    Why per-modality caches
    -----------------------
    Video extraction is the slowest step in the pipeline; isolating each
    modality lets you re-run text or audio extraction (seconds-to-minutes)
    without touching the (long) video cache.

    Per-modality cache layout (one ``.pt`` file per modality per split)::

        {
            "embeddings": FloatTensor (N, K, dim),
            "labels":     list[int],
            "video_ids":  list[str],
            "weights_fingerprint": dict,
            "extraction_config":   dict,
        }

    Alignment
    ---------
    The three caches need not have identical ``video_ids`` lists (a modality
    might miss a few samples).  At load time we intersect by ``video_id`` and
    reorder all three to a single common index list.  Labels are taken from
    the text cache (they must agree across modalities; a sanity check warns
    on mismatch).

    View selection
    --------------
    For each ``__getitem__(idx)``:
      - In **training mode** (``random_view=True``): a fresh random view is
        drawn independently per modality, so the head can see up to
        ``K_t * K_a * K_v`` distinct embedding combinations per sample.
      - In **eval mode** (``random_view=False``): deterministic view 0 is
        always returned — matches what the extractors store for val/test.

    Parameters
    ----------
    text_cache_path, audio_cache_path, video_cache_path : Path
        Paths to the three per-modality cache files for the same split.
    random_view : bool
        Pick a fresh view per modality per ``__getitem__`` call.  Use True
        for train, False for val/test.
    expected_fingerprint : dict or None
        Optional ``{"text": fp, "audio": fp, "video": fp}`` mapping.  If
        provided, each modality's stored fingerprint is compared and a
        UserWarning is emitted on any mismatch (does NOT block — extractors
        already enforce this on write).
    """

    def __init__(
        self,
        text_cache_path,
        audio_cache_path,
        video_cache_path,
        random_view: bool = True,
        expected_fingerprint: Optional[dict] = None,
    ):
        per_mod_paths = {
            "text":  Path(text_cache_path),
            "audio": Path(audio_cache_path),
            "video": Path(video_cache_path),
        }

        # ── Load per-modality caches ─────────────────────────────────────────
        caches = {
            mod: _load_per_modality_cache(per_mod_paths[mod], mod)
            for mod in ("text", "audio", "video")
        }

        # ── Intersect video_ids across the three modalities ──────────────────
        ids_per_mod = {mod: list(caches[mod]["video_ids"]) for mod in caches}
        common = sorted(
            set(ids_per_mod["text"])
            & set(ids_per_mod["audio"])
            & set(ids_per_mod["video"])
        )
        if not common:
            raise RuntimeError(
                "No common video_ids across text/audio/video caches.  "
                "Re-extract the affected modalities."
            )

        # Report any drops so the user knows alignment cost them samples
        for mod, ids in ids_per_mod.items():
            dropped = set(ids) - set(common)
            if dropped:
                warnings.warn(
                    f"[CachedFusionDataset] {mod} cache had "
                    f"{len(dropped)} videos not present in the other modalities — "
                    f"those samples will be skipped.",
                    UserWarning,
                    stacklevel=2,
                )

        self.video_ids: List[str] = common

        # ── Reorder each modality's tensor + labels to common ID order ───────
        per_mod_tensors = {}
        per_mod_labels = {}
        for mod, cache in caches.items():
            idx_of = {vid: i for i, vid in enumerate(cache["video_ids"])}
            order = [idx_of[v] for v in common]
            per_mod_tensors[mod] = cache["embeddings"][order]  # (N, K_mod, dim_mod)
            per_mod_labels[mod] = [int(cache["labels"][i]) for i in order]

        # Sanity-check label agreement across modalities
        if (
            per_mod_labels["text"] != per_mod_labels["audio"]
            or per_mod_labels["text"] != per_mod_labels["video"]
        ):
            warnings.warn(
                "Label mismatch detected across per-modality caches for some "
                "video_ids.  Using text-cache labels by convention; consider "
                "re-extracting affected modalities.",
                UserWarning,
                stacklevel=2,
            )
        self.labels: List[int] = per_mod_labels["text"]

        self.text:  torch.Tensor = per_mod_tensors["text"]   # (N, K_t, dim_t)
        self.audio: torch.Tensor = per_mod_tensors["audio"]  # (N, K_a, dim_a)
        self.video: torch.Tensor = per_mod_tensors["video"]  # (N, K_v, dim_v)

        self.K_t = self.text.shape[1]
        self.K_a = self.audio.shape[1]
        self.K_v = self.video.shape[1]
        self.random_view = bool(random_view)

        self.weights_fingerprint = {
            mod: caches[mod].get("weights_fingerprint") for mod in caches
        }
        self.extraction_config = {
            mod: caches[mod].get("extraction_config") for mod in caches
        }

        if expected_fingerprint is not None:
            for mod in ("text", "audio", "video"):
                want = expected_fingerprint.get(mod)
                got = self.weights_fingerprint.get(mod)
                if want is not None and got != want:
                    warnings.warn(
                        f"Cached {mod} embeddings at {per_mod_paths[mod]} were "
                        f"produced with a different backbone fingerprint.  "
                        f"Consider re-running extract_{mod}.py.\n"
                        f"  cached  : {got}\n"
                        f"  current : {want}",
                        UserWarning,
                        stacklevel=2,
                    )

        print(
            f"[CachedFusionDataset] N={self.text.shape[0]}  "
            f"K_t={self.K_t}  K_a={self.K_a}  K_v={self.K_v}  "
            f"random_view={self.random_view}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if self.random_view:
            kt = int(torch.randint(self.K_t, (1,)).item()) if self.K_t > 1 else 0
            ka = int(torch.randint(self.K_a, (1,)).item()) if self.K_a > 1 else 0
            kv = int(torch.randint(self.K_v, (1,)).item()) if self.K_v > 1 else 0
        else:
            kt = ka = kv = 0
        return (
            self.text[idx, kt],
            self.audio[idx, ka],
            self.video[idx, kv],
            self.labels[idx],
        )


def get_cached_multimodal_splits(
    cache_dir,
    train_random_view: bool = True,
    expected_fingerprint: Optional[dict] = None,
) -> Dict[str, "MultimodalCachedFusionDataset"]:
    """
    Convenience constructor: build cached datasets for train/val/test from
    the three per-modality cache files.

    Parameters
    ----------
    cache_dir : Path or str
        Directory containing ``{text,audio,video}_embs_{train,val,test}.pt``.
    train_random_view : bool
        If True (default), train samples randomly pick a view per modality
        per fetch — the cheap "augmentation" knob.  Val/test always use
        the deterministic view 0.
    expected_fingerprint : dict or None
        Optional ``{"text": fp, "audio": fp, "video": fp}`` mapping;
        mismatches emit UserWarnings.
    """
    cache_dir = Path(cache_dir)

    def _paths_for(split: str):
        return {
            "text_cache_path":  cache_dir / f"text_embs_{split}.pt",
            "audio_cache_path": cache_dir / f"audio_embs_{split}.pt",
            "video_cache_path": cache_dir / f"video_embs_{split}.pt",
        }

    return {
        "train": MultimodalCachedFusionDataset(
            **_paths_for("train"),
            random_view=train_random_view,
            expected_fingerprint=expected_fingerprint,
        ),
        "val": MultimodalCachedFusionDataset(
            **_paths_for("val"),
            random_view=False,
            expected_fingerprint=expected_fingerprint,
        ),
        "test": MultimodalCachedFusionDataset(
            **_paths_for("test"),
            random_view=False,
            expected_fingerprint=expected_fingerprint,
        ),
    }
