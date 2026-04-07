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

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("data")
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
            raw_df = (
                raw_df.groupby("video_folder", group_keys=False)
                .apply(lambda g: g.iloc[::subsample])
                .reset_index(drop=True)
            )

        self.data = raw_df.reset_index(drop=True)
        print(f"FrameDataset [{split}]: {len(self.data):,} frames loaded")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        import cv2

        row = self.data.iloc[idx]
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
            raw_df = (
                raw_df.groupby("video_folder", group_keys=False)
                .apply(lambda g: g.iloc[::subsample])
                .reset_index(drop=True)
            )

        valid_rows: List[dict] = []
        for _, row in raw_df.iterrows():
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

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)

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

        row = self.data.iloc[idx]
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

        raw_df = load_frame_split_file(SPLIT_FRAMES_DIR / f"{split}.txt")
        raw_df["_fnum"] = raw_df["frame_name"].apply(_frame_number)
        raw_df = raw_df.sort_values(["video_folder", "_fnum"]).reset_index(drop=True)

        self.clips: List[Tuple[str, int, List[str]]] = []
        for vfolder, grp in raw_df.groupby("video_folder", sort=False):
            label = int(grp["label"].iloc[0])
            paths = grp["frame_path"].tolist()
            n = len(paths)
            start = 0
            while start < n:
                end = start + clip_len
                clip_paths = paths[start:end]
                if len(clip_paths) < clip_len and not pad_last:
                    break
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
) -> Dict[str, VideoClipDataset]:
    """Return ``{'train': …, 'val': …, 'test': …}`` VideoClipDataset."""
    kw = dict(
        frames_dir=frames_dir,
        image_size=image_size,
        clip_len=clip_len,
        clip_stride=clip_stride,
        pad_last=pad_last,
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

        raw_df = load_frame_split_file(SPLIT_FRAMES_DIR / f"{split}.txt")
        raw_df["_fnum"] = raw_df["frame_name"].apply(_frame_number)
        raw_df = raw_df.sort_values(["video_folder", "_fnum"]).reset_index(drop=True)

        # Filter out skipped / missing segmented frames
        def _seg_ok(fp: str) -> bool:
            p = self.seg_dir / fp
            return p.exists() and not p.with_suffix(".skip").exists()

        raw_df = raw_df[raw_df["frame_path"].apply(_seg_ok)].reset_index(drop=True)

        self.clips: List[Tuple[str, int, List[str]]] = []
        for vfolder, grp in raw_df.groupby("video_folder", sort=False):
            label = int(grp["label"].iloc[0])
            paths = grp["frame_path"].tolist()
            n = len(paths)
            start = 0
            while start < n:
                end = start + clip_len
                clip_paths = paths[start:end]
                if len(clip_paths) < clip_len and not pad_last:
                    break
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
) -> Dict[str, SegmentedVideoClipDataset]:
    """Return ``{'train': …, 'val': …, 'test': …}`` SegmentedVideoClipDataset."""
    kw = dict(
        seg_dir=seg_dir,
        image_size=image_size,
        clip_len=clip_len,
        clip_stride=clip_stride,
        pad_last=pad_last,
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

    Loads audio from pre-extracted files in ``data/audio/`` (preferred) or
    falls back to extracting on-the-fly from the original ``.mp4`` video files
    using ``torchaudio`` (preferred) or ``librosa``.

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
        audio_dir: Optional[Path] = None,
        video_dir: Path = VIDEO_DIR,
        sample_rate: int = 16000,
        max_length_sec: float = 30.0,
        feature_extractor=None,
    ):
        self.split = split
        self.audio_dir = Path(audio_dir) if audio_dir is not None else AUDIO_DIR
        self.video_dir = Path(video_dir)
        self.sample_rate = sample_rate
        self.max_length_sec = max_length_sec
        self.max_samples = int(sample_rate * max_length_sec)
        self.feature_extractor = feature_extractor

        split_file = SPLIT_DIR / f"{split}.txt"
        self.data = load_video_split_file(split_file)

        self.video_data: List[dict] = []
        for _, row in self.data.iterrows():
            # Prefer pre-extracted audio: data/audio/<participant>/<question>/<stem>.wav
            audio_path = self.audio_dir / Path(row["video_path"]).with_suffix(".wav")
            if audio_path.exists():
                self.video_data.append(
                    {
                        "audio_path": str(audio_path),
                        "label": row["label"],
                        "video_id": row["video_id"],
                        "source": "audio_dir",
                    }
                )
                continue

            # Fallback: load from original .mp4
            video_path = DATA_DIR / row["video_path"]
            if video_path.exists():
                self.video_data.append(
                    {
                        "audio_path": str(video_path),
                        "label": row["label"],
                        "video_id": row["video_id"],
                        "source": "video_dir",
                    }
                )

        n_audio = sum(1 for d in self.video_data if d["source"] == "audio_dir")
        n_video = sum(1 for d in self.video_data if d["source"] == "video_dir")
        print(
            f"AudioDataset [{split}]: {len(self.video_data):,} samples "
            f"(pre-extracted: {n_audio}, from video: {n_video}, "
            f"sr={sample_rate}, max={max_length_sec}s)"
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
    audio_dir: Optional[Path] = None,
    video_dir: Path = VIDEO_DIR,
) -> Dict[str, AudioDataset]:
    """Return ``{'train': …, 'val': …, 'test': …}`` AudioDataset.

    Looks for pre-extracted ``.wav`` files under ``audio_dir`` (defaults to
    ``data/audio/``).  Falls back to loading audio from the original ``.mp4``
    files when a pre-extracted file is not found.
    """
    kw = dict(
        audio_dir=audio_dir,
        video_dir=video_dir,
        sample_rate=sample_rate,
        max_length_sec=max_length_sec,
        feature_extractor=feature_extractor,
    )
    return {
        "train": AudioDataset("train", **kw),
        "val": AudioDataset("val", **kw),
        "test": AudioDataset("test", **kw),
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
    ):
        self.split = split
        self.max_length = max_length
        self.tokenizer = tokenizer

        split_file = SPLIT_DIR / f"{split}.txt"
        self.data = load_video_split_file(split_file)

        print(f"TextDataset [{split}]: {len(self.data):,} samples loaded")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        text = row["transcript"]
        label = int(row["label"])

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
) -> Dict[str, TextDataset]:
    """Return ``{'train': …, 'val': …, 'test': …}`` TextDataset."""
    return {
        "train": TextDataset("train", max_length=max_length, tokenizer=tokenizer),
        "val": TextDataset("val", max_length=max_length, tokenizer=tokenizer),
        "test": TextDataset("test", max_length=max_length, tokenizer=tokenizer),
    }
