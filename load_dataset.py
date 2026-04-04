"""
BAH Dataset Multi-Modal PyTorch Dataset Loaders
================================================

This script provides PyTorch Dataset classes for the BAH (Behavioral Affective Halos) 
dataset with proper train/validation/test splits for all modalities (video, audio, text).

Dataset Structure:
- data/split/train.txt, val.txt, test.txt - Video-level splits with labels and transcripts
- data/split-frames/train.txt, val.txt, test.txt - Frame-level splits
- data/Videos/ - Original video files
- data/cropped-aligned-faces/ - Face images (frames)
- data/transcription/ - Text transcripts
- data/audio/ - Audio files (if available)

Labels:
- 0: No A-H (No Affective Halo)
- 1: With A-H (Affective Halo present)

Usage:
    from load_dataset import VideoDataset, TextDataset, AudioDataset, FrameDataset
    
    # Create datasets
    train_video_dataset = VideoDataset(split='train')
    train_text_dataset = TextDataset(split='train')
    train_audio_dataset = AudioDataset(split='train')
    train_frame_dataset = FrameDataset(split='train')
    
    # Use with DataLoader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_video_dataset, batch_size=32, shuffle=True)
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


# Configuration
DATA_DIR = Path('data')
SPLIT_DIR = DATA_DIR / 'split'
SPLIT_FRAMES_DIR = DATA_DIR / 'split-frames'
VIDEO_DIR = DATA_DIR / 'Videos'
FACES_DIR = DATA_DIR / 'cropped-aligned-faces'
TRANSCRIPTION_DIR = DATA_DIR / 'transcription'
AUDIO_DIR = DATA_DIR / 'audio'

# Class mapping
CLASS_MAPPING = {
    'With A-H': 1,
    'No A-H': 0
}


# ============================================================================
# Base loading functions
# ============================================================================

def load_video_split_file(split_file: Path) -> pd.DataFrame:
    """
    Load a video-level split file and return a DataFrame.
    
    Args:
        split_file: Path to the split file (train.txt, val.txt, or test.txt)
        
    Returns:
        DataFrame with columns: video_path, label, transcript
    """
    data = []
    
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',', 2)
                if len(parts) >= 2:
                    video_path = parts[0]
                    label = int(parts[1])
                    transcript = parts[2] if len(parts) > 2 else ""
                    
                    data.append({
                        'video_path': video_path,
                        'label': label,
                        'transcript': transcript,
                        'video_id': Path(video_path).stem,
                        'participant_id': video_path.split('/')[1],
                        'question': video_path.split('/')[2].replace('_Question_', 'Q').replace('_Video.mp4', ''),
                    })
    
    return pd.DataFrame(data)


def load_frame_split_file(split_file: Path) -> pd.DataFrame:
    """
    Load a frame-level split file and return a DataFrame.
    
    Args:
        split_file: Path to the split file (train.txt, val.txt, or test.txt)
        
    Returns:
        DataFrame with columns: frame_path, label
    """
    data = []
    
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.rsplit(',', 1)
                if len(parts) == 2:
                    frame_path = parts[0]
                    label = int(parts[1])
                    
                    frame_name = Path(frame_path).name
                    video_folder = Path(frame_path).parent.name
                    
                    data.append({
                        'frame_path': frame_path,
                        'label': label,
                        'frame_name': frame_name,
                        'video_folder': video_folder,
                    })
    
    return pd.DataFrame(data)


# ============================================================================
# Video Dataset
# ============================================================================

class VideoDataset(Dataset):
    """
    Video Dataset for BAH dataset.
    
    Reads directly from video files (.mp4).
    """
    
    def __init__(
        self,
        split: str = 'train',
        video_dir: Path = VIDEO_DIR,
        image_size: Tuple[int, int] = (224, 224),
        max_frames: int = 16,
        transform=None
    ):
        """
        Initialize Video Dataset.
        
        Args:
            split: 'train', 'val', or 'test'
            video_dir: Directory containing video files
            image_size: Target image size (height, width)
            max_frames: Maximum number of frames to sample per video
            transform: Optional transform to apply to images
        """
        self.split = split
        self.video_dir = video_dir
        self.image_size = image_size
        self.max_frames = max_frames
        self.transform = transform
        
        # Load video-level data
        split_file = SPLIT_DIR / f'{split}.txt'
        self.data = load_video_split_file(split_file)
        
        # Store video paths
        self.video_data = []
        for _, row in self.data.iterrows():
            video_path = row['video_path']
            # video_path already contains "Videos/" prefix, so use it directly
            # from DATA_DIR (which is the parent of VIDEO_DIR)
            full_path = DATA_DIR / video_path
            
            if full_path.exists():
                self.video_data.append({
                    'video_path': str(full_path),
                    'label': row['label'],
                    'video_id': row['video_id'],
                    'participant_id': row['participant_id'],
                })
            else:
                # Debug: print missing video
                print(f"Video not found: {full_path}")
        
        print(f"VideoDataset ({split}): {len(self.video_data)} videos loaded")
    
    def __len__(self) -> int:
        return len(self.video_data)
    
    def _read_video(self, video_path: str) -> List[np.ndarray]:
        """Read frames from video file using OpenCV."""
        import cv2
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return frames
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return frames
        
        # Sample frame indices uniformly
        if total_frames <= self.max_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
        
        # Read selected frames
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
                frames.append(frame)
        
        cap.release()
        return frames
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a video sample.
        
        Returns:
            Tuple of (frames_tensor, label, video_id)
        """
        import cv2
        
        video_info = self.video_data[idx]
        video_path = video_info['video_path']
        
        # Read frames from video
        frames = self._read_video(video_path)
        
        # Process frames
        processed_frames = []
        for img in frames:
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            processed_frames.append(img)
        
        # Pad if necessary
        while len(processed_frames) < self.max_frames:
            processed_frames.append(torch.zeros(3, self.image_size[0], self.image_size[1]))
        
        # Stack frames: (T, C, H, W)
        frames_tensor = torch.stack(processed_frames[:self.max_frames])
        
        return frames_tensor, video_info['label'], video_info['video_id']


def get_video_dataloader(
    split: str = 'train',
    batch_size: int = 8,
    image_size: Tuple[int, int] = (224, 224),
    max_frames: int = 16,
    num_workers: int = 4,
    shuffle: bool = None
) -> DataLoader:
    """Get DataLoader for video dataset."""
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = VideoDataset(
        split=split,
        image_size=image_size,
        max_frames=max_frames
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# ============================================================================
# Text Dataset
# ============================================================================

class TextDataset(Dataset):
    """
    Text Dataset for BAH dataset.
    
    Returns text transcripts with labels.
    """
    
    def __init__(
        self,
        split: str = 'train',
        max_length: int = 512,
        tokenizer=None
    ):
        """
        Initialize Text Dataset.
        
        Args:
            split: 'train', 'val', or 'test'
            max_length: Maximum sequence length for tokenization
            tokenizer: Optional tokenizer (e.g., from transformers)
        """
        self.split = split
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Load text data
        split_file = SPLIT_DIR / f'{split}.txt'
        self.data = load_video_split_file(split_file)
        
        print(f"TextDataset ({split}): {len(self.data)} samples loaded")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[dict, int, str]:
        """
        Get a text sample.
        
        Returns:
            Tuple of (input_ids, attention_mask), label, text
        """
        row = self.data.iloc[idx]
        text = row['transcript']
        label = row['label']
        video_id = row['video_id']
        
        if self.tokenizer is not None:
            # Tokenize text
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            # Remove batch dimension
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            
            return {'input_ids': input_ids, 'attention_mask': attention_mask}, label, text
        
        else:
            # Return raw text
            return {'text': text}, label, text  # Return text as the third element for consistency


def get_text_dataloader(
    split: str = 'train',
    batch_size: int = 16,
    max_length: int = 512,
    tokenizer=None,
    num_workers: int = 4,
    shuffle: bool = None
) -> DataLoader:
    """Get DataLoader for text dataset."""
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = TextDataset(
        split=split,
        max_length=max_length,
        tokenizer=tokenizer
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# ============================================================================
# Audio Dataset
# ============================================================================

class AudioDataset(Dataset):
    """
    Audio Dataset for BAH dataset.
    
    Returns audio features with labels.
    """
    
    def __init__(
        self,
        split: str = 'train',
        audio_dir: Path = AUDIO_DIR,
        sample_rate: int = 16000,
        max_length: int = 10,  # seconds
        feature_extractor=None
    ):
        """
        Initialize Audio Dataset.
        
        Args:
            split: 'train', 'val', or 'test'
            audio_dir: Directory containing audio files
            sample_rate: Audio sample rate
            max_length: Maximum audio length in seconds
            feature_extractor: Optional feature extractor
        """
        self.split = split
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.feature_extractor = feature_extractor
        
        # Load audio data (using video paths as reference)
        split_file = SPLIT_DIR / f'{split}.txt'
        self.data = load_video_split_file(split_file)
        
        # Add audio path column
        self.data['audio_path'] = self.data['video_path'].str.replace('.mp4', '.wav', regex=False)
        
        print(f"AudioDataset ({split}): {len(self.data)} samples loaded")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get an audio sample.
        
        Returns:
            Tuple of (audio_features, label, video_id)
        """
        import cv2
        import numpy as np
        
        row = self.data.iloc[idx]
        audio_path = row['audio_path']
        label = row['label']
        video_id = row['video_id']
        
        # Try to load audio
        full_audio_path = self.audio_dir / audio_path.replace('Videos/', '')
        
        if full_audio_path.exists():
            try:
                import librosa
                audio, sr = librosa.load(str(full_audio_path), sr=self.sample_rate)
                
                # Pad or truncate to max_length
                max_samples = self.sample_rate * self.max_length
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
                else:
                    audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
                
                audio_tensor = torch.from_numpy(audio).float()
                
            except Exception as e:
                # Return zeros if loading fails
                audio_tensor = torch.zeros(self.sample_rate * self.max_length)
        else:
            # Audio file not found - return zeros
            audio_tensor = torch.zeros(self.sample_rate * self.max_length)
        
        return audio_tensor, label, video_id


def get_audio_dataloader(
    split: str = 'train',
    batch_size: int = 16,
    sample_rate: int = 16000,
    max_length: int = 10,
    num_workers: int = 4,
    shuffle: bool = None
) -> DataLoader:
    """Get DataLoader for audio dataset."""
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = AudioDataset(
        split=split,
        sample_rate=sample_rate,
        max_length=max_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# ============================================================================
# Frame Dataset (for image-based video processing)
# ============================================================================

class FrameDataset(Dataset):
    """
    Frame Dataset for BAH dataset.
    
    Returns individual face frames with labels.
    """
    
    def __init__(
        self,
        split: str = 'train',
        faces_dir: Path = FACES_DIR,
        image_size: Tuple[int, int] = (224, 224),
        transform=None
    ):
        """
        Initialize Frame Dataset.
        
        Args:
            split: 'train', 'val', or 'test'
            faces_dir: Directory containing face images
            image_size: Target image size (height, width)
            transform: Optional transform to apply to images
        """
        self.split = split
        self.faces_dir = faces_dir
        self.image_size = image_size
        self.transform = transform
        
        # Load frame-level data
        split_file = SPLIT_FRAMES_DIR / f'{split}.txt'
        self.data = load_frame_split_file(split_file)
        
        print(f"FrameDataset ({split}): {len(self.data)} frames loaded")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a frame sample.
        
        Returns:
            Tuple of (image_tensor, label, frame_path)
        """
        import cv2
        
        row = self.data.iloc[idx]
        frame_path = row['frame_path']
        label = row['label']
        
        # Load image
        full_path = self.faces_dir / frame_path
        img = cv2.imread(str(full_path))
        
        if img is None:
            # Create blank image if not found
            img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img, label, frame_path


def get_frame_dataloader(
    split: str = 'train',
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    shuffle: bool = None
) -> DataLoader:
    """Get DataLoader for frame dataset."""
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = FrameDataset(
        split=split,
        image_size=image_size
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# ============================================================================
# Utility functions
# ============================================================================

def load_all_dataloaders(
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    max_frames: int = 16,
    num_workers: int = 4
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Create all dataloaders for all modalities and splits.
    
    Returns:
        Dictionary with modality -> split -> DataLoader
    """
    dataloaders = {}
    
    # Video dataloaders
    print("\nCreating Video DataLoaders...")
    dataloaders['video'] = {
        split: get_video_dataloader(split, batch_size=batch_size // 2, image_size=image_size, max_frames=max_frames, num_workers=num_workers)
        for split in ['train', 'val', 'test']
    }
    
    # Text dataloaders
    print("Creating Text DataLoaders...")
    dataloaders['text'] = {
        split: get_text_dataloader(split, batch_size=batch_size, num_workers=num_workers)
        for split in ['train', 'val', 'test']
    }
    
    # Audio dataloaders
    print("Creating Audio DataLoaders...")
    dataloaders['audio'] = {
        split: get_audio_dataloader(split, batch_size=batch_size, num_workers=num_workers)
        for split in ['train', 'val', 'test']
    }
    
    # Frame dataloaders
    print("Creating Frame DataLoaders...")
    dataloaders['frame'] = {
        split: get_frame_dataloader(split, batch_size=batch_size, image_size=image_size, num_workers=num_workers)
        for split in ['train', 'val', 'test']
    }
    
    return dataloaders


def print_dataset_info():
    """Print information about all datasets."""
    print("=" * 60)
    print("BAH Dataset Information")
    print("=" * 60)
    
    # Video-level data
    for split in ['train', 'val', 'test']:
        split_file = SPLIT_DIR / f'{split}.txt'
        if split_file.exists():
            df = load_video_split_file(split_file)
            label_counts = df['label'].value_counts()
            print(f"\n{split.upper()} (Video/Text/Audio):")
            print(f"  Total: {len(df)}")
            print(f"  Class 0: {label_counts.get(0, 0)}")
            print(f"  Class 1: {label_counts.get(1, 0)}")
    
    # Frame-level data
    for split in ['train', 'val', 'test']:
        split_file = SPLIT_FRAMES_DIR / f'{split}.txt'
        if split_file.exists():
            df = load_frame_split_file(split_file)
            label_counts = df['label'].value_counts()
            print(f"\n{split.upper()} (Frames):")
            print(f"  Total: {len(df)}")
            print(f"  Class 0: {label_counts.get(0, 0)}")
            print(f"  Class 1: {label_counts.get(1, 0)}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print_dataset_info()
    
    print("\n" + "=" * 60)
    print("Testing datasets...")
    print("=" * 60)
    
    # Test Video Dataset
    print("\n--- Video Dataset Test ---")
    video_ds = VideoDataset(split='train', max_frames=8)
    if len(video_ds) > 0:
        frames, label, vid = video_ds[0]
        print(f"  Shape: {frames.shape}, Label: {label}, ID: {vid}")
    
    # Test Text Dataset
    print("\n--- Text Dataset Test ---")
    text_ds = TextDataset(split='train')
    if len(text_ds) > 0:
        data, label, text = text_ds[0]
        print(f"  Keys: {data.keys()}, Label: {label}")
        print(f"  Text preview: {text[:50]}...")
    
    # Test Audio Dataset
    print("\n--- Audio Dataset Test ---")
    audio_ds = AudioDataset(split='train')
    if len(audio_ds) > 0:
        audio, label, vid = audio_ds[0]
        print(f"  Shape: {audio.shape}, Label: {label}, ID: {vid}")
    
    # Test Frame Dataset
    print("\n--- Frame Dataset Test ---")
    frame_ds = FrameDataset(split='train')
    if len(frame_ds) > 0:
        img, label, path = frame_ds[0]
        print(f"  Shape: {img.shape}, Label: {label}")
    
    print("\nAll tests completed!")
