"""
BAH A/H Detection — Live Demo Application
==========================================

A Tkinter-based GUI that:
  1. Shows a live camera preview.
  2. Records video + audio on button press.
  3. Runs the full multimodal pipeline:
       Whisper STT → YOLO segmentation → embed text/audio/video →
       per-modality MLP heads → late-fusion → A/H prediction.
  4. Logs every step in a sidebar.

Usage
-----
    cd <repo_root>
    python demo/app.py

Requirements beyond the main project:
    pip install openai-whisper sounddevice
"""

from __future__ import annotations

import datetime
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext

import cv2
import numpy as np
import toml
import torch
from PIL import Image, ImageTk

# ── Resolve repo root (demo/ lives one level below) ──────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# ── Load config ──────────────────────────────────────────────────────────────
CFG = toml.load(SCRIPT_DIR / "config.toml")
RECORDINGS_DIR = REPO_ROOT / CFG["paths"]["recordings_dir"]
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline helpers (run in a background thread)
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Step 1: extract audio from video ─────────────────────────────────────────

def extract_audio_from_video(video_path: Path, audio_path: Path, sr: int = 16000) -> Path:
    """Extract audio track from video file using ffmpeg, save as WAV."""
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
        str(audio_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return audio_path


# ── Step 2: Whisper transcription ────────────────────────────────────────────

def transcribe_audio(audio_path: Path, model_size: str = "base") -> str:
    """Run OpenAI Whisper on the audio file and return the transcript."""
    import whisper
    model = whisper.load_model(model_size)
    result = model.transcribe(str(audio_path))
    return result["text"].strip()


# ── Step 3: YOLO segmentation ───────────────────────────────────────────────

def segment_frames_yolo(
    video_path: Path,
    output_dir: Path,
    yolo_model_path: Path,
    image_size: int = 224,
) -> list[Path]:
    """
    Extract frames from video, run YOLO instance segmentation to isolate the
    largest person (background removed via mask), crop to bounding box, and
    save the segmented frames.

    Returns a list of saved frame paths (skips frames with no person detected).
    """
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(yolo_model_path))

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    saved_paths: list[Path] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        # Find the largest person detection (class 0 = person in COCO).
        best_idx = -1
        best_area = 0
        best_result = None
        for r in results:
            if r.boxes is None or r.masks is None:
                continue
            for i, (box, cls) in enumerate(
                zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy())
            ):
                if int(cls) != 0:  # person class
                    continue
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_idx = i
                    best_result = r

        if best_idx >= 0 and best_result is not None:
            # Get the segmentation mask for the best person.
            mask = best_result.masks.data[best_idx].cpu().numpy()  # (H_mask, W_mask)
            h, w = frame.shape[:2]
            # Resize mask to frame size if needed.
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            # Apply mask: white background where mask is 0, keep person pixels.
            mask_bool = mask > 0.5
            mask_3ch = mask_bool.astype(np.uint8)[:, :, np.newaxis]
            white_bg = np.full_like(frame, 255)  # white canvas
            segmented = np.where(mask_3ch, frame, white_bg)

            # Crop to bounding box.
            box = best_result.boxes.xyxy[best_idx].cpu().numpy()
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = segmented[y1:y2, x1:x2]
            crop = cv2.resize(crop, (image_size, image_size))

            out_path = output_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), crop)
            saved_paths.append(out_path)

        frame_idx += 1

    cap.release()
    return saved_paths


# ── Step 4: Embedding extraction ────────────────────────────────────────────

def extract_text_embedding(text: str, device: torch.device) -> torch.Tensor:
    """Extract text embedding using the frozen DistilRoBERTa-Emotional backbone."""
    sys.path.insert(0, str(REPO_ROOT / "multimodal" / "src"))
    from embedders import TextEmbedder

    cfg_t = CFG["text"]
    embedder = TextEmbedder(
        model_name=cfg_t["model_name"],
        weights_path=str(REPO_ROOT / cfg_t["weights_path"]),
        device=device,
        max_length=cfg_t["max_length"],
    )
    emb = embedder.embed([text])  # (1, 768)
    return emb.squeeze(0).cpu()   # (768,)


def extract_audio_embedding(audio_path: Path, device: torch.device) -> torch.Tensor:
    """Extract audio embedding using the frozen Wav2Vec2-Emotional backbone."""
    import librosa
    sys.path.insert(0, str(REPO_ROOT / "multimodal" / "src"))
    from embedders import AudioEmbedder

    cfg_a = CFG["audio"]
    sr = cfg_a["sample_rate"]
    max_sec = cfg_a["max_length_sec"]
    max_samples = int(sr * max_sec)

    wf, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    if len(wf) > max_samples:
        wf = wf[:max_samples]
    elif len(wf) < max_samples:
        wf = np.pad(wf, (0, max_samples - len(wf)), mode="constant")

    embedder = AudioEmbedder(
        model_name=cfg_a["model_name"],
        weights_path=str(REPO_ROOT / cfg_a["weights_path"]),
        device=device,
        sample_rate=sr,
        max_length_sec=max_sec,
    )
    wf_tensor = torch.from_numpy(wf.astype(np.float32)).unsqueeze(0)  # (1, samples)
    emb = embedder.embed_waveforms(wf_tensor)  # (1, 256)
    return emb.squeeze(0).cpu()  # (256,)


def extract_video_embedding(
    frame_paths: list[Path], device: torch.device
) -> torch.Tensor:
    """
    Extract video embedding using the frozen video backbone.

    Supports both Swin-Tiny and VideoMAE via the ``backbone`` key in config.
    Loads segmented frames, builds a clip tensor (1, T, C, H, W),
    normalises with ImageNet stats, and runs through the embedder.
    """
    sys.path.insert(0, str(REPO_ROOT / "multimodal" / "src"))
    from embedders import VideoEmbedder, VideoMAEEmbedder
    from transformers import AutoImageProcessor

    cfg_v = CFG["video"]
    backbone = cfg_v.get("backbone", "swin")
    clip_len = cfg_v["clip_len"]
    image_size = cfg_v["image_size"]

    # Load and preprocess frames.
    frames: list[torch.Tensor] = []
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size))
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (C, H, W)
        frames.append(t)

    if not frames:
        # No valid frames — return zero embedding.
        return torch.zeros(768)

    # Subsample or pad to clip_len.
    if len(frames) > clip_len:
        indices = np.linspace(0, len(frames) - 1, clip_len, dtype=int)
        frames = [frames[i] for i in indices]
    while len(frames) < clip_len:
        frames.append(torch.zeros(3, image_size, image_size))

    clip = torch.stack(frames).unsqueeze(0)  # (1, T, C, H, W)

    # ImageNet normalisation.
    processor = AutoImageProcessor.from_pretrained(cfg_v["model_name"])
    mean = torch.tensor(list(processor.image_mean)).view(1, 1, 3, 1, 1)
    std = torch.tensor(list(processor.image_std)).view(1, 1, 3, 1, 1)
    clip = (clip - mean) / std
    clip = clip.to(device)

    # Select embedder based on backbone.
    if backbone == "videomae":
        embedder = VideoMAEEmbedder(
            model_name=cfg_v["model_name"],
            weights_path=str(REPO_ROOT / cfg_v["weights_path"]),
            device=device,
        )
    else:
        embedder = VideoEmbedder(
            model_name=cfg_v["model_name"],
            weights_path=str(REPO_ROOT / cfg_v["weights_path"]),
            device=device,
        )
    emb = embedder.embed(clip)  # (1, 768)
    return emb.squeeze(0).cpu()  # (768,)


# ── Step 5: Late-fusion classification ──────────────────────────────────────

def load_modality_mlp(ckpt_path: Path, device: torch.device):
    """Load a trained ModalityMLP from a late-fusion checkpoint."""
    import torch.nn as nn

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim = ckpt["input_dim"]
    hidden_dims = ckpt["hidden_dims"]
    dropout = ckpt.get("dropout", 0.3)
    input_dropout = ckpt.get("input_dropout", 0.0)

    # Rebuild ModalityMLP architecture.
    layers: list[nn.Module] = []
    if input_dropout > 0:
        layers.append(nn.Dropout(p=input_dropout))
    in_dim = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(p=dropout)]
        in_dim = h
    layers.append(nn.Linear(in_dim, 2))
    model = nn.Sequential(*layers)
    # The ModalityMLP checkpoint wraps layers in self.mlp, so keys are
    # prefixed with "mlp." — strip that prefix for our bare Sequential.
    sd = ckpt["state_dict"]
    sd = {k.removeprefix("mlp."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def run_late_fusion(
    text_emb: torch.Tensor,
    audio_emb: torch.Tensor,
    video_emb: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Run the full late-fusion pipeline:
      1. Scale each embedding with its fitted StandardScaler.
      2. Forward through per-modality MLP heads → softmax probs.
      3. Fuse via simple probability averaging.

    Returns a dict with per-modality probs and the fused prediction.
    """
    import joblib

    fcfg = CFG["fusion"]
    wdir = REPO_ROOT / fcfg["weights_dir"]

    # Load scalers.
    scaler_t = joblib.load(wdir / fcfg["text_scaler"])
    scaler_a = joblib.load(wdir / fcfg["audio_scaler"])
    scaler_v = joblib.load(wdir / fcfg["video_scaler"])

    # Scale embeddings.
    text_scaled = torch.from_numpy(
        scaler_t.transform(text_emb.unsqueeze(0).numpy()).astype(np.float32)
    ).to(device)
    audio_scaled = torch.from_numpy(
        scaler_a.transform(audio_emb.unsqueeze(0).numpy()).astype(np.float32)
    ).to(device)
    video_scaled = torch.from_numpy(
        scaler_v.transform(video_emb.unsqueeze(0).numpy()).astype(np.float32)
    ).to(device)

    # Load MLP heads.
    mlp_t = load_modality_mlp(wdir / fcfg["text_head"], device)
    mlp_a = load_modality_mlp(wdir / fcfg["audio_head"], device)
    mlp_v = load_modality_mlp(wdir / fcfg["video_head"], device)

    # Forward pass → softmax probs.
    with torch.no_grad():
        p_t = torch.softmax(mlp_t(text_scaled), dim=-1).cpu().numpy()[0]
        p_a = torch.softmax(mlp_a(audio_scaled), dim=-1).cpu().numpy()[0]
        p_v = torch.softmax(mlp_v(video_scaled), dim=-1).cpu().numpy()[0]

    # Load meta artifact for temperatures + weights.
    meta = joblib.load(wdir / fcfg["meta"])

    # Use PSO-optimized temperatures and weights (prob space) for the
    # text+audio+video subset — these were tuned on the validation set.
    subset_key = "text+audio+video"
    subset_meta = meta.get("subsets", {}).get(subset_key, {})
    pso_temps = subset_meta.get("pso_temps_prob", meta.get("temperatures", {}))
    pso_weights = subset_meta.get("pso_weights_prob", {
        "text": 1/3, "audio": 1/3, "video": 1/3,
    })

    # Apply temperature scaling.
    def _apply_temp(probs, T):
        if T is None or T == 1.0:
            return probs
        logits = np.log(np.clip(probs, 1e-12, None))
        scaled = logits / T
        exp_s = np.exp(scaled - scaled.max())
        return exp_s / exp_s.sum()

    p_t_cal = _apply_temp(p_t, pso_temps.get("text"))
    p_a_cal = _apply_temp(p_a, pso_temps.get("audio"))
    p_v_cal = _apply_temp(p_v, pso_temps.get("video"))

    # Fuse: PSO-optimized weighted mean of calibrated probabilities.
    w_t = pso_weights.get("text", 1/3)
    w_a = pso_weights.get("audio", 1/3)
    w_v = pso_weights.get("video", 1/3)
    fused = w_t * p_t_cal + w_a * p_a_cal + w_v * p_v_cal
    pred = int(fused.argmax())

    return {
        "text_probs": p_t,
        "audio_probs": p_a,
        "video_probs": p_v,
        "text_probs_cal": p_t_cal,
        "audio_probs_cal": p_a_cal,
        "video_probs_cal": p_v_cal,
        "fused_probs": fused,
        "prediction": pred,
        "label": "With A/H" if pred == 1 else "No A/H",
        "temperatures": pso_temps,
        "weights": pso_weights,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Full pipeline orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(video_path: Path, log_fn) -> dict:
    """
    Execute the full A/H detection pipeline on a recorded video.

    Parameters
    ----------
    video_path : Path to the recorded .avi file.
    log_fn     : callable(str) — logs a message to the UI sidebar.

    Returns
    -------
    dict with prediction results.
    """
    device = _resolve_device()
    log_fn(f"🖥️  Device: {device}")
    session_dir = video_path.parent

    # ── 1. Audio ─────────────────────────────────────────────────────────────
    audio_path = session_dir / "audio.wav"
    if audio_path.exists():
        log_fn("🔊 Step 1/6: Audio already recorded by microphone — skipping extraction")
    else:
        log_fn("🔊 Step 1/6: Extracting audio from video via ffmpeg ...")
        t0 = time.time()
        extract_audio_from_video(video_path, audio_path, sr=CFG["audio"]["sample_rate"])
        log_fn(f"   ✓ Audio extracted ({time.time() - t0:.1f}s)")

    # ── 2. Whisper transcription ─────────────────────────────────────────────
    log_fn("📝 Step 2/6: Transcribing audio with Whisper ...")
    t0 = time.time()
    transcript = transcribe_audio(audio_path, model_size=CFG["whisper"]["model_size"])
    log_fn(f"   ✓ Transcript ({time.time() - t0:.1f}s):")
    log_fn(f"   \"{transcript[:200]}{'...' if len(transcript) > 200 else ''}\"")

    # ── 3. YOLO segmentation ────────────────────────────────────────────────
    log_fn("🎯 Step 3/6: Running YOLO person segmentation ...")
    t0 = time.time()
    frames_dir = session_dir / "segmented_frames"
    yolo_path = REPO_ROOT / CFG["paths"]["yolo_model"]
    frame_paths = segment_frames_yolo(
        video_path, frames_dir, yolo_path,
        image_size=CFG["video"]["image_size"],
    )
    log_fn(f"   ✓ {len(frame_paths)} segmented frames ({time.time() - t0:.1f}s)")

    # Save an example segmented frame for visual inspection.
    if frame_paths:
        import shutil
        example_src = frame_paths[len(frame_paths) // 2]  # middle frame
        example_dst = session_dir / "example_segmented_frame.jpg"
        shutil.copy2(str(example_src), str(example_dst))
        log_fn(f"   📸 Example frame saved: {example_dst.name}")

    # ── 4. Extract embeddings ───────────────────────────────────────────────
    log_fn("🧠 Step 4/6: Extracting text embedding ...")
    t0 = time.time()
    text_emb = extract_text_embedding(transcript, device)
    log_fn(f"   ✓ Text embedding: shape={tuple(text_emb.shape)} ({time.time() - t0:.1f}s)")

    log_fn("🧠 Step 4b/6: Extracting audio embedding ...")
    t0 = time.time()
    audio_emb = extract_audio_embedding(audio_path, device)
    log_fn(f"   ✓ Audio embedding: shape={tuple(audio_emb.shape)} ({time.time() - t0:.1f}s)")

    log_fn("🧠 Step 4c/6: Extracting video embedding ...")
    t0 = time.time()
    video_emb = extract_video_embedding(frame_paths, device)
    log_fn(f"   ✓ Video embedding: shape={tuple(video_emb.shape)} ({time.time() - t0:.1f}s)")

    # ── 5. Late-fusion classification ───────────────────────────────────────
    log_fn("⚡ Step 6/6: Running late-fusion classifier ...")
    t0 = time.time()
    result = run_late_fusion(text_emb, audio_emb, video_emb, device)
    log_fn(f"   ✓ Classification done ({time.time() - t0:.1f}s)")

    # ── 6. Report ───────────────────────────────────────────────────────────
    log_fn("")
    log_fn("=" * 50)
    log_fn("📊 RESULTS")
    log_fn("=" * 50)
    w = result.get("weights", {})
    log_fn(f"  Text  raw   : No A/H={result['text_probs'][0]:.3f}  "
           f"With A/H={result['text_probs'][1]:.3f}")
    log_fn(f"  Text  cal   : No A/H={result['text_probs_cal'][0]:.3f}  "
           f"With A/H={result['text_probs_cal'][1]:.3f}  "
           f"(w={w.get('text', 0):.3f})")
    log_fn(f"  Audio raw   : No A/H={result['audio_probs'][0]:.3f}  "
           f"With A/H={result['audio_probs'][1]:.3f}")
    log_fn(f"  Audio cal   : No A/H={result['audio_probs_cal'][0]:.3f}  "
           f"With A/H={result['audio_probs_cal'][1]:.3f}  "
           f"(w={w.get('audio', 0):.3f})")
    log_fn(f"  Video raw   : No A/H={result['video_probs'][0]:.3f}  "
           f"With A/H={result['video_probs'][1]:.3f}")
    log_fn(f"  Video cal   : No A/H={result['video_probs_cal'][0]:.3f}  "
           f"With A/H={result['video_probs_cal'][1]:.3f}  "
           f"(w={w.get('video', 0):.3f})")
    log_fn(f"  Fused probs : No A/H={result['fused_probs'][0]:.3f}  "
           f"With A/H={result['fused_probs'][1]:.3f}")
    log_fn("")
    if result["prediction"] == 1:
        log_fn("  🔴  PREDICTION: A/H DETECTED")
    else:
        log_fn("  🟢  PREDICTION: No A/H detected")
    log_fn("=" * 50)

    # Save transcript for reference.
    (session_dir / "transcript.txt").write_text(transcript, encoding="utf-8")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Tkinter GUI
# ═══════════════════════════════════════════════════════════════════════════════

class DemoApp:
    """Main application window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BAH A/H Detection — Live Demo")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)

        # State.
        self.is_recording = False
        self.video_writer = None
        self.audio_frames: list[np.ndarray] = []
        self.current_session_dir: Path | None = None
        self.current_video_path: Path | None = None
        self.pipeline_running = False

        # Camera config.
        cam_cfg = CFG.get("camera", {})
        self.cam_index = cam_cfg.get("device_index", 0)
        self.cam_fps = cam_cfg.get("fps", 30)
        self.cam_w = cam_cfg.get("width", 640)
        self.cam_h = cam_cfg.get("height", 480)

        # Log message queue (thread-safe).
        self.log_queue: queue.Queue[str] = queue.Queue()

        self._build_ui()
        self._open_camera()
        self._tick_camera()
        self._tick_log_queue()

    # ── UI layout ────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Left panel: camera + controls.
        left = tk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Fixed-size container for the camera preview (prevents runaway growth).
        cam_frame = tk.Frame(left, width=self.cam_w, height=self.cam_h, bg="black")
        cam_frame.pack(fill=tk.NONE, expand=False, pady=(0, 5))
        cam_frame.pack_propagate(False)

        self.camera_label = tk.Label(cam_frame, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=5)

        self.record_btn = tk.Button(
            btn_frame, text="🔴  Start Recording", font=("Helvetica", 14, "bold"),
            bg="#e74c3c", fg="white", command=self._toggle_recording,
            height=2,
        )
        self.record_btn.pack(fill=tk.X, padx=5)

        self.status_label = tk.Label(
            left, text="Ready — press the button to start recording",
            font=("Helvetica", 11), anchor="w",
        )
        self.status_label.pack(fill=tk.X, padx=5)

        # Result banner.
        self.result_label = tk.Label(
            left, text="", font=("Helvetica", 18, "bold"),
            anchor="center", height=2,
        )
        self.result_label.pack(fill=tk.X, padx=5, pady=5)

        # Right panel: log sidebar.
        right = tk.Frame(self.root, width=450)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        right.pack_propagate(False)

        tk.Label(right, text="Pipeline Log", font=("Helvetica", 12, "bold")).pack(
            anchor="w", padx=5
        )

        self.log_text = scrolledtext.ScrolledText(
            right, wrap=tk.WORD, font=("Courier", 10), state=tk.DISABLED,
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="white",
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ── Camera ───────────────────────────────────────────────────────────────

    def _open_camera(self):
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
        self.cap.set(cv2.CAP_PROP_FPS, self.cam_fps)

    def _tick_camera(self):
        """Read one frame from the camera and update the preview."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Write to video file if recording.
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(frame)

                # Recording indicator overlay.
                if self.is_recording:
                    cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
                    cv2.putText(
                        frame, "REC", (50, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )

                # Convert BGR → RGB → PIL → Tk.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                # Resize to fixed preview size (prevents feedback loop).
                preview_w, preview_h = self.cam_w, self.cam_h
                scale = min(preview_w / img.width, preview_h / img.height)
                new_w = int(img.width * scale)
                new_h = int(img.height * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk  # keep reference
                self.camera_label.configure(image=imgtk)

        self.root.after(33, self._tick_camera)  # ~30 fps

    # ── Recording ────────────────────────────────────────────────────────────

    def _toggle_recording(self):
        if self.pipeline_running:
            return
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = RECORDINGS_DIR / ts
        session_dir.mkdir(parents=True, exist_ok=True)
        self.current_session_dir = session_dir

        video_path = session_dir / "recording.mp4"
        self.current_video_path = video_path

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_writer = cv2.VideoWriter(
            str(video_path), fourcc, self.cam_fps, (actual_w, actual_h)
        )

        # Start audio recording in a separate thread.
        self.audio_frames = []
        self._start_audio_recording()

        self.is_recording = True
        self.record_btn.configure(text="⏹  Stop Recording", bg="#2ecc71")
        self.status_label.configure(text="Recording ... press Stop when done")
        self.result_label.configure(text="", bg=self.root.cget("bg"))
        self._log("🎬 Recording started ...")

    def _start_audio_recording(self):
        """Start capturing audio from the default microphone."""
        import sounddevice as sd

        sr = CFG["audio"]["sample_rate"]
        self._audio_stream = sd.InputStream(
            samplerate=sr, channels=1, dtype="float32",
            callback=self._audio_callback,
        )
        self._audio_stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        self.audio_frames.append(indata.copy())

    def _stop_recording(self):
        self.is_recording = False

        # Stop video writer.
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        # Stop audio stream and save.
        if hasattr(self, "_audio_stream"):
            self._audio_stream.stop()
            self._audio_stream.close()

        # Save audio to WAV.
        if self.audio_frames and self.current_session_dir:
            import soundfile as sf
            audio_data = np.concatenate(self.audio_frames, axis=0)
            audio_path = self.current_session_dir / "audio.wav"
            sf.write(str(audio_path), audio_data, CFG["audio"]["sample_rate"])
            self._log(f"🔊 Audio saved: {audio_path.name}")

        self.record_btn.configure(text="🔴  Start Recording", bg="#e74c3c")
        self.status_label.configure(text="Recording saved. Running pipeline ...")
        self._log(f"🎬 Recording stopped → {self.current_video_path}")

        # Run the pipeline in a background thread.
        self.pipeline_running = True
        self.record_btn.configure(state=tk.DISABLED)
        threading.Thread(
            target=self._run_pipeline_thread, daemon=True
        ).start()

    # ── Pipeline thread ──────────────────────────────────────────────────────

    def _run_pipeline_thread(self):
        """Run the full pipeline in a background thread."""
        try:
            result = run_pipeline(self.current_video_path, log_fn=self._log_threadsafe)
            # Schedule UI update on the main thread.
            self.root.after(0, lambda: self._show_result(result))
        except Exception as e:
            self._log_threadsafe(f"❌ Pipeline error: {e}")
            import traceback
            self._log_threadsafe(traceback.format_exc())
        finally:
            self.root.after(0, self._pipeline_done)

    def _show_result(self, result: dict):
        """Update the result banner on the main thread."""
        if result["prediction"] == 1:
            self.result_label.configure(
                text="🔴  A/H DETECTED", bg="#e74c3c", fg="white",
            )
        else:
            self.result_label.configure(
                text="🟢  No A/H detected", bg="#2ecc71", fg="white",
            )

    def _pipeline_done(self):
        """Called on main thread after pipeline completes."""
        self.pipeline_running = False
        self.record_btn.configure(state=tk.NORMAL)
        self.status_label.configure(text="Ready — press the button to start recording")

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        """Append a message to the log sidebar (must be called from main thread)."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _log_threadsafe(self, msg: str):
        """Enqueue a log message from any thread; drained by _tick_log_queue."""
        self.log_queue.put(msg)

    def _tick_log_queue(self):
        """Drain the log queue and write messages to the sidebar (main thread)."""
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                self._log(msg)
            except queue.Empty:
                break
        self.root.after(100, self._tick_log_queue)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _on_close(self):
        """Release resources and close the window."""
        if self.is_recording:
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
            if hasattr(self, "_audio_stream"):
                self._audio_stream.stop()
                self._audio_stream.close()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app = DemoApp(root)
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    root.mainloop()
