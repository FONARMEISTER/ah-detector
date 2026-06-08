"""
Swin-Small Video-Clip Training Pipeline
========================================

Trains ``SwinForImageClassification`` (microsoft/swin-tiny-patch4-window7-224)
on sliding-window clips of segmented frames for binary A/H detection.

Clip-level labelling
--------------------
Same as ViViT: ``label_agg="any"`` — a clip is labelled **1 (With A/H)** if
*at least one* of its constituent frames carries label 1.

Usage
-----
    cd video/src
    python swin_training.py
"""


def _aggregate_to_video_level(video_ids, clip_preds, clip_labels):
    """Aggregate clip-level predictions to video-level.

    For each video, the video-level prediction is 1 if *any* clip in that
    video was predicted as positive (A/H), otherwise 0.  The video-level
    ground-truth label is derived the same way (any positive clip → 1).

    Returns
    -------
    vid_preds : list[int]   – one prediction per video
    vid_labels : list[int]  – one ground-truth label per video
    """
    from collections import defaultdict

    vid_pred_map = defaultdict(list)
    vid_label_map = defaultdict(list)
    for vid, pred, label in zip(video_ids, clip_preds, clip_labels):
        vid_pred_map[vid].append(pred)
        vid_label_map[vid].append(label)

    vid_preds, vid_labels = [], []
    for vid in vid_pred_map:
        vid_preds.append(int(max(vid_pred_map[vid])))    # any-positive rule
        vid_labels.append(int(max(vid_label_map[vid])))  # any-positive rule
    return vid_preds, vid_labels


def _get_clip_video_ids(dataset):
    """Extract the video folder (= video identity) for every clip in a
    SegmentedVideoClipDataset, preserving index order."""
    return [clip[0] for clip in dataset.clips]


def plot_history(history: dict, save_path) -> None:
    """Save a 2-panel figure: Loss and Macro-F1 curves for train/val."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["tr_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["tr_loss"], "o-", label="Train loss")
    axes[0].plot(epochs, history["va_loss"], "s--", label="Val loss")
    axes[0].set_title("Loss per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["tr_f1"], "o-", label="Train macro-F1")
    axes[1].plot(epochs, history["va_f1"], "s--", label="Val macro-F1")
    axes[1].plot(epochs, history["va_acc"], "^:", label="Val accuracy")
    axes[1].set_title("Metrics per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved → {save_path}")


def main():
    import sys
    import time
    import platform
    from pathlib import Path
    from collections import Counter

    import toml
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import SwinForImageClassification, AutoImageProcessor
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from tqdm.auto import tqdm

    # ── Repo root & imports ──────────────────────────────────────────────────
    REPO_ROOT = Path('/home3/iasarantsev').resolve()
    sys.path.insert(0, str(REPO_ROOT / "utils"))
    from load_dataset import get_segmented_video_clip_splits

    # ── Config ───────────────────────────────────────────────────────────────
    CFG = toml.load('config.toml')

    IMAGE_SIZE = CFG["image"]["image_size"]  # 224
    # NOTE: IMG_MEAN / IMG_STD are not used — AutoImageProcessor applies
    # its own normalisation stats automatically.

    SWIN_NAME = CFG["swin"]["model_name"]
    CLIP_LEN = CFG["swin"]["clip_len"]
    CLIP_STRIDE = CFG["swin"]["clip_stride"]
    LABEL_AGG = CFG["swin"]["label_agg"]
    FREEZE_STAGES = CFG["swin"]["freeze_stages"]

    EPOCHS = CFG["swin"]["training"]["epochs"]
    BATCH_SIZE = CFG["swin"]["training"]["batch_size"]
    LR = CFG["swin"]["training"]["learning_rate"]
    WD = CFG["swin"]["training"]["weight_decay"]
    PATIENCE = CFG["swin"]["training"]["early_stopping_patience"]

    WEIGHTS = (
        REPO_ROOT
        / "video"
        / CFG["swin"]["output"]["weights_dir"]
        / CFG["swin"]["output"]["weights_name"]
    )

    # ── Device ───────────────────────────────────────────────────────────────
    DEVICE = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print("=" * 70)
    print("Swin-Small Training Pipeline — BAH A/H Detection")
    print("=" * 70)
    print(f"Device        : {DEVICE}")
    print(f"Model         : {SWIN_NAME}")
    print(f"Image size    : {IMAGE_SIZE}")
    print(f"Clip length   : {CLIP_LEN} frames")
    print(f"Clip stride   : {CLIP_STRIDE} frames  (overlap={CLIP_LEN - CLIP_STRIDE})")
    print(f"Label agg     : {LABEL_AGG}")
    print(f"Freeze stages : {FREEZE_STAGES}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Epochs        : {EPOCHS}")
    print(f"LR            : {LR}  (backbone: {LR * 0.1})")
    print(f"Weight decay  : {WD}")
    print(f"Patience      : {PATIENCE}")
    print(f"Weights path  : {WEIGHTS}")
    print("=" * 70)

    # ── Image processor for normalisation ─────────────────────────────────────
    processor = AutoImageProcessor.from_pretrained(SWIN_NAME)
    print(f"\n[Processor] AutoImageProcessor loaded from {SWIN_NAME}")
    print(f"  size       : {processor.size}")
    print(f"  do_resize  : {processor.do_resize}")
    print(f"  do_normalize: {processor.do_normalize}")
    if hasattr(processor, "image_mean"):
        print(f"  image_mean : {processor.image_mean}")
    if hasattr(processor, "image_std"):
        print(f"  image_std  : {processor.image_std}")

    # ── Dataset (clips of segmented frames) ──────────────────────────────────
    print("\n[Dataset] Loading segmented video clip splits ...")
    t0 = time.time()
    splits = get_segmented_video_clip_splits(
        image_size=IMAGE_SIZE,
        clip_len=CLIP_LEN,
        clip_stride=CLIP_STRIDE,
        pad_last=True,
        label_agg=LABEL_AGG,
    )
    print(f"[Dataset] Loaded in {time.time() - t0:.1f}s")

    # ── Dataset diagnostics ──────────────────────────────────────────────────
    for split_name in ("train", "val", "test"):
        ds = splits[split_name]
        labels = [clip[1] for clip in ds.clips]
        cnt = Counter(labels)
        n_total = len(labels)
        pct_pos = cnt.get(1, 0) / n_total * 100 if n_total > 0 else 0
        folders = set(clip[0] for clip in ds.clips)
        clip_lengths = [len(clip[2]) for clip in ds.clips]
        full_clips = sum(1 for cl in clip_lengths if cl == CLIP_LEN)
        padded_clips = sum(1 for cl in clip_lengths if cl < CLIP_LEN)
        print(
            f"  [{split_name:5s}]  {n_total:>5,} clips  |  "
            f"No A-H: {cnt.get(0, 0):,}  With A-H: {cnt.get(1, 0):,}  "
            f"({pct_pos:.1f}% positive)  |  "
            f"{len(folders)} videos  |  "
            f"full: {full_clips}, padded: {padded_clips}"
        )

    # ── Single-sample sanity check ───────────────────────────────────────────
    print("\n[Sanity] Loading single sample from train dataset ...")
    sample_clip, sample_label = splits["train"][0]
    print(
        f"  Raw clip tensor : shape={sample_clip.shape}, "
        f"dtype={sample_clip.dtype}, "
        f"min={sample_clip.min():.3f}, max={sample_clip.max():.3f}"
    )
    print(f"  Label           : {sample_label}")
    nonzero_frames = sum(
        1
        for t in range(sample_clip.shape[0])
        if sample_clip[t].abs().sum() > 0
    )
    print(f"  Non-zero frames : {nonzero_frames}/{sample_clip.shape[0]}")
    if nonzero_frames == 0:
        print("  ⚠ WARNING: All frames are zero! Check that SegmentedFrames exist.")

    # ── Collate function ─────────────────────────────────────────────────────
    def swin_collate_fn(batch):
        """
        Collate clips for Swin (image-level model).

        Each item is (clip_tensor, label) where clip_tensor has shape
        (clip_len, C, H, W) with values in [0, 1].

        We flatten all frames across the batch into a single list, run them
        through the image processor, then reshape back to (B, T, C, H, W).
        """
        clips, labels = zip(*batch)
        B = len(clips)
        T = clips[0].shape[0]

        # Flatten all frames: list of (H, W, C) uint8 numpy arrays
        all_frames = []
        for clip in clips:
            # clip: (T, C, H, W) float [0, 1]
            np_clip = (clip.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            for t in range(np_clip.shape[0]):
                all_frames.append(np_clip[t])

        # Process all frames at once
        processed = processor(all_frames, return_tensors="pt")
        # pixel_values: (B*T, C, H, W)
        pixel_values = processed["pixel_values"]
        # Reshape to (B, T, C, H, W)
        C_out, H_out, W_out = pixel_values.shape[1:]
        pixel_values = pixel_values.view(B, T, C_out, H_out, W_out)

        labels = torch.tensor(labels, dtype=torch.long)
        return pixel_values, labels

    # ── Class-balanced sampler ───────────────────────────────────────────────
    train_labels = [clip[1] for clip in splits["train"].clips]
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = 1.0 / class_counts.astype(np.float64)
    sample_weights = torch.tensor(
        [class_weights[l] for l in train_labels], dtype=torch.float64
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    print(
        f"\n[Sampler] Class counts — No A-H: {class_counts[0]:,}  |  "
        f"With A-H: {class_counts[1]:,}"
    )
    print(
        f"[Sampler] Class weights — No A-H: {class_weights[0]:.6f}  |  "
        f"With A-H: {class_weights[1]:.6f}"
    )
    print(
        f"[Loss]   Normalised weights → "
        f"[{class_weights[0]/class_weights.sum():.4f}, "
        f"{class_weights[1]/class_weights.sum():.4f}]"
    )

    # ── DataLoaders ──────────────────────────────────────────────────────────
    _mp_ctx = "spawn" if platform.system() == "Darwin" else None

    train_loader = DataLoader(
        splits["train"],
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        drop_last=True,
        persistent_workers=True,
        multiprocessing_context=_mp_ctx,
        collate_fn=swin_collate_fn,
    )
    val_loader = DataLoader(
        splits["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context=_mp_ctx,
        collate_fn=swin_collate_fn,
    )
    test_loader = DataLoader(
        splits["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context=_mp_ctx,
        collate_fn=swin_collate_fn,
    )

    # ── Precompute video IDs for video-level evaluation ────────────────────
    val_video_ids  = _get_clip_video_ids(splits['val'])
    test_video_ids = _get_clip_video_ids(splits['test'])

    print(
        f'\n[DataLoader] Train: {len(splits["train"]):,} clips  |  '
        f'Val: {len(splits["val"]):,}  |  Test: {len(splits["test"]):,}'
    )
    n_val_videos  = len(set(val_video_ids))
    n_test_videos = len(set(test_video_ids))
    print(f"[DataLoader] Val videos: {n_val_videos}  |  Test videos: {n_test_videos}")
    print(
        f"[DataLoader] Train batches: {len(train_loader):,}  |  "
        f"Val: {len(val_loader):,}  |  Test: {len(test_loader):,}"
    )

    # ── Probe first batch ────────────────────────────────────────────────────
    print("\n[Probe] Fetching first train batch (this tests collate + processor) ...")
    t0 = time.time()
    first_pv, first_lb = next(iter(train_loader))
    batch_time = time.time() - t0
    print(
        f"  pixel_values : shape={first_pv.shape}, dtype={first_pv.dtype}, "
        f"min={first_pv.min():.3f}, max={first_pv.max():.3f}"
    )
    print(f"  labels       : {first_lb.tolist()}")
    print(f"  batch time   : {batch_time:.2f}s")
    expected_shape = (BATCH_SIZE, CLIP_LEN, 3, IMAGE_SIZE, IMAGE_SIZE)
    if first_pv.shape != expected_shape:
        print(f"  ⚠ WARNING: Expected shape {expected_shape}, got {tuple(first_pv.shape)}")
        print(f"    This may be fine if the processor resizes to a different resolution.")
    else:
        print(f"  ✓ Shape matches expected {expected_shape}")

    # ── Model: Swin + temporal mean-pooling wrapper ──────────────────────────
    # Swin is an image model, so we wrap it to process clips frame-by-frame
    # and average the features across the temporal dimension.

    class SwinVideoClassifier(nn.Module):
        """
        Wraps ``SwinForImageClassification`` for video-clip classification.

        For each clip of T frames:
        1. Each frame is passed through the Swin backbone independently.
        2. The pooled output features (before the classifier head) are
           averaged across the T frames (temporal mean-pooling).
        3. The averaged feature vector is passed through a new 2-class head.
        """

        def __init__(self, model_name: str, num_labels: int = 2):
            super().__init__()
            self.swin = SwinForImageClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
            # Replace the classifier head with our own (after temporal pooling)
            hidden_size = self.swin.config.hidden_size  # 768 for swin-small
            self.swin.classifier = nn.Identity()  # disable original head
            self.classifier = nn.Linear(hidden_size, num_labels)

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            pixel_values : (B, T, C, H, W)

            Returns
            -------
            logits : (B, num_labels)
            """
            B, T, C, H, W = pixel_values.shape
            # Flatten batch and time: (B*T, C, H, W)
            flat = pixel_values.reshape(B * T, C, H, W)
            # Forward through Swin backbone
            outputs = self.swin(pixel_values=flat)
            # outputs.logits is actually the identity output = pooled features
            # shape: (B*T, hidden_size)
            features = outputs.logits
            # Reshape back: (B, T, hidden_size)
            features = features.view(B, T, -1)
            # Temporal mean-pooling: (B, hidden_size)
            pooled = features.mean(dim=1)
            # Classify
            logits = self.classifier(pooled)
            return logits

    print("\n[Model] Loading SwinVideoClassifier ...")
    model = SwinVideoClassifier(SWIN_NAME, num_labels=2).to(DEVICE)

    # ── Freeze early stages ──────────────────────────────────────────────────
    # Swin-Small has: swin.embeddings, swin.encoder.layers[0..3]
    # Each "layer" is a Swin stage. We freeze embeddings + first N stages.
    swin_backbone = model.swin.swin
    n_stages = len(swin_backbone.encoder.layers)
    print(f"[Model] Swin encoder has {n_stages} stages")
    print(
        f"[Model] Freezing: embeddings + first {FREEZE_STAGES}/{n_stages} stages"
    )
    print(
        f"[Model] Trainable: last {n_stages - FREEZE_STAGES} stages + "
        f"layernorm + classifier head"
    )

    for param in swin_backbone.embeddings.parameters():
        param.requires_grad = False
    for i, stage in enumerate(swin_backbone.encoder.layers):
        if i < FREEZE_STAGES:
            for param in stage.parameters():
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_params = total_params - trainable_params
    print(
        f"[Model] Parameters: {trainable_params:,} trainable  |  "
        f"{frozen_params:,} frozen  |  {total_params:,} total  "
        f"({trainable_params / total_params * 100:.1f}% trainable)"
    )

    # ── Loss, Optimizer, Scheduler ───────────────────────────────────────────
    loss_weights = torch.tensor(
        class_weights / class_weights.sum(), dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    # Differential LRs: backbone gets lower LR, classifier head gets full LR
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    LR_BACKBONE = LR * 0.1
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": classifier_params, "lr": LR},
        ],
        weight_decay=WD,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 1e-2
    )

    # ── Training loop ────────────────────────────────────────────────────────
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    PLOT_PATH = WEIGHTS.parent / "swin_training_curves.png"

    best_f1 = -1.0
    no_improve = 0
    GRAD_CLIP = 1.0

    history = {
        "tr_loss": [],
        "va_loss": [],
        "tr_f1": [],
        "va_f1": [],
        "va_acc": [],
    }

    print("\n" + "=" * 70)
    print("Starting training ...")
    print("=" * 70)
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_preds, tr_labels = 0.0, [], []
        batch_times = []

        for batch_idx, (pixel_values, labels) in enumerate(
            tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS} train", leave=True)
        ):
            bt0 = time.time()
            pixel_values = pixel_values.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(pixel_values)
            loss = criterion(logits, labels)

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_CLIP
            )
            optimizer.step()

            tr_loss += loss.item() * len(labels)
            tr_preds.extend(logits.argmax(1).cpu().tolist())
            tr_labels.extend(labels.cpu().tolist())
            batch_times.append(time.time() - bt0)

            # Log first batch of first epoch in detail
            if epoch == 1 and batch_idx == 0:
                print(
                    f"\n  [Epoch 1, Batch 0] loss={loss.item():.4f}  "
                    f"grad_norm={grad_norm:.4f}  "
                    f"logits_sample={logits[0].detach().cpu().tolist()}  "
                    f"preds={logits.argmax(1).cpu().tolist()}  "
                    f"labels={labels.cpu().tolist()}"
                )

        # ── Val ──────────────────────────────────────────────────────────────
        model.eval()
        va_loss, va_preds, va_labels = 0.0, [], []

        with torch.no_grad():
            for pixel_values, labels in tqdm(
                val_loader, desc=f"Ep {epoch}/{EPOCHS} val  ", leave=True
            ):
                pixel_values = pixel_values.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(pixel_values)
                va_loss += criterion(logits, labels).item() * len(labels)
                va_preds.extend(logits.argmax(1).cpu().tolist())
                va_labels.extend(labels.cpu().tolist())

        scheduler.step()

        # Train metrics stay clip-level (matches the training objective)
        tr_f1 = f1_score(tr_labels, tr_preds, average="macro", zero_division=0)

        # Val metrics: aggregate to video-level (any positive clip → video=1)
        va_vid_preds, va_vid_labels = _aggregate_to_video_level(
            val_video_ids, va_preds, va_labels,
        )
        va_f1 = f1_score(va_vid_labels, va_vid_preds, average="macro", zero_division=0)
        va_acc = accuracy_score(va_vid_labels, va_vid_preds)
        _tr_loss_avg = tr_loss / max(len(train_loader) * BATCH_SIZE, 1)
        _va_loss_avg = va_loss / max(len(va_labels), 1)

        history["tr_loss"].append(_tr_loss_avg)
        history["va_loss"].append(_va_loss_avg)
        history["tr_f1"].append(tr_f1)
        history["va_f1"].append(va_f1)
        history["va_acc"].append(va_acc)

        epoch_time = time.time() - epoch_start
        avg_batch = np.mean(batch_times) if batch_times else 0
        cur_lr = optimizer.param_groups[0]["lr"]

        # Train prediction distribution
        tr_pred_cnt = Counter(tr_preds)
        va_pred_cnt = Counter(va_preds)

        print(
            f"Epoch {epoch:>3}/{EPOCHS}  "
            f"tr_loss={_tr_loss_avg:.4f}  tr_f1={tr_f1:.4f}  "
            f"va_loss={_va_loss_avg:.4f}  va_f1(video)={va_f1:.4f}  va_acc(video)={va_acc:.4f}  "
            f"lr={cur_lr:.2e}  "
            f"time={epoch_time:.0f}s ({avg_batch:.2f}s/batch)"
        )
        print(
            f"         train preds: {dict(tr_pred_cnt)}  |  "
            f"val preds: {dict(va_pred_cnt)}"
        )

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), WEIGHTS)
            print(f"  ✓ New best val F1={best_f1:.4f} — saved {WEIGHTS}")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  ✗ No improvement ({no_improve}/{PATIENCE})")
            if PATIENCE > 0 and no_improve >= PATIENCE:
                print(f"  ⛔ Early stopping at epoch {epoch}")
                break

    total_time = time.time() - train_start
    print(f"\n{'=' * 70}")
    print(f"Training done in {total_time / 60:.1f} min. Best val F1 = {best_f1:.4f}")
    print(f"{'=' * 70}")
    plot_history(history, PLOT_PATH)

    # ── Test evaluation ──────────────────────────────────────────────────────
    print("\n[Test] Loading best checkpoint and evaluating ...")
    model.load_state_dict(
        torch.load(WEIGHTS, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for pixel_values, labels in tqdm(
            test_loader, desc="Test inference", leave=True
        ):
            pixel_values = pixel_values.to(DEVICE)
            preds = model(pixel_values).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Aggregate clip-level predictions to video-level
    test_vid_preds, test_vid_labels = _aggregate_to_video_level(
        test_video_ids, all_preds, all_labels,
    )
    test_pred_cnt = Counter(test_vid_preds)
    test_label_cnt = Counter(test_vid_labels)
    print(f"\n── Video-level Test Results ──")
    print(f"[Test] Label distribution : {dict(test_label_cnt)}")
    print(f"[Test] Prediction dist.  : {dict(test_pred_cnt)}")
    print(
        classification_report(
            test_vid_labels,
            test_vid_preds,
            target_names=["No A-H", "With A-H"],
            digits=4,
        )
    )


if __name__ == "__main__":
    main()
