def _aggregate_to_video_level(video_ids, frame_preds, frame_labels):
    """Aggregate frame-level predictions to video-level.

    For each video, the video-level prediction is 1 if *any* frame in that
    video was predicted as positive (A/H), otherwise 0.  The video-level
    ground-truth label is derived the same way (any positive frame → 1).

    Returns
    -------
    vid_preds : list[int]   – one prediction per video
    vid_labels : list[int]  – one ground-truth label per video
    """
    from collections import defaultdict

    vid_pred_map = defaultdict(list)
    vid_label_map = defaultdict(list)
    for vid, pred, label in zip(video_ids, frame_preds, frame_labels):
        vid_pred_map[vid].append(pred)
        vid_label_map[vid].append(label)

    vid_preds, vid_labels = [], []
    for vid in vid_pred_map:
        vid_preds.append(int(max(vid_pred_map[vid])))    # any-positive rule
        vid_labels.append(int(max(vid_label_map[vid])))  # any-positive rule
    return vid_preds, vid_labels


def _get_video_ids(dataset):
    """Extract the video folder (= video identity) for every frame in a
    SegmentedFrameDataset, preserving index order."""
    from pathlib import Path
    return [str(Path(row['frame_path']).parent) for row in dataset.data]


def plot_history(history: dict, save_path) -> None:
    """Save a 2-panel figure: Loss and Macro-F1 curves for train/val."""
    import matplotlib
    matplotlib.use('Agg')   # non-interactive backend — safe for scripts
    import matplotlib.pyplot as plt

    epochs = range(1, len(history['tr_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Loss ──────────────────────────────────────────────────────────────────
    axes[0].plot(epochs, history['tr_loss'], 'o-', label='Train loss')
    axes[0].plot(epochs, history['va_loss'], 's--', label='Val loss')
    axes[0].set_title('Loss per epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Macro F1 ──────────────────────────────────────────────────────────────
    axes[1].plot(epochs, history['tr_f1'],  'o-', label='Train macro-F1')
    axes[1].plot(epochs, history['va_f1'],  's--', label='Val macro-F1')
    axes[1].plot(epochs, history['va_acc'], '^:', label='Val accuracy')
    axes[1].set_title('Metrics per epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Training curves saved → {save_path}')


def main():
    import sys, random, platform
    from pathlib import Path

    import toml
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import transforms
    from transformers import ViTModel
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from tqdm.auto import tqdm

    # ── Repo root & load_dataset ──────────────────────────────────────────────────
    REPO_ROOT = Path('/home3/iasarantsev').resolve()
    sys.path.insert(0, str(REPO_ROOT / 'utils'))
    from load_dataset import get_segmented_frame_splits

    # ── Config ────────────────────────────────────────────────────────────────────
    CFG = toml.load('config.toml')
    IMAGE_SIZE = CFG['image']['image_size']          # 224
    IMG_MEAN   = CFG['image']['mean']
    IMG_STD    = CFG['image']['std']
    EPOCHS     = CFG['training']['epochs']
    BATCH_SIZE = CFG['training']['batch_size']
    LR         = CFG['training']['learning_rate']
    WD         = CFG['training']['weight_decay']
    PATIENCE   = CFG['training']['early_stopping_patience']
    VIT_NAME   = CFG['vit']['model_name']
    WEIGHTS    = Path(CFG['output']['weights_dir']) / CFG['output']['weights_name']

    # ── Device ────────────────────────────────────────────────────────────────────
    DEVICE = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cpu')
    )
    print(f'Device: {DEVICE}  |  Image size: {IMAGE_SIZE}  |  Batch: {BATCH_SIZE}')


    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),   # slightly larger then random crop
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),   # Cutout-style regularization
    ])

    eval_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])

    splits = get_segmented_frame_splits(
        image_size      = IMAGE_SIZE,
        train_transform = train_tf,
        eval_transform  = eval_tf,
        subsample_every  = 2,
    )

    # ── Class-balanced sampler ────────────────────────────────────────────────
    train_labels = [row['label'] for row in splits['train'].data]
    class_counts = np.bincount(train_labels)                          # [n_class0, n_class1]
    class_weights = 1.0 / class_counts.astype(np.float64)            # inverse-frequency
    sample_weights = torch.tensor([class_weights[l] for l in train_labels], dtype=torch.float64)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    print(f'Class counts — No A-H: {class_counts[0]:,}  |  With A-H: {class_counts[1]:,}')
    print(f'Class weights — No A-H: {class_weights[0]:.6f}  |  With A-H: {class_weights[1]:.6f}')

    # 'spawn' is required on macOS (default fork breaks MPS/CUDA); Linux uses fork by default
    _mp_ctx = 'spawn' if platform.system() == 'Darwin' else None

    # shuffle=False because sampler handles ordering; drop_last avoids uneven last batch
    train_loader = DataLoader(splits['train'], batch_size=BATCH_SIZE, sampler=sampler,  num_workers=4, drop_last=True, persistent_workers=True, multiprocessing_context=_mp_ctx)
    val_loader   = DataLoader(splits['val'],   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, multiprocessing_context=_mp_ctx)
    test_loader  = DataLoader(splits['test'],  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, multiprocessing_context=_mp_ctx)

    # ── Precompute video IDs for video-level evaluation ────────────────────
    val_video_ids  = _get_video_ids(splits['val'])
    test_video_ids = _get_video_ids(splits['test'])

    print(f'Train: {len(splits["train"]):,} frames  |  Val: {len(splits["val"]):,}  |  Test: {len(splits["test"]):,}')
    n_val_videos  = len(set(val_video_ids))
    n_test_videos = len(set(test_video_ids))
    print(f'Val videos: {n_val_videos}  |  Test videos: {n_test_videos}')
    print(f'Train batches: {len(train_loader):,}  |  Val: {len(val_loader):,}  |  Test: {len(test_loader):,}')


    # Full ViT encoder — all parameters trainable
    encoder = ViTModel.from_pretrained(VIT_NAME).to(DEVICE)

    # Probe embed dim via pooler_output (CLS token after pooler, shape [1, hidden_size])
    with torch.no_grad():
        _d = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        EMBED_DIM = encoder(pixel_values=_d).pooler_output.shape[-1]
    
    head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(EMBED_DIM, 2),
    ).to(DEVICE)
    loss_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    LR_ENCODER = LR * 0.1
    optimizer = torch.optim.AdamW([
        {'params': encoder.parameters(), 'lr': LR_ENCODER},
        {'params': head.parameters(),    'lr': LR},
    ], weight_decay=WD)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 1e-2)

    total_params     = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in head.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f'Encoder embed dim: {EMBED_DIM}  |  Trainable params: {trainable_params:,} / {total_params:,}')


    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    PLOT_PATH = WEIGHTS.parent / 'training_curves.png'

    best_f1    = -1.0
    no_improve = 0
    GRAD_CLIP  = 1.0

    # ── History buffers ───────────────────────────────────────────────────────
    history = {'tr_loss': [], 'va_loss': [], 'tr_f1': [], 'va_f1': [], 'va_acc': []}

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────────
        encoder.train()
        head.train()
        tr_loss, tr_preds, tr_labels = 0.0, [], []
        for imgs, labels in tqdm(train_loader, desc=f'Ep {epoch}/{EPOCHS} train', leave=True):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            feats = encoder(pixel_values=imgs).pooler_output
            logits = head(feats)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()),
                max_norm=GRAD_CLIP,
            )
            optimizer.step()
            tr_loss  += loss.item() * len(labels)
            tr_preds.extend(logits.argmax(1).cpu().tolist())
            tr_labels.extend(labels.cpu().tolist())

        # ── Val ───────────────────────────────────────────────────────────────────
        encoder.eval()
        head.eval()
        va_loss, va_preds, va_labels = 0.0, [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f'Ep {epoch}/{EPOCHS} val  ', leave=True):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                feats  = encoder(pixel_values=imgs).pooler_output
                logits = head(feats)
                va_loss += criterion(logits, labels).item() * len(labels)
                va_preds.extend(logits.argmax(1).cpu().tolist())
                va_labels.extend(labels.cpu().tolist())

        scheduler.step()

        # Train metrics stay frame-level (matches the training objective)
        tr_f1  = f1_score(tr_labels, tr_preds, average='macro', zero_division=0)

        # Val metrics: aggregate to video-level (any positive frame → video=1)
        va_vid_preds, va_vid_labels = _aggregate_to_video_level(
            val_video_ids, va_preds, va_labels,
        )
        va_f1  = f1_score(va_vid_labels, va_vid_preds, average='macro', zero_division=0)
        va_acc = accuracy_score(va_vid_labels, va_vid_preds)
        _tr_loss_avg = tr_loss / len(train_loader.dataset)
        _va_loss_avg = va_loss / len(val_loader.dataset)

        history['tr_loss'].append(_tr_loss_avg)
        history['va_loss'].append(_va_loss_avg)
        history['tr_f1'].append(tr_f1)
        history['va_f1'].append(va_f1)
        history['va_acc'].append(va_acc)

        print(f'Epoch {epoch:>3}  tr_loss={_tr_loss_avg:.4f}  tr_f1={tr_f1:.4f}'
              f'  va_loss={_va_loss_avg:.4f}  va_f1(video)={va_f1:.4f}  va_acc(video)={va_acc:.4f}')

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                'encoder': encoder.state_dict(),
                'head':    head.state_dict(),
            }, WEIGHTS)
            print(f'  ✓ New best val F1={best_f1:.4f} — saved {WEIGHTS}')
            no_improve = 0
        else:
            no_improve += 1
            if PATIENCE > 0 and no_improve >= PATIENCE:
                print(f'Early stopping at epoch {epoch}')
                break

    print(f'\nTraining done. Best val F1 = {best_f1:.4f}')
    plot_history(history, PLOT_PATH)


    ckpt = torch.load(WEIGHTS, map_location=DEVICE)
    encoder.load_state_dict(ckpt['encoder'])
    head.load_state_dict(ckpt['head'])
    encoder.eval()
    head.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc='Test inference', leave=True):
            imgs   = imgs.to(DEVICE)
            feats  = encoder(pixel_values=imgs).pooler_output
            preds  = head(feats).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Aggregate frame-level predictions to video-level
    test_vid_preds, test_vid_labels = _aggregate_to_video_level(
        test_video_ids, all_preds, all_labels,
    )
    print('\n── Video-level Test Results ──')
    print(classification_report(test_vid_labels, test_vid_preds, target_names=['No A-H', 'With A-H'], digits=4))

if __name__ == '__main__':
    main()