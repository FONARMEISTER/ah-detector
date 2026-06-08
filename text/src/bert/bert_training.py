"""
BERT Text Classification Training Pipeline
============================================

Fine-tunes ``BertForSequenceClassification`` on interview transcripts
for binary A/H (Ambivalence/Hesitancy) detection.

Each sample is a transcript string paired with a binary label:
  - 0 : No A/H
  - 1 : With A/H

The pipeline uses the ``TextDataset`` / ``get_text_splits()`` loader from
``utils/load_dataset.py``, which reads video-level split files containing
``video_path,label,transcript`` rows.  Each row is one sample — no
video-level aggregation is needed.

Usage
-----
    cd text/src
    python bert_training.py
"""


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
    from pathlib import Path
    from collections import Counter

    import toml
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import BertForSequenceClassification, AutoTokenizer
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from tqdm.auto import tqdm

    # ── Repo root & imports ──────────────────────────────────────────────────
    REPO_ROOT = Path('/home3/iasarantsev').resolve()
    sys.path.insert(0, str(REPO_ROOT / "utils"))
    from load_dataset import get_text_splits

    # ── Config ───────────────────────────────────────────────────────────────
    CFG = toml.load('config.toml')

    MAX_LENGTH = CFG["text"]["max_length"]

    BERT_NAME = CFG["bert"]["model_name"]
    FREEZE_LAYERS = CFG["bert"]["freeze_layers"]

    EPOCHS = CFG["bert"]["training"]["epochs"]
    BATCH_SIZE = CFG["bert"]["training"]["batch_size"]
    LR = CFG["bert"]["training"]["learning_rate"]
    WD = CFG["bert"]["training"]["weight_decay"]
    PATIENCE = CFG["bert"]["training"]["early_stopping_patience"]
    GRAD_CLIP = CFG["training"]["grad_clip"]
    SEED = CFG["training"]["seed"]

    WEIGHTS = (
        REPO_ROOT
        / "text"
        / CFG["bert"]["output"]["weights_dir"]
        / CFG["bert"]["output"]["weights_name"]
    )

    # ── Seed ─────────────────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Device ───────────────────────────────────────────────────────────────
    DEVICE = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print("=" * 70)
    print("BERT Training Pipeline — BAH A/H Text Classification")
    print("=" * 70)
    print(f"Device        : {DEVICE}")
    print(f"Model         : {BERT_NAME}")
    print(f"Max length    : {MAX_LENGTH} tokens")
    print(f"Freeze layers : {FREEZE_LAYERS}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Epochs        : {EPOCHS}")
    print(f"LR            : {LR}  (encoder: {LR * 0.1})")
    print(f"Weight decay  : {WD}")
    print(f"Patience      : {PATIENCE}")
    print(f"Grad clip     : {GRAD_CLIP}")
    print(f"Seed          : {SEED}")
    print(f"Weights path  : {WEIGHTS}")
    print("=" * 70)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"\n[Tokenizer] Loading AutoTokenizer from {BERT_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    print(f"  Vocab size     : {tokenizer.vocab_size:,}")
    print(f"  Model max len  : {tokenizer.model_max_length}")
    print(f"  Padding token  : {tokenizer.pad_token}")
    print(f"  Using max_len  : {MAX_LENGTH}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\n[Dataset] Loading text splits ...")
    t0 = time.time()
    splits = get_text_splits(
        max_length=MAX_LENGTH,
        tokenizer=tokenizer,
    )
    print(f"[Dataset] Loaded in {time.time() - t0:.1f}s")

    # ── Dataset diagnostics ──────────────────────────────────────────────────
    for split_name in ("train", "val", "test"):
        ds = splits[split_name]
        labels = ds.data["label"].tolist()
        cnt = Counter(labels)
        n_total = len(labels)
        pct_pos = cnt.get(1, 0) / n_total * 100 if n_total > 0 else 0

        # Transcript length statistics
        transcript_lengths = ds.data["transcript"].str.split().str.len()
        avg_words = transcript_lengths.mean()
        max_words = transcript_lengths.max()
        min_words = transcript_lengths.min()

        print(
            f"  [{split_name:5s}]  {n_total:>5,} samples  |  "
            f"No A-H: {cnt.get(0, 0):,}  With A-H: {cnt.get(1, 0):,}  "
            f"({pct_pos:.1f}% positive)  |  "
            f"words: avg={avg_words:.0f}, min={min_words}, max={max_words}"
        )

    # ── Single-sample sanity check ───────────────────────────────────────────
    print("\n[Sanity] Loading single sample from train dataset ...")
    sample_encoded, sample_label = splits["train"][0]
    print(f"  Label           : {sample_label}")
    print(f"  Encoded keys    : {list(sample_encoded.keys())}")
    for k, v in sample_encoded.items():
        print(f"  {k:16s} : shape={v.shape}, dtype={v.dtype}")
    decoded = tokenizer.decode(sample_encoded["input_ids"], skip_special_tokens=True)
    print(f"  Decoded text    : {decoded[:120]}{'...' if len(decoded) > 120 else ''}")
    n_tokens = sample_encoded["attention_mask"].sum().item()
    print(f"  Active tokens   : {n_tokens}/{MAX_LENGTH}")

    # ── Collate function ─────────────────────────────────────────────────────
    def text_collate_fn(batch):
        """Collate tokenised text samples into batched tensors."""
        encoded_list, labels = zip(*batch)
        batch_encoded = {}
        for key in encoded_list[0].keys():
            batch_encoded[key] = torch.stack([e[key] for e in encoded_list])
        labels = torch.tensor(labels, dtype=torch.long)
        return batch_encoded, labels

    # ── Class-balanced sampler ───────────────────────────────────────────────
    train_labels = splits["train"].data["label"].tolist()
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
        f"[{class_weights[0] / class_weights.sum():.4f}, "
        f"{class_weights[1] / class_weights.sum():.4f}]"
    )

    # ── DataLoaders ──────────────────────────────────────────────────────────
    _num_workers = CFG["training"]["num_workers"]

    train_loader = DataLoader(
        splits["train"],
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=_num_workers,
        drop_last=True,
        collate_fn=text_collate_fn,
    )
    val_loader = DataLoader(
        splits["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=_num_workers,
        collate_fn=text_collate_fn,
    )
    test_loader = DataLoader(
        splits["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=_num_workers,
        collate_fn=text_collate_fn,
    )

    print(
        f'\n[DataLoader] Train: {len(splits["train"]):,} samples  |  '
        f'Val: {len(splits["val"]):,}  |  Test: {len(splits["test"]):,}'
    )
    print(
        f"[DataLoader] Train batches: {len(train_loader):,}  |  "
        f"Val: {len(val_loader):,}  |  Test: {len(test_loader):,}"
    )

    # ── Probe first batch ────────────────────────────────────────────────────
    print("\n[Probe] Fetching first train batch ...")
    t0 = time.time()
    first_enc, first_lb = next(iter(train_loader))
    batch_time = time.time() - t0
    for k, v in first_enc.items():
        print(f"  {k:16s} : shape={v.shape}, dtype={v.dtype}")
    print(f"  labels       : {first_lb.tolist()}")
    print(f"  batch time   : {batch_time:.2f}s")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"\n[Model] Loading BertForSequenceClassification from {BERT_NAME} ...")
    model = BertForSequenceClassification.from_pretrained(
        BERT_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    # ── Freeze early layers ──────────────────────────────────────────────────
    n_encoder_layers = len(model.bert.encoder.layer)
    print(f"[Model] BERT encoder has {n_encoder_layers} layers")
    print(
        f"[Model] Freezing: embeddings + first {FREEZE_LAYERS}/{n_encoder_layers} encoder layers"
    )
    print(
        f"[Model] Trainable: last {n_encoder_layers - FREEZE_LAYERS} encoder layers + "
        f"pooler + classifier head"
    )

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for i, layer in enumerate(model.bert.encoder.layer):
        if i < FREEZE_LAYERS:
            for param in layer.parameters():
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

    # Log model architecture summary
    print(f"\n[Model] Architecture summary:")
    print(f"  Hidden size    : {model.config.hidden_size}")
    print(f"  Num heads      : {model.config.num_attention_heads}")
    print(f"  Intermediate   : {model.config.intermediate_size}")
    print(f"  Hidden dropout : {model.config.hidden_dropout_prob}")
    print(f"  Attn dropout   : {model.config.attention_probs_dropout_prob}")
    print(f"  Max position   : {model.config.max_position_embeddings}")

    # ── Loss, Optimizer, Scheduler ───────────────────────────────────────────
    loss_weights = torch.tensor(
        class_weights / class_weights.sum(), dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    # Differential LRs: encoder layers get lower LR, classifier head gets full LR
    encoder_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "pooler" in name:
            classifier_params.append(param)
        else:
            encoder_params.append(param)

    LR_ENCODER = LR * 0.1
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": LR_ENCODER},
            {"params": classifier_params, "lr": LR},
        ],
        weight_decay=WD,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 1e-2
    )

    print(f"\n[Optimizer] AdamW with differential LRs:")
    print(f"  Encoder params  : {len(encoder_params)} tensors, lr={LR_ENCODER:.2e}")
    print(f"  Classifier params: {len(classifier_params)} tensors, lr={LR:.2e}")
    print(f"  Weight decay    : {WD}")
    print(f"[Scheduler] CosineAnnealingLR, T_max={EPOCHS}, eta_min={LR * 1e-2:.2e}")

    # ── Training loop ────────────────────────────────────────────────────────
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    PLOT_PATH = WEIGHTS.parent / "bert_training_curves.png"

    best_f1 = -1.0
    no_improve = 0

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

        for batch_idx, (encoded, labels) in enumerate(
            tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS} train", leave=True)
        ):
            bt0 = time.time()
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(**encoded)
            logits = outputs.logits
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
            for encoded, labels in tqdm(
                val_loader, desc=f"Ep {epoch}/{EPOCHS} val  ", leave=True
            ):
                encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
                labels = labels.to(DEVICE)

                outputs = model(**encoded)
                logits = outputs.logits
                va_loss += criterion(logits, labels).item() * len(labels)
                va_preds.extend(logits.argmax(1).cpu().tolist())
                va_labels.extend(labels.cpu().tolist())

        scheduler.step()

        # Metrics (sample-level — each transcript is one sample)
        tr_f1 = f1_score(tr_labels, tr_preds, average="macro", zero_division=0)
        va_f1 = f1_score(va_labels, va_preds, average="macro", zero_division=0)
        va_acc = accuracy_score(va_labels, va_preds)
        _tr_loss_avg = tr_loss / max(len(train_loader) * BATCH_SIZE, 1)
        _va_loss_avg = va_loss / max(len(va_labels), 1)

        history["tr_loss"].append(_tr_loss_avg)
        history["va_loss"].append(_va_loss_avg)
        history["tr_f1"].append(tr_f1)
        history["va_f1"].append(va_f1)
        history["va_acc"].append(va_acc)

        epoch_time = time.time() - epoch_start
        avg_batch = np.mean(batch_times) if batch_times else 0
        cur_lr_enc = optimizer.param_groups[0]["lr"]
        cur_lr_cls = optimizer.param_groups[1]["lr"]

        # Prediction distribution
        tr_pred_cnt = Counter(tr_preds)
        va_pred_cnt = Counter(va_preds)

        print(
            f"Epoch {epoch:>3}/{EPOCHS}  "
            f"tr_loss={_tr_loss_avg:.4f}  tr_f1={tr_f1:.4f}  "
            f"va_loss={_va_loss_avg:.4f}  va_f1={va_f1:.4f}  va_acc={va_acc:.4f}  "
            f"lr_enc={cur_lr_enc:.2e}  lr_cls={cur_lr_cls:.2e}  "
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
        for encoded, labels in tqdm(
            test_loader, desc="Test inference", leave=True
        ):
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            preds = model(**encoded).logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    test_pred_cnt = Counter(all_preds)
    test_label_cnt = Counter(all_labels)
    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    test_acc = accuracy_score(all_labels, all_preds)

    print(f"\n{'=' * 70}")
    print(f"Test Results — BERT ({BERT_NAME})")
    print(f"{'=' * 70}")
    print(f"[Test] Label distribution : {dict(test_label_cnt)}")
    print(f"[Test] Prediction dist.  : {dict(test_pred_cnt)}")
    print(f"[Test] Macro-F1  : {test_f1:.4f}")
    print(f"[Test] Accuracy  : {test_acc:.4f}")
    print()
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=["No A-H", "With A-H"],
            digits=4,
        )
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"BERT Training Summary")
    print(f"{'=' * 70}")
    print(f"  Model           : {BERT_NAME}")
    print(f"  Frozen layers   : {FREEZE_LAYERS}/{n_encoder_layers}")
    print(f"  Trainable params: {trainable_params:,} / {total_params:,}")
    print(f"  Best val F1     : {best_f1:.4f}")
    print(f"  Test F1         : {test_f1:.4f}")
    print(f"  Test accuracy   : {test_acc:.4f}")
    print(f"  Training time   : {total_time / 60:.1f} min")
    print(f"  Weights saved   : {WEIGHTS}")
    print(f"  Curves saved    : {PLOT_PATH}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
