"""
Unified Text Classification Training Pipeline (BERT / RoBERTa / DistilBERT)
===========================================================================

Single entry point that fine-tunes any of three transformer encoders on the
BAH interview-transcript dataset for binary A/H detection.

Usage
-----
    python text_training.py                 # uses [text].default_model
    python text_training.py bert
    python text_training.py roberta
    python text_training.py distilbert
"""


def plot_history(history: dict, save_path, model_key: str) -> None:
    """Save a 2-panel figure: Loss and Macro-F1 curves for train/val."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["tr_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["tr_loss"], "o-", label="Train loss")
    axes[0].plot(epochs, history["va_loss"], "s--", label="Val loss")
    axes[0].set_title(f"Loss per epoch ({model_key})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["tr_f1"], "o-", label="Train macro-F1")
    axes[1].plot(epochs, history["va_f1"], "s--", label="Val macro-F1")
    axes[1].plot(epochs, history["va_acc"], "^:", label="Val accuracy")
    axes[1].set_title(f"Metrics per epoch ({model_key})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved → {save_path}")


def _get_encoder_layers(model):
    """
    Return (embeddings_module, list_of_encoder_layers) regardless of backbone.

    BERT / RoBERTa expose ``base_model.encoder.layer``.
    DistilBERT exposes ``base_model.transformer.layer``.
    Both expose ``base_model.embeddings``.
    """
    base = model.base_model
    embeddings = base.embeddings
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        return embeddings, base.encoder.layer
    if hasattr(base, "transformer") and hasattr(base.transformer, "layer"):
        return embeddings, base.transformer.layer
    raise RuntimeError(
        "Unrecognised transformer architecture: neither "
        "base_model.encoder.layer nor base_model.transformer.layer found."
    )


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
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from tqdm.auto import tqdm

    # ── Repo root & imports ──────────────────────────────────────────────────
    REPO_ROOT = Path('/home3/iasarantsev').resolve()
    sys.path.insert(0, str(REPO_ROOT / "utils"))
    from load_dataset import get_text_splits

    # ── Config ───────────────────────────────────────────────────────────────
    CFG = toml.load("config.toml")

    SUPPORTED = ("bert", "roberta", "distilbert", "distilroberta_emotional")
    if len(sys.argv) > 1:
        MODEL_KEY = sys.argv[1].strip().lower()
    else:
        MODEL_KEY = str(CFG["text"].get("default_model", "distilbert")).lower()

    if MODEL_KEY not in SUPPORTED:
        raise SystemExit(
            f"Unsupported model '{MODEL_KEY}'. "
            f"Choose one of: {', '.join(SUPPORTED)}"
        )

    MODEL_CFG = CFG[MODEL_KEY]

    MAX_LENGTH = CFG["text"]["max_length"]

    MODEL_NAME = MODEL_CFG["model_name"]
    FREEZE_LAYERS = MODEL_CFG["freeze_layers"]
    LABEL_SMOOTH = float(MODEL_CFG.get("label_smoothing", 0.0))

    EPOCHS = MODEL_CFG["training"]["epochs"]
    BATCH_SIZE = MODEL_CFG["training"]["batch_size"]
    LR = MODEL_CFG["training"]["learning_rate"]
    WD = MODEL_CFG["training"]["weight_decay"]
    PATIENCE = MODEL_CFG["training"]["early_stopping_patience"]
    GRAD_CLIP = CFG["training"]["grad_clip"]
    SEED = CFG["training"]["seed"]

    WEIGHTS = (
        REPO_ROOT
        / "text"
        / MODEL_CFG["output"]["weights_dir"]
        / MODEL_CFG["output"]["weights_name"]
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
    print(f"Text Training Pipeline — BAH A/H Classification  [{MODEL_KEY}]")
    print("=" * 70)
    print(f"Device        : {DEVICE}")
    print(f"Model key     : {MODEL_KEY}")
    print(f"Model name    : {MODEL_NAME}")
    print(f"Max length    : {MAX_LENGTH} tokens")
    print(f"Freeze layers : {FREEZE_LAYERS}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Epochs        : {EPOCHS}")
    print(f"LR            : {LR}  (encoder: {LR * 0.1})")
    print(f"Weight decay  : {WD}")
    print(f"Label smooth  : {LABEL_SMOOTH}")
    print(f"Patience      : {PATIENCE}")
    print(f"Grad clip     : {GRAD_CLIP}")
    print(f"Seed          : {SEED}")
    print(f"Weights path  : {WEIGHTS}")
    print("=" * 70)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"\n[Tokenizer] Loading AutoTokenizer from {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  Vocab size     : {tokenizer.vocab_size:,}")
    print(f"  Model max len  : {tokenizer.model_max_length}")
    print(f"  Padding token  : {tokenizer.pad_token}")
    print(f"  Mask token     : {tokenizer.mask_token}")
    print(f"  Using max_len  : {MAX_LENGTH}")

    # ── Augmentation config ──────────────────────────────────────────────────
    aug_section = CFG.get("augmentation", {})
    AUGMENT_TRAIN = bool(aug_section.get("augment_train", False))
    AUG_CFG = {k: v for k, v in aug_section.items() if k != "augment_train"}
    print(
        f"\n[Augmentation] train={AUGMENT_TRAIN}  "
        f"mask_prob={AUG_CFG.get('text_mask_prob', 0.15)}  "
        f"drop_prob={AUG_CFG.get('text_drop_prob', 0.10)}"
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\n[Dataset] Loading text splits ...")
    t0 = time.time()
    splits = get_text_splits(
        max_length=MAX_LENGTH,
        tokenizer=tokenizer,
        augment_train=AUGMENT_TRAIN,
        aug_cfg=AUG_CFG,
        seed=SEED,
    )
    print(f"[Dataset] Loaded in {time.time() - t0:.1f}s")

    # ── Dataset diagnostics ──────────────────────────────────────────────────
    for split_name in ("train", "val", "test"):
        ds = splits[split_name]
        labels = ds.data["label"].tolist()
        cnt = Counter(labels)
        n_total = len(labels)
        pct_pos = cnt.get(1, 0) / n_total * 100 if n_total > 0 else 0

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
    print(f"\n[Model] Loading AutoModelForSequenceClassification from {MODEL_NAME} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    # ── Freeze early layers ──────────────────────────────────────────────────
    embeddings_mod, encoder_layers = _get_encoder_layers(model)
    n_encoder_layers = len(encoder_layers)
    print(f"[Model] Backbone has {n_encoder_layers} encoder layers")
    print(
        f"[Model] Freezing: embeddings + first {FREEZE_LAYERS}/{n_encoder_layers} encoder layers"
    )
    print(
        f"[Model] Trainable: last {n_encoder_layers - FREEZE_LAYERS} encoder layers + "
        f"classifier head"
    )

    for param in embeddings_mod.parameters():
        param.requires_grad = False
    for i, layer in enumerate(encoder_layers):
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
    cfg_obj = model.config
    print(f"\n[Model] Architecture summary:")
    print(f"  Hidden size    : {getattr(cfg_obj, 'hidden_size', '?')}")
    print(
        f"  Num heads      : "
        f"{getattr(cfg_obj, 'num_attention_heads', getattr(cfg_obj, 'n_heads', '?'))}"
    )
    print(
        f"  Num layers     : "
        f"{getattr(cfg_obj, 'num_hidden_layers', getattr(cfg_obj, 'n_layers', '?'))}"
    )
    print(
        f"  Dropout        : "
        f"{getattr(cfg_obj, 'hidden_dropout_prob', getattr(cfg_obj, 'dropout', '?'))}"
    )
    print(
        f"  Attn dropout   : "
        f"{getattr(cfg_obj, 'attention_probs_dropout_prob', getattr(cfg_obj, 'attention_dropout', '?'))}"
    )
    print(f"  Max position   : {getattr(cfg_obj, 'max_position_embeddings', '?')}")

    # ── Loss, Optimizer, Scheduler ───────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    print(
        f"\n[Loss] CrossEntropyLoss  label_smoothing={LABEL_SMOOTH}  "
        f"(class balancing handled by WeightedRandomSampler only)"
    )

    encoder_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "pre_classifier" in name:
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
    PLOT_PATH = WEIGHTS.parent / f"{MODEL_KEY}_training_curves.png"

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
    print(f"Starting training [{MODEL_KEY}] ...")
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
    plot_history(history, PLOT_PATH, MODEL_KEY)

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
    print(f"Test Results — {MODEL_KEY} ({MODEL_NAME})")
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
    print(f"Training Summary — {MODEL_KEY}")
    print(f"{'=' * 70}")
    print(f"  Model           : {MODEL_NAME}")
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
