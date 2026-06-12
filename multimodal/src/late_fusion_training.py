"""
Late-Fusion Multimodal Training Pipeline (cache-driven)
========================================================

Trains an INDEPENDENT MLP classifier per modality on its pre-extracted
embeddings, then fuses their predictions *after* the fact (late fusion).

Why late fusion here
--------------------
With only ~1500 training videos, joint architectures such as cross-attention
fusion overfit immediately: the number of fusion parameters dwarfs the
sample count.  Late fusion keeps each per-modality head small and trained on
its own clean signal, then combines only the 2-class probability vectors.

Modalities used
---------------
  * Text   : DistilBERT/DistilRoBERTa embeddings (cached → ``text_embs_*.pt``)
  * Audio  : wav2vec2-emotional embeddings       (cached → ``audio_embs_*.pt``)
  * Video  : Swin-Tiny  embeddings               (cached → ``video_embs_*.pt``)

All three caches were produced by ``multimodal/src/extract_{text,audio,video}.py``
in a multi-view format ``(N, K, dim)``.  Per-modality MLPs randomly sample
one view per modality per step at train time; val/test always use view 0.

Pipeline
--------
                ┌─────────────────────────────────────────────┐
                │  per-modality cache  (N, K, dim) on disk    │
                └────────────┬──────────────┬──────────────┬───┘
                             │              │              │
                  ┌──────────▼─┐  ┌─────────▼──┐  ┌────────▼───┐
                  │  Text MLP  │  │  Audio MLP │  │  Video MLP │
                  │  early-fit │  │  early-fit │  │  early-fit │
                  │  (z-score) │  │  (z-score) │  │  (z-score) │
                  └──────┬─────┘  └─────┬──────┘  └─────┬──────┘
                         │ softmax      │ softmax       │ softmax
                         ▼              ▼               ▼
                   p_text(2)       p_audio(2)      p_video(2)
                                       │
                                       ▼
                ┌──────────────────────────────────────────────┐
                │  Late-fusion combiner                        │
                │  • mean averaging                            │
                │  • weighted averaging (val-tuned weights)    │
                │  • meta-learner (logistic regression stacker)│
                └──────────────────────────────────────────────┘

Outputs / artifacts
-------------------
  * ``multimodal/weights/late_fusion_<modality>_best.pth``  — per-modality MLP
  * ``multimodal/weights/late_fusion_<modality>.scaler.joblib`` — its StandardScaler
  * ``multimodal/weights/late_fusion_meta.joblib``           — stacking meta-clf
  * ``multimodal/weights/late_fusion_training_curves.png``   — combined curves
  * Console: per-modality val/test scores PLUS all three fusion-strategy
    scores side-by-side.

Usage
-----
    python late_fusion_training.py

    CACHE_DIR=/path/to/cache python late_fusion_training.py
"""

import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


# ── Repo root ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path("/home3/iasarantsev").resolve()
sys.path.insert(0, str(REPO_ROOT / "utils"))


# ── Plot helper ───────────────────────────────────────────────────────────────


def plot_history(histories: Dict[str, dict], save_path: Path) -> None:
    """Plot per-modality train/val curves on a single figure (one row per modality)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_mods = len(histories)
    fig, axes = plt.subplots(n_mods, 2, figsize=(12, 4 * n_mods), squeeze=False)
    for r, (mod, hist) in enumerate(histories.items()):
        epochs = range(1, len(hist["tr_loss"]) + 1)
        axes[r][0].plot(epochs, hist["tr_loss"], "o-", label="Train loss")
        axes[r][0].plot(epochs, hist["va_loss"], "s--", label="Val loss")
        axes[r][0].set_title(f"[{mod}] Loss per epoch")
        axes[r][0].set_xlabel("Epoch")
        axes[r][0].set_ylabel("Loss")
        axes[r][0].legend()
        axes[r][0].grid(True, alpha=0.3)

        axes[r][1].plot(epochs, hist["tr_f1"], "o-", label="Train macro-F1")
        axes[r][1].plot(epochs, hist["va_f1"], "s--", label="Val macro-F1")
        axes[r][1].plot(epochs, hist["va_acc"], "^:", label="Val accuracy")
        axes[r][1].set_title(f"[{mod}] Metrics per epoch")
        axes[r][1].set_xlabel("Epoch")
        axes[r][1].set_ylabel("Score")
        axes[r][1].set_ylim(0, 1)
        axes[r][1].legend()
        axes[r][1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved → {save_path}")


# ── Per-modality MLP head ─────────────────────────────────────────────────────


class ModalityMLP(nn.Module):
    """
    Simple Linear → GELU → Dropout (× len(hidden_dims)) → Linear classifier.

    Operates on a single z-scored modality embedding vector.  Kept tiny so
    that with ~1500 training samples we don't blow up the parameter budget
    per modality.

    Parameters
    ----------
    input_dropout : float
        Dropout applied to the raw input vector BEFORE the first Linear.
        Useful for high-dim modalities (e.g. Swin's 768-d pooled output)
        where many feature dimensions are noisy and overfitting kicks in
        within 2-3 epochs.  Set to 0 to disable.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int = 2,
        dropout: float = 0.3,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        if input_dropout > 0:
            layers.append(nn.Dropout(p=input_dropout))
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(p=dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ── Per-modality hyperparameter resolution ────────────────────────────────────


def _resolve_modality_cfg(CFG: dict, modality: str) -> dict:
    """
    Build the per-modality hyperparameter dict by stacking defaults:

        global fusion/training defaults
        → ``[fusion.modality_defaults]`` overrides
        → ``[fusion.modality.<mod>]`` overrides

    This lets you regularise video aggressively (dropout, weight_decay) and
    relax audio (wider hidden_dims, lower dropout) without touching the text
    head's already-tuned configuration.

    Returns a flat dict consumed by ``train_modality_head``.
    """
    fcfg = CFG["fusion"]
    tcfg = CFG["training"]

    # Stage 1: global defaults from existing config.
    resolved = {
        "hidden_dims":    fcfg.get("modality_hidden_dims", fcfg.get("hidden_dims", [128])),
        "dropout":        float(fcfg.get("modality_dropout", fcfg.get("dropout", 0.3))),
        "input_dropout":  float(fcfg.get("modality_input_dropout", 0.0)),
        "label_smoothing": float(fcfg.get("label_smoothing", 0.1)),
        "learning_rate":  float(tcfg["learning_rate"]),
        "weight_decay":   float(tcfg["weight_decay"]),
        "epochs":         int(tcfg["epochs"]),
        "patience":       int(tcfg["early_stopping_patience"]),
    }

    # Stage 2: per-modality block under ``[fusion.modality.<mod>]``.
    mod_block = fcfg.get("modality", {}).get(modality, {}) if isinstance(
        fcfg.get("modality", {}), dict
    ) else {}
    if mod_block:
        # Only the keys we explicitly support — silently ignore unknown ones
        # so the TOML can carry comments / annotations without breaking.
        for key in (
            "hidden_dims", "dropout", "input_dropout", "label_smoothing",
            "learning_rate", "weight_decay", "epochs", "patience",
        ):
            if key in mod_block:
                resolved[key] = mod_block[key]
        # Normalise numeric types so TOML "1e-2" strings don't sneak through.
        for key in ("dropout", "input_dropout", "label_smoothing",
                    "learning_rate", "weight_decay"):
            resolved[key] = float(resolved[key])
        for key in ("epochs", "patience"):
            resolved[key] = int(resolved[key])

    return resolved


# ── Per-modality dataset wrapper ──────────────────────────────────────────────


class _SingleModalityViewDataset(Dataset):
    """
    Wraps a single modality slice of a ``MultimodalCachedFusionDataset``.

    Stores the pre-scaled tensor on CPU as a numpy array and randomly samples
    a view at every ``__getitem__`` call when ``random_view=True``;
    deterministic view 0 otherwise.
    """

    def __init__(
        self,
        emb_tensor: torch.Tensor,         # (N, K, dim)
        labels: List[int],
        scaler: StandardScaler,
        random_view: bool,
    ):
        super().__init__()
        t = emb_tensor.numpy().astype(np.float32)
        flat = t.reshape(-1, t.shape[-1])
        flat = scaler.transform(flat)
        self.scaled = flat.reshape(t.shape)   # (N, K, dim)
        self.labels = list(labels)
        self.K = self.scaled.shape[1]
        self.random_view = bool(random_view)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.random_view and self.K > 1:
            k = int(np.random.randint(self.K))
        else:
            k = 0
        return (
            torch.from_numpy(self.scaled[idx, k]),
            int(self.labels[idx]),
        )


# ── Per-modality training loop ────────────────────────────────────────────────


def train_modality_head(
    modality: str,
    train_ds,
    val_ds,
    test_ds,
    CFG: dict,
    weights_dir: Path,
    device: torch.device,
) -> Tuple[nn.Module, StandardScaler, dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a StandardScaler on the train cache for ``modality``, train an MLP
    head with early stopping on val macro-F1, and finally return per-sample
    softmax probabilities for train/val/test (always using view 0).

    The probabilities are what the late-fusion combiner consumes.

    Returns
    -------
    best_model : nn.Module        — checkpoint restored to best val F1
    scaler     : StandardScaler   — fitted on all train (N*K) views
    history    : dict             — per-epoch curves for plotting
    p_train    : (N_train, 2)     — softmax probs on train view 0
    p_val      : (N_val, 2)
    p_test     : (N_test, 2)
    """
    tcfg = CFG["training"]
    BATCH_SIZE = int(tcfg["batch_size"])
    NUM_WORKERS = int(tcfg["num_workers"])
    GRAD_CLIP = float(tcfg["grad_clip"])

    # Per-modality config (globals + ``[fusion.modality.<mod>]`` overrides).
    mcfg = _resolve_modality_cfg(CFG, modality)
    HIDDEN = mcfg["hidden_dims"]
    DROPOUT = float(mcfg["dropout"])
    INPUT_DROPOUT = float(mcfg["input_dropout"])
    LABEL_SMOOTH = float(mcfg["label_smoothing"])
    LR = float(mcfg["learning_rate"])
    WD = float(mcfg["weight_decay"])
    EPOCHS = int(mcfg["epochs"])
    PATIENCE = int(mcfg["patience"])

    emb_train = getattr(train_ds, modality)            # (N, K, dim)
    emb_val = getattr(val_ds, modality)
    emb_test = getattr(test_ds, modality)
    input_dim = emb_train.shape[-1]

    # Fit StandardScaler on ALL N*K train views (matches the early-fusion baseline).
    scaler = StandardScaler()
    scaler.fit(emb_train.reshape(-1, input_dim).numpy().astype(np.float32))

    train_td = _SingleModalityViewDataset(
        emb_train, train_ds.labels, scaler, random_view=True
    )
    val_td = _SingleModalityViewDataset(
        emb_val, val_ds.labels, scaler, random_view=False
    )
    test_td = _SingleModalityViewDataset(
        emb_test, test_ds.labels, scaler, random_view=False
    )

    # Class-balanced sampler so the head sees minority A-H samples often enough.
    class_counts = np.bincount(train_ds.labels, minlength=2)
    class_weights = 1.0 / class_counts.astype(np.float64)
    sample_weights = torch.tensor(
        [class_weights[l] for l in train_ds.labels], dtype=torch.float64
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    drop_last = (len(train_td) % BATCH_SIZE) == 1
    train_loader = DataLoader(
        train_td, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, drop_last=drop_last,
    )
    val_loader = DataLoader(val_td, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_td, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = ModalityMLP(
        input_dim=input_dim, hidden_dims=HIDDEN,
        num_classes=2, dropout=DROPOUT, input_dropout=INPUT_DROPOUT,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\n[{modality}] input_dim={input_dim}  hidden_dims={HIDDEN}  "
        f"dropout={DROPOUT}  input_dropout={INPUT_DROPOUT}  "
        f"lr={LR:.2e}  weight_decay={WD}  epochs={EPOCHS}  patience={PATIENCE}"
    )
    print(f"[{modality}] Architecture:\n{model}")
    print(f"[{modality}] Trainable params: {trainable:,}")
    print(f"[{modality}] Class counts — No A-H: {class_counts[0]:,}  With A-H: {class_counts[1]:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    criterion_eval = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 1e-2
    )

    best_f1 = -1.0
    prev_f1 = -1.0
    no_improve = 0
    history = {"tr_loss": [], "va_loss": [], "tr_f1": [], "va_f1": [], "va_acc": []}
    ckpt_path = weights_dir / f"late_fusion_{modality}_best.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_state: dict = {}

    print(f"\n[{modality}] Training MLP head ...")
    train_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_preds, tr_labels_ep = 0.0, [], []
        for x, y in tqdm(
            train_loader, desc=f"[{modality}] Ep {epoch}/{EPOCHS} train", leave=False
        ):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            tr_loss += loss.item() * len(y)
            tr_preds.extend(logits.argmax(1).detach().cpu().tolist())
            tr_labels_ep.extend(y.cpu().tolist())

        model.eval()
        va_loss, va_preds, va_labels_ep = 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                va_loss += criterion_eval(logits, y).item() * len(y)
                va_preds.extend(logits.argmax(1).cpu().tolist())
                va_labels_ep.extend(y.cpu().tolist())
        scheduler.step()

        tr_f1 = f1_score(tr_labels_ep, tr_preds, average="macro", zero_division=0)
        va_f1 = f1_score(va_labels_ep, va_preds, average="macro", zero_division=0)
        va_acc = accuracy_score(va_labels_ep, va_preds)
        history["tr_loss"].append(tr_loss / max(len(tr_labels_ep), 1))
        history["va_loss"].append(va_loss / max(len(va_labels_ep), 1))
        history["tr_f1"].append(tr_f1)
        history["va_f1"].append(va_f1)
        history["va_acc"].append(va_acc)

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{modality}] Epoch {epoch:>3}/{EPOCHS}  "
            f"tr_loss={history['tr_loss'][-1]:.4f}  tr_f1={tr_f1:.4f}  "
            f"va_loss={history['va_loss'][-1]:.4f}  va_f1={va_f1:.4f}  "
            f"va_acc={va_acc:.4f}  lr={cur_lr:.2e}"
        )

        # All-time-best tracking — this is what gets restored at the end,
        # independent of the early-stopping counter below.
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": best_state,
                    "input_dim": input_dim,
                    "hidden_dims": HIDDEN,
                    "dropout": DROPOUT,
                    "input_dropout": INPUT_DROPOUT,
                    "modality": modality,
                },
                ckpt_path,
            )
            import joblib

            joblib.dump(
                scaler,
                ckpt_path.with_suffix(".scaler.joblib"),
                compress=3,
            )
            print(f"  ✓ New best val F1={best_f1:.4f} — saved {ckpt_path}")

        # Early-stopping patience is measured against the PREVIOUS epoch's
        # val F1 (local improvement), NOT the all-time best.  This lets the
        # run keep going through oscillations/plateaus as long as it's still
        # climbing epoch-to-epoch, while the all-time best above guarantees we
        # always restore the single best checkpoint at the end.
        if va_f1 > prev_f1:
            no_improve = 0
        else:
            no_improve += 1
            print(f"  ✗ No improvement vs prev epoch ({no_improve}/{PATIENCE})")

        prev_f1 = va_f1

        if PATIENCE > 0 and no_improve >= PATIENCE:
            print(f"  ⛔ Early stopping at epoch {epoch} (best val F1={best_f1:.4f})")
            break

    total_time = time.time() - train_start
    print(f"[{modality}] Trained in {total_time / 60:.1f} min — best val F1={best_f1:.4f}")

    # Restore best weights.
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Score train/val/test on deterministic view 0 → softmax probs for fusion.
    def _probs(loader: DataLoader) -> np.ndarray:
        out_chunks: List[np.ndarray] = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = model(x)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                out_chunks.append(probs)
        return np.concatenate(out_chunks, axis=0)

    # Loaders that DON'T shuffle and DON'T sample views randomly — give us
    # deterministic, label-aligned probability matrices for the combiner.
    eval_train_td = _SingleModalityViewDataset(
        emb_train, train_ds.labels, scaler, random_view=False
    )
    eval_train_loader = DataLoader(eval_train_td, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    p_train = _probs(eval_train_loader)
    p_val = _probs(val_loader)
    p_test = _probs(test_loader)
    return model, scaler, history, p_train, p_val, p_test


# ── Temperature scaling & logit-space utilities ───────────────────────────────


_LOG_EPS = 1e-12


def _safe_log(p: np.ndarray) -> np.ndarray:
    """log(p) with epsilon clipping so we never feed -inf into the combiner."""
    return np.log(np.clip(p, _LOG_EPS, 1.0))


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(
    probs_val: np.ndarray,
    y_val: np.ndarray,
    *,
    grid: Optional[np.ndarray] = None,
) -> float:
    """
    Fit a single scalar temperature T on val by minimising NLL of
    ``softmax(log(probs_val) / T)`` against ``y_val``.

    We grid-search on a log-spaced range — cheap, robust, no LBFGS required
    for 2-class problems with ≤200 val samples.  Returns T ≥ 0.05.
    """
    if grid is None:
        grid = np.concatenate(
            [np.linspace(0.25, 1.0, 16), np.linspace(1.05, 5.0, 40)]
        )
    z = _safe_log(probs_val)  # (N, 2), acts as logits up to per-sample constant
    best_T = 1.0
    best_nll = float("inf")
    N = z.shape[0]
    for T in grid:
        p = _softmax_np(z / float(T))
        nll = -float(np.log(np.clip(p[np.arange(N), y_val], _LOG_EPS, 1.0)).mean())
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)
    return best_T


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Return temperature-scaled probabilities."""
    return _softmax_np(_safe_log(probs) / max(T, 1e-3))


# ── Late-fusion combiners ─────────────────────────────────────────────────────


def fuse_mean(probs_per_mod: Dict[str, np.ndarray]) -> np.ndarray:
    """Unweighted average of per-modality softmax probs."""
    stack = np.stack(list(probs_per_mod.values()), axis=0)  # (M, N, 2)
    return stack.mean(axis=0)


def fuse_logit_mean(probs_per_mod: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Unweighted mean in logit (log-prob) space → softmax.
    Equivalent to the renormalised geometric mean of per-modality probs.
    More robust to over-confident heads than arithmetic prob averaging.
    """
    logp = np.stack([_safe_log(p) for p in probs_per_mod.values()], axis=0)  # (M,N,2)
    return _softmax_np(logp.mean(axis=0))


def fuse_logit_weighted(
    probs_per_mod: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """Weighted mean in logit space → softmax (weights normalised to sum=1)."""
    mods = list(probs_per_mod.keys())
    w = np.array([weights[m] for m in mods], dtype=np.float64)
    w = w / max(w.sum(), 1e-9)
    logp = np.stack([_safe_log(probs_per_mod[m]) for m in mods], axis=0)  # (M,N,2)
    return _softmax_np((w[:, None, None] * logp).sum(axis=0))


def fuse_weighted(
    probs_per_mod: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """
    Weighted convex average of per-modality probs.

    Weights are normalised to sum to 1 inside; this function is consumed by
    the val-tuned grid search below.
    """
    mods = list(probs_per_mod.keys())
    w = np.array([weights[m] for m in mods], dtype=np.float64)
    w = w / max(w.sum(), 1e-9)
    stack = np.stack([probs_per_mod[m] for m in mods], axis=0)  # (M, N, 2)
    return (w[:, None, None] * stack).sum(axis=0)


def grid_search_weights(
    p_val_per_mod: Dict[str, np.ndarray],
    y_val: np.ndarray,
    step: float = 0.05,
    *,
    space: str = "prob",
) -> Tuple[Dict[str, float], float]:
    """
    Brute-force search over the discrete simplex for the (text, audio, video)
    weights that maximise val macro-F1.  Search resolution defaults to a
    21-point grid per modality — cheap and bias-free for 3 modalities.

    Returns ``(best_weights_dict, best_val_f1)``.
    """
    mods = list(p_val_per_mod.keys())
    best = (None, -1.0)
    grid = np.arange(0.0, 1.0 + step / 2, step)
    _fuse = fuse_logit_weighted if space == "logit" else fuse_weighted

    if len(mods) == 1:
        return ({mods[0]: 1.0}, f1_score(
            y_val, p_val_per_mod[mods[0]].argmax(1), average="macro", zero_division=0
        ))

    if len(mods) == 2:
        for w0 in grid:
            w1 = 1.0 - w0
            if w1 < -1e-9:
                continue
            weights = {mods[0]: float(w0), mods[1]: float(max(w1, 0.0))}
            p = _fuse(p_val_per_mod, weights)
            f1 = f1_score(y_val, p.argmax(1), average="macro", zero_division=0)
            if f1 > best[1]:
                best = (weights, float(f1))
        return best  # type: ignore[return-value]

    # 3-modality case (typical here): enumerate the discrete 2-simplex.
    for w0 in grid:
        for w1 in grid:
            w2 = 1.0 - w0 - w1
            if w2 < -1e-9:
                continue
            weights = {mods[0]: float(w0), mods[1]: float(w1), mods[2]: float(max(w2, 0.0))}
            p = _fuse(p_val_per_mod, weights)
            f1 = f1_score(y_val, p.argmax(1), average="macro", zero_division=0)
            if f1 > best[1]:
                best = (weights, float(f1))
    return best  # type: ignore[return-value]


def pso_search_weights(
    p_val_per_mod: Dict[str, np.ndarray],
    y_val: np.ndarray,
    *,
    space: str = "prob",
    n_particles: int = 40,
    n_iter: int = 200,
    seed: int = 42,
    w_inertia: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    T_min: float = 0.25,
    T_max: float = 5.0,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Particle Swarm Optimisation over per-modality fusion weights **and**
    per-modality temperatures, jointly maximising val macro-F1.

    Each particle encodes a 2M-dimensional vector:
        [w_0, …, w_{M-1},  T_0, …, T_{M-1}]

    Weights are projected onto the probability simplex (softmax) before
    evaluating the objective so the search space is unconstrained.
    Temperatures are clipped to [T_min, T_max].

    Parameters
    ----------
    p_val_per_mod : dict  {modality → (N, 2) raw probs}
    y_val         : (N,) int labels
    space         : "prob" | "logit"  — fusion space (same as grid search)
    n_particles   : swarm size
    n_iter        : number of PSO iterations
    seed          : RNG seed for reproducibility
    w_inertia     : inertia weight (momentum)
    c1, c2        : cognitive / social acceleration coefficients
    T_min, T_max  : temperature search bounds

    Returns
    -------
    best_weights  : dict {modality → weight}  (sum to 1)
    best_temps    : dict {modality → temperature}
    best_val_f1   : float
    """
    rng = np.random.default_rng(seed)
    mods = list(p_val_per_mod.keys())
    M = len(mods)
    _fuse = fuse_logit_weighted if space == "logit" else fuse_weighted

    # Dimension layout: [w_0..w_{M-1}, T_0..T_{M-1}]
    D = 2 * M

    def _decode(x: np.ndarray):
        """Return (weights_dict, temps_dict) from a raw particle position."""
        raw_w = x[:M]
        # Softmax projection → simplex (always sums to 1, all positive)
        raw_w = raw_w - raw_w.max()
        exp_w = np.exp(raw_w)
        w = exp_w / exp_w.sum()
        T = np.clip(x[M:], T_min, T_max)
        weights = {mods[i]: float(w[i]) for i in range(M)}
        temps   = {mods[i]: float(T[i]) for i in range(M)}
        return weights, temps

    def _objective(x: np.ndarray) -> float:
        """Return val macro-F1 (higher = better)."""
        weights, temps = _decode(x)
        p_cal = {m: apply_temperature(p_val_per_mod[m], temps[m]) for m in mods}
        p_fused = _fuse(p_cal, weights)
        return float(f1_score(y_val, p_fused.argmax(1), average="macro", zero_division=0))

    # ── Initialise swarm ──────────────────────────────────────────────────────
    # Weight dimensions: uniform in [-2, 2] (softmax maps this to a broad simplex)
    # Temperature dimensions: uniform in [T_min, T_max]
    pos = np.empty((n_particles, D))
    pos[:, :M] = rng.uniform(-2.0, 2.0, size=(n_particles, M))
    pos[:, M:] = rng.uniform(T_min, T_max, size=(n_particles, M))

    vel = np.zeros_like(pos)
    vel[:, :M] = rng.uniform(-0.5, 0.5, size=(n_particles, M))
    vel[:, M:] = rng.uniform(-0.5, 0.5, size=(n_particles, M))

    pbest_pos = pos.copy()
    pbest_val = np.array([_objective(pos[i]) for i in range(n_particles)])

    gbest_idx = int(pbest_val.argmax())
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = float(pbest_val[gbest_idx])

    # ── PSO main loop ─────────────────────────────────────────────────────────
    for _ in range(n_iter):
        r1 = rng.uniform(0.0, 1.0, size=(n_particles, D))
        r2 = rng.uniform(0.0, 1.0, size=(n_particles, D))

        vel = (
            w_inertia * vel
            + c1 * r1 * (pbest_pos - pos)
            + c2 * r2 * (gbest_pos[None, :] - pos)
        )
        pos = pos + vel

        # Evaluate new positions
        for i in range(n_particles):
            val = _objective(pos[i])
            if val > pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = pos[i].copy()
                if val > gbest_val:
                    gbest_val = val
                    gbest_pos = pos[i].copy()

    best_weights, best_temps = _decode(gbest_pos)
    return best_weights, best_temps, gbest_val


def fit_meta_stacker(
    p_train_per_mod: Dict[str, np.ndarray],
    y_train: np.ndarray,
    seed: int,
):
    """
    Fit a small meta-classifier on per-modality probabilities (stacking).

    Logistic regression on the per-modality positive-class probabilities is
    almost always sufficient with only 3 modalities and limited val data —
    fancier stackers overfit the meta level.
    """
    from sklearn.linear_model import LogisticRegression

    mods = list(p_train_per_mod.keys())
    # Use BOTH classes' probs to give the meta learner full per-modality posterior.
    # Shape: (N, 2 * M).
    X = np.concatenate([p_train_per_mod[m] for m in mods], axis=1)
    meta = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=seed
    )
    meta.fit(X, y_train)
    return meta, mods


def meta_predict(meta, mods: List[str], p_per_mod: Dict[str, np.ndarray]) -> np.ndarray:
    """Run a fitted meta-stacker on aligned per-modality probability dicts."""
    X = np.concatenate([p_per_mod[m] for m in mods], axis=1)
    return meta.predict_proba(X)


# ── Reporting helper ──────────────────────────────────────────────────────────


def report_block(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[{label}] Macro-F1: {f1:.4f}  |  Accuracy: {acc:.4f}")
    print(f"[{label}] Label dist : {dict(Counter(y_true.tolist()))}")
    print(f"[{label}] Pred  dist : {dict(Counter(y_pred.tolist()))}")
    print(
        classification_report(
            y_true, y_pred, target_names=["No A-H", "With A-H"], digits=4, zero_division=0
        )
    )
    return f1, acc


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    from load_dataset import get_cached_multimodal_splits

    CFG = toml.load("config_late_fusion.toml")

    ACTIVE_MODS = CFG["fusion"].get(
        "active_modalities", ["text", "audio", "video"]
    )
    _cache_dir_raw = CFG["fusion"].get("cache_dir", "multimodal/cache")
    CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(REPO_ROOT / _cache_dir_raw)))
    VIDEO_BACKBONE = CFG["fusion"].get("video_backbone", "swin")
    AUDIO_BACKBONE = CFG["fusion"].get("audio_backbone", "wav2vec2emotional")

    SEED = int(CFG["training"]["seed"])
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    WEIGHTS_DIR = REPO_ROOT / "multimodal" / CFG["output"]["weights_dir"]
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = WEIGHTS_DIR / "late_fusion_training_curves.png"

    DEVICE = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print("=" * 70)
    print("Multimodal Late-Fusion Training — cache-driven")
    print("=" * 70)
    print(f"Device              : {DEVICE}")
    print(f"Cache dir           : {CACHE_DIR}")
    print(f"Video backbone      : {VIDEO_BACKBONE}")
    print(f"Audio backbone      : {AUDIO_BACKBONE}")
    print(f"Active modalities   : {ACTIVE_MODS}")
    print(f"Seed                : {SEED}")
    print(f"Weights dir         : {WEIGHTS_DIR}")
    print("=" * 70)

    # Load the three per-modality caches once and reuse for every head.
    # ``train_random_view=True`` lets each modality head do its own per-step
    # view-augmentation; val/test always use view 0.
    print("\n[Cache] Loading multi-view embedding cache ...")
    splits = get_cached_multimodal_splits(
        cache_dir=CACHE_DIR,
        train_random_view=True,
        expected_fingerprint=None,
        video_backbone=VIDEO_BACKBONE,
        audio_backbone=AUDIO_BACKBONE,
    )
    train_ds, val_ds, test_ds = splits["train"], splits["val"], splits["test"]

    for ds_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        cnt = Counter(ds.labels)
        n = len(ds.labels)
        pct_pos = cnt.get(1, 0) / n * 100 if n > 0 else 0
        print(
            f"  [{ds_name:5s}]  {n:>5,} videos  |  "
            f"No A-H: {cnt.get(0, 0):,}  With A-H: {cnt.get(1, 0):,}  "
            f"({pct_pos:.1f}% positive)  |  "
            f"K_t={ds.K_t} K_a={ds.K_a} K_v={ds.K_v}"
        )

    # Sanity: labels MUST be aligned across modalities (load_dataset already
    # warns on mismatch, but late fusion is pointless without alignment).
    y_train = np.asarray(train_ds.labels, dtype=np.int64)
    y_val = np.asarray(val_ds.labels, dtype=np.int64)
    y_test = np.asarray(test_ds.labels, dtype=np.int64)

    # ── Stage 1: train one MLP per modality ──────────────────────────────────
    histories: Dict[str, dict] = {}
    p_train: Dict[str, np.ndarray] = {}
    p_val: Dict[str, np.ndarray] = {}
    p_test: Dict[str, np.ndarray] = {}

    for mod in ACTIVE_MODS:
        print("\n" + "=" * 70)
        print(f"Training modality head: {mod}")
        print("=" * 70)
        _, _, hist, ptr, pva, pte = train_modality_head(
            modality=mod,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            CFG=CFG,
            weights_dir=WEIGHTS_DIR,
            device=DEVICE,
        )
        histories[mod] = hist
        p_train[mod] = ptr
        p_val[mod] = pva
        p_test[mod] = pte

    plot_history(histories, plot_path)

    # ── Stage 2: report per-modality scores (sanity baselines) ────────────────
    print("\n" + "=" * 70)
    print("Per-modality test scores (before fusion)")
    print("=" * 70)
    per_mod_f1: Dict[str, float] = {}
    for mod in ACTIVE_MODS:
        preds = p_test[mod].argmax(1)
        f1, _ = report_block(f"{mod} (test, view 0)", y_test, preds)
        per_mod_f1[mod] = f1

    # ── Stage 3: per-modality temperature scaling on val ──────────────────────
    # Each modality head outputs its own probability scale (text is tight,
    # video is over-confident-wrong, audio is near-flat).  Arithmetic prob
    # averaging is dominated by the most peaked head, so we first fit a
    # scalar T per modality on val NLL, then build two parallel views of the
    # per-modality probs: raw and temperature-calibrated.
    print("\n" + "=" * 70)
    print("Temperature scaling (fit on val NLL)")
    print("=" * 70)
    temps: Dict[str, float] = {}
    p_val_T: Dict[str, np.ndarray] = {}
    p_test_T: Dict[str, np.ndarray] = {}
    p_train_T: Dict[str, np.ndarray] = {}
    for mod in ACTIVE_MODS:
        T = fit_temperature(p_val[mod], y_val)
        temps[mod] = T
        p_val_T[mod] = apply_temperature(p_val[mod], T)
        p_test_T[mod] = apply_temperature(p_test[mod], T)
        p_train_T[mod] = apply_temperature(p_train[mod], T)
        print(f"  [{mod}] T*={T:.3f}  (T>1 ⇒ softening over-confident head)")

    # Persist for inference reproducibility.
    import joblib

    # ── Stage 4: late-fusion combiners over (subset × strategy) grid ──────────
    print("\n" + "=" * 70)
    print("Late-fusion combiners — full set + leave-out subsets")
    print("=" * 70)

    y_combined = np.concatenate([y_val, y_test], axis=0)

    def _row(name: str, p_v: np.ndarray, p_t: np.ndarray):
        vp = p_v.argmax(1)
        tp = p_t.argmax(1)
        cp = np.concatenate([vp, tp], axis=0)
        return (
            name,
            f1_score(y_val,      vp, average="macro", zero_division=0),
            accuracy_score(y_val, vp),
            f1_score(y_test,     tp, average="macro", zero_division=0),
            accuracy_score(y_test, tp),
            f1_score(y_combined, cp, average="macro", zero_division=0),
            accuracy_score(y_combined, cp),
        )

    summary_rows: List[Tuple[str, float, float, float, float, float, float]] = []

    # Unimodal rows (raw + calibrated, since calibration changes argmax only
    # rarely but lets us confirm the per-modality head wasn't degraded).
    for mod in ACTIVE_MODS:
        summary_rows.append(_row(f"unimodal: {mod}", p_val[mod], p_test[mod]))

    # Candidate subsets.  Always include the full active set; if both text and
    # audio are active, also report the 2-way {text, audio} fusion as an
    # explicit "video dropped" baseline.
    subsets: List[List[str]] = [list(ACTIVE_MODS)]
    if {"text", "audio"}.issubset(set(ACTIVE_MODS)) and len(ACTIVE_MODS) > 2:
        subsets.append(["text", "audio"])
    if {"text", "video"}.issubset(set(ACTIVE_MODS)) and len(ACTIVE_MODS) > 2:
        subsets.append(["text", "video"])

    persisted_artifacts: Dict[str, dict] = {}

    for subset in subsets:
        tag = "+".join(subset)
        print(f"\n── Subset: {{ {tag} }} ──")

        # Restrict per-modality dicts to this subset (preserves order).
        pV_raw = {m: p_val[m] for m in subset}
        pT_raw = {m: p_test[m] for m in subset}
        pTr_raw = {m: p_train[m] for m in subset}

        pV_cal = {m: p_val_T[m] for m in subset}
        pT_cal = {m: p_test_T[m] for m in subset}
        pTr_cal = {m: p_train_T[m] for m in subset}

        # (a) Arithmetic mean — raw + calibrated.
        summary_rows.append(_row(
            f"[{tag}] mean (raw)", fuse_mean(pV_raw), fuse_mean(pT_raw)
        ))
        summary_rows.append(_row(
            f"[{tag}] mean (T-scaled)", fuse_mean(pV_cal), fuse_mean(pT_cal)
        ))

        # (b) Logit-space mean — equivalent to renormalised geometric mean.
        summary_rows.append(_row(
            f"[{tag}] logit-mean (raw)",
            fuse_logit_mean(pV_raw), fuse_logit_mean(pT_raw),
        ))
        summary_rows.append(_row(
            f"[{tag}] logit-mean (T-scaled)",
            fuse_logit_mean(pV_cal), fuse_logit_mean(pT_cal),
        ))

        # (c) Weighted average — grid search on val, in both spaces.
        bw_p, _ = grid_search_weights(pV_cal, y_val, step=0.05, space="prob")
        bw_l, _ = grid_search_weights(pV_cal, y_val, step=0.05, space="logit")
        print(f"  weighted-prob   best val weights = {bw_p}")
        print(f"  weighted-logit  best val weights = {bw_l}")
        summary_rows.append(_row(
            f"[{tag}] weighted (T-scaled, prob)",
            fuse_weighted(pV_cal, bw_p), fuse_weighted(pT_cal, bw_p),
        ))
        summary_rows.append(_row(
            f"[{tag}] weighted (T-scaled, logit)",
            fuse_logit_weighted(pV_cal, bw_l), fuse_logit_weighted(pT_cal, bw_l),
        ))

        # (d) Meta-LR stacker on calibrated train probs.
        meta_s, meta_mods_s = fit_meta_stacker(pTr_cal, y_train, seed=SEED)
        summary_rows.append(_row(
            f"[{tag}] meta-LR (T-scaled)",
            meta_predict(meta_s, meta_mods_s, pV_cal),
            meta_predict(meta_s, meta_mods_s, pT_cal),
        ))

        # (e) PSO — jointly optimise per-modality weights + temperatures on val.
        pso_cfg = CFG.get("pso", {})
        _pso_kwargs = dict(
            n_particles=int(pso_cfg.get("n_particles", 40)),
            n_iter=int(pso_cfg.get("n_iter", 200)),
            seed=SEED,
            w_inertia=float(pso_cfg.get("w_inertia", 0.7)),
            c1=float(pso_cfg.get("c1", 1.5)),
            c2=float(pso_cfg.get("c2", 1.5)),
            T_min=float(pso_cfg.get("T_min", 0.25)),
            T_max=float(pso_cfg.get("T_max", 5.0)),
        )
        pso_w_p, pso_T_p, pso_vf1_p = pso_search_weights(
            pV_raw, y_val, space="prob", **_pso_kwargs
        )
        pso_w_l, pso_T_l, pso_vf1_l = pso_search_weights(
            pV_raw, y_val, space="logit", **_pso_kwargs
        )
        print(f"  PSO-prob   best val weights = {pso_w_p}  temps = {pso_T_p}  val_f1={pso_vf1_p:.4f}")
        print(f"  PSO-logit  best val weights = {pso_w_l}  temps = {pso_T_l}  val_f1={pso_vf1_l:.4f}")

        def _pso_fuse_test(weights, temps_d, space_s):
            p_cal_v = {m: apply_temperature(pV_raw[m], temps_d[m]) for m in subset}
            p_cal_t = {m: apply_temperature(pT_raw[m], temps_d[m]) for m in subset}
            _f = fuse_logit_weighted if space_s == "logit" else fuse_weighted
            return _f(p_cal_v, weights), _f(p_cal_t, weights)

        pso_pv_p, pso_pt_p = _pso_fuse_test(pso_w_p, pso_T_p, "prob")
        pso_pv_l, pso_pt_l = _pso_fuse_test(pso_w_l, pso_T_l, "logit")
        summary_rows.append(_row(f"[{tag}] PSO (prob)",  pso_pv_p, pso_pt_p))
        summary_rows.append(_row(f"[{tag}] PSO (logit)", pso_pv_l, pso_pt_l))

        persisted_artifacts[tag] = {
            "subset": subset,
            "temperatures": {m: temps[m] for m in subset},
            "best_weights_prob_space": bw_p,
            "best_weights_logit_space": bw_l,
            "meta_learner": meta_s,
            "meta_input_modalities_order": meta_mods_s,
            "pso_weights_prob": pso_w_p,
            "pso_temps_prob": pso_T_p,
            "pso_weights_logit": pso_w_l,
            "pso_temps_logit": pso_T_l,
        }

    joblib.dump(
        {
            "active_modalities": ACTIVE_MODS,
            "temperatures": temps,
            "subsets": persisted_artifacts,
        },
        WEIGHTS_DIR / "late_fusion_meta.joblib",
        compress=3,
    )

    # ── Final summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 104)
    print("Late-Fusion Summary  (macro-F1 / accuracy; V+T = val + test pooled)")
    print("=" * 104)
    header = (
        f"{'Strategy':<42}  {'Val F1':>8}  {'Val acc':>8}  "
        f"{'Test F1':>8}  {'Test acc':>8}  {'V+T F1':>8}  {'V+T acc':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, vf1, vacc, tf1, tacc, cf1, cacc in summary_rows:
        print(
            f"{name:<42}  {vf1:>8.4f}  {vacc:>8.4f}  "
            f"{tf1:>8.4f}  {tacc:>8.4f}  {cf1:>8.4f}  {cacc:>8.4f}"
        )
    print("=" * 104)
    print(f"Temperatures         : {temps}")
    print(f"Per-modality weights : {WEIGHTS_DIR}/late_fusion_<mod>_best.pth")
    print(f"Per-modality scalers : {WEIGHTS_DIR}/late_fusion_<mod>_best.scaler.joblib")
    print(f"Meta + temps + ws    : {WEIGHTS_DIR}/late_fusion_meta.joblib")
    print(f"Training curves      : {plot_path}")


if __name__ == "__main__":
    main()
