"""
Multimodal Fusion-Head Training Pipeline (cache-driven)
=========================================================

Trains a fusion classifier on top of pre-computed, multi-view per-modality
embeddings produced by the ``extract_{text,audio,video}.py`` scripts.

The classifier family is selected via ``[fusion] classifier`` in
``multimodal/config.toml``.  Supported values:

  * ``"mlp"``             — neural MLP head (default, with Mixup + dropout)
  * ``"svm"``             — sklearn ``SVC`` with RBF kernel
  * ``"random_forest"``   — sklearn ``RandomForestClassifier``
  * ``"gradient_boosting"`` — XGBoost or CatBoost (selectable per sub-section)

All four classifiers share the SAME feature-construction pipeline:

  1.  Read multi-view caches (text / audio / video).
  2.  Per-modality z-score standardisation using TRAIN statistics
      (substitutes for LayerNorm in the neural head; necessary because the
      three modalities live at very different scales).
  3.  Concatenate the per-modality vectors → (N, sum(dims)).

For MLP training, multi-view sampling happens batch-by-batch through the
``MultimodalCachedFusionDataset``.  For classical classifiers, all train
views are stacked into one design matrix (so each training sample becomes
``K_t * K_a * K_v`` rows — bounded above by ``max_train_rows`` in config to
keep RF/SVM tractable).  Val/test always use view 0.

Pipeline split
--------------
1. ``multimodal/src/extract_{text,audio,video}.py``   ← per-modality K-view
   extraction (one cache per modality per split).
2. ``multimodal/src/fusion_training.py``               ← this file.

Usage
-----
    python fusion_training.py

    CACHE_DIR=/path/to/cache python fusion_training.py
"""

import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import toml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm.auto import tqdm


# ── Repo root ────────────────────────────────────────────────────────────────

REPO_ROOT = Path('/home3/iasarantsev').resolve()
sys.path.insert(0, str(REPO_ROOT / "utils"))


# ── Plot helper ───────────────────────────────────────────────────────────────


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


# ── MLP fusion head (operates on pre-fused, pre-scaled, optionally PCA'd input)


class SimpleMLPFusionHead(nn.Module):
    """
    Linear → GELU → Dropout (× len(hidden_dims)) → Linear → logits.

    Input is the EARLY-FUSED vector produced by ``EarlyFusionPreprocessor``:
    per-modality ``StandardScaler`` outputs are concatenated and optionally
    projected through PCA.  Per-feature standardisation has therefore
    already been done outside the model, so there is no LayerNorm here.

    With ``hidden_dims == []`` this degenerates to plain logistic
    regression on the fused input.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ── Mixup ─────────────────────────────────────────────────────────────────────


def mixup_batch(
    emb_list: list,
    labels: torch.Tensor,
    alpha: float,
    num_classes: int = 2,
):
    """Apply embedding-space Mixup to a batch."""
    if alpha <= 0.0:
        soft = torch.zeros(labels.shape[0], num_classes, device=labels.device)
        soft.scatter_(1, labels.view(-1, 1), 1.0)
        return emb_list, soft, 1.0

    B = labels.shape[0]
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(B, device=labels.device)
    mixed = [lam * e + (1.0 - lam) * e[perm] for e in emb_list]

    onehot = torch.zeros(B, num_classes, device=labels.device)
    onehot.scatter_(1, labels.view(-1, 1), 1.0)
    soft = lam * onehot + (1.0 - lam) * onehot[perm]
    return mixed, soft, lam


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against soft targets (for Mixup)."""
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


# ── Early-fusion preprocessor (shared by ALL classifiers, incl. MLP) ──────────


class EarlyFusionPreprocessor:
    """
    Fit-on-TRAIN preprocessing pipeline shared by every classifier path:

        per-modality StandardScaler  →  concat  →  PCA (optional)

    Both stages are fitted on training data only, then applied identically
    to train / val / test / future inference samples.

    Parameters
    ----------
    active_mods : list[str]
        Subset of ["text", "audio", "video"] in the order their dims are
        concatenated.
    pca_components : int or float, optional
        Forwarded to ``sklearn.decomposition.PCA``:
          * ``0`` or ``None`` → skip PCA (default).
          * ``int >= 1``      → keep that many components.
          * ``0 < float < 1`` → keep enough components to retain that
            fraction of explained variance.
    pca_whiten : bool
        If True, components are scaled to unit variance after projection
        (useful for distance-based classifiers like SVC with RBF).
    seed : int
        Random state for ``PCA(svd_solver="randomized")``.
    """

    def __init__(
        self,
        active_mods: list,
        pca_components=None,
        pca_whiten: bool = False,
        seed: int = 42,
    ):
        from sklearn.preprocessing import StandardScaler

        self.active_mods = list(active_mods)
        self.scalers = {m: StandardScaler() for m in self.active_mods}
        self.pca = None
        self.pca_components = pca_components
        self.pca_whiten = bool(pca_whiten)
        self.seed = int(seed)
        self.out_dim: int = -1
        # Per-modality dimensionality, set at fit time (used to split a
        # concatenated vector back into modality blocks during transform).
        self.dims: List[int] = []

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _flatten_views(self, tensor: torch.Tensor) -> np.ndarray:
        """(N, K, dim) → (N*K, dim) numpy float32."""
        return tensor.reshape(-1, tensor.shape[-1]).numpy().astype(np.float32)

    def _select_view0(self, tensor: torch.Tensor) -> np.ndarray:
        """(N, K, dim) → (N, dim) numpy float32 using deterministic view 0."""
        return tensor[:, 0, :].numpy().astype(np.float32)

    # ── Fit / transform API ──────────────────────────────────────────────────

    def fit(self, train_ds) -> "EarlyFusionPreprocessor":
        """Fit per-modality scalers and (optional) PCA on the TRAIN cache."""
        # Stage 1: fit StandardScaler per modality over ALL N*K train views.
        scaled_blocks_for_pca = []
        self.dims = []
        for mod in self.active_mods:
            arr = self._flatten_views(getattr(train_ds, mod))   # (N*K, dim_mod)
            self.scalers[mod].fit(arr)
            self.dims.append(arr.shape[1])
            # For PCA fitting we only use view-0 to keep the design matrix
            # one-row-per-train-video (PCA isn't view-augmentation-aware).
            v0 = self._select_view0(getattr(train_ds, mod))
            scaled_blocks_for_pca.append(self.scalers[mod].transform(v0))

        # Concatenated, train-scaled, view-0 matrix used only for PCA fit.
        X_concat_train = np.concatenate(scaled_blocks_for_pca, axis=1).astype(np.float32)

        # Stage 2: optional PCA.
        pc = self.pca_components
        if pc in (None, 0, False):
            self.pca = None
            self.out_dim = X_concat_train.shape[1]
        else:
            from sklearn.decomposition import PCA

            n_samples, n_features = X_concat_train.shape
            if isinstance(pc, float) and 0.0 < pc < 1.0:
                resolved = pc
            else:
                pc_int = int(pc)
                cap = min(n_samples, n_features)
                if pc_int > cap:
                    warnings.warn(
                        f"[EarlyFusionPreprocessor] PCA components={pc_int} > "
                        f"min(n_samples={n_samples}, n_features={n_features}); "
                        f"clamping to {cap}.",
                        UserWarning,
                        stacklevel=2,
                    )
                    pc_int = cap
                resolved = pc_int
            self.pca = PCA(
                n_components=resolved,
                whiten=self.pca_whiten,
                svd_solver="randomized",
                random_state=self.seed,
            )
            self.pca.fit(X_concat_train)
            self.out_dim = int(self.pca.n_components_)
            print(
                f"[EarlyFusionPreprocessor] PCA fitted: "
                f"{X_concat_train.shape[1]} → {self.out_dim} dims  "
                f"(retained var = {float(self.pca.explained_variance_ratio_.sum()):.4f})"
            )
        return self

    def transform_view0(self, ds) -> np.ndarray:
        """Use view 0 per modality (deterministic) → preprocessed (N, out_dim)."""
        blocks = []
        for mod in self.active_mods:
            arr = self._select_view0(getattr(ds, mod))         # (N, dim_mod)
            blocks.append(self.scalers[mod].transform(arr))
        X = np.concatenate(blocks, axis=1).astype(np.float32)
        if self.pca is not None:
            X = self.pca.transform(X).astype(np.float32)
        return X

    def transform_all_views(
        self, ds, max_rows: int = -1, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Expand a dataset's full view grid into a single design matrix.

        Returns ``(X, y)`` where ``X`` has shape ``(R, out_dim)`` with
        ``R = N * K_t * K_a * K_v`` (capped by ``max_rows`` when positive).
        Used for fitting classical classifiers on view-augmented data.
        """
        per_mod_scaled = {}
        for mod in self.active_mods:
            t = getattr(ds, mod).numpy().astype(np.float32)    # (N, K_mod, dim)
            flat = t.reshape(-1, t.shape[-1])
            flat = self.scalers[mod].transform(flat)
            per_mod_scaled[mod] = flat.reshape(t.shape)        # (N, K_mod, dim)

        n = per_mod_scaled[self.active_mods[0]].shape[0]
        ks = [per_mod_scaled[m].shape[1] for m in self.active_mods]
        total = n * int(np.prod(ks))

        rng = np.random.default_rng(seed)
        if max_rows > 0 and total > max_rows:
            sample_idx = rng.integers(0, n, size=max_rows)
            view_idx = [rng.integers(0, k, size=max_rows) for k in ks]
        else:
            grids = np.meshgrid(
                np.arange(n), *[np.arange(k) for k in ks], indexing="ij"
            )
            sample_idx = grids[0].reshape(-1)
            view_idx = [g.reshape(-1) for g in grids[1:]]

        blocks = [
            per_mod_scaled[mod][sample_idx, vi]
            for mod, vi in zip(self.active_mods, view_idx)
        ]
        X = np.concatenate(blocks, axis=1).astype(np.float32)
        if self.pca is not None:
            X = self.pca.transform(X).astype(np.float32)
        y = np.asarray(ds.labels, dtype=np.int64)[sample_idx]
        return X, y


class _PreprocessedViewDataset(torch.utils.data.Dataset):
    """
    Torch ``Dataset`` that draws a random view per modality per
    ``__getitem__`` call and returns the EarlyFusion-preprocessed vector
    (``StandardScaler`` per modality → concat → optional PCA).  Used for
    MLP training only — for classical training the full flat matrix is
    built up-front via ``EarlyFusionPreprocessor.transform_all_views``.

    Stores per-modality scaled tensors on CPU as numpy arrays for fast
    indexing.  PCA (if present) is applied on-the-fly per batch row in
    ``__getitem__`` because applying it ahead of time would defeat the
    K-view randomisation.

    Parameters
    ----------
    base_ds : MultimodalCachedFusionDataset
    pre     : fitted EarlyFusionPreprocessor
    random_view : bool
        True for train (random per-modality view), False for val/test
        (always view 0).
    """

    def __init__(self, base_ds, pre: EarlyFusionPreprocessor, random_view: bool):
        super().__init__()
        self.pre = pre
        self.random_view = bool(random_view)
        self.labels = base_ds.labels

        # Pre-scale every (modality, sample, view) once — cheap, avoids
        # repeated StandardScaler calls per __getitem__.
        self._scaled = {}
        for mod in pre.active_mods:
            t = getattr(base_ds, mod).numpy().astype(np.float32)  # (N, K_mod, dim)
            flat = t.reshape(-1, t.shape[-1])
            flat = pre.scalers[mod].transform(flat)
            self._scaled[mod] = flat.reshape(t.shape)             # (N, K_mod, dim)

        self.ks = {mod: self._scaled[mod].shape[1] for mod in pre.active_mods}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        blocks = []
        for mod in self.pre.active_mods:
            if self.random_view and self.ks[mod] > 1:
                k = int(np.random.randint(self.ks[mod]))
            else:
                k = 0
            blocks.append(self._scaled[mod][idx, k])
        x = np.concatenate(blocks, axis=0).astype(np.float32)
        if self.pre.pca is not None:
            x = self.pre.pca.transform(x.reshape(1, -1))[0].astype(np.float32)
        return torch.from_numpy(x), int(self.labels[idx])


# ── Classical classifier factory ──────────────────────────────────────────────


def build_classical_classifier(name: str, cfg: dict, seed: int):
    """
    Instantiate a sklearn-compatible classifier from config.

    Each branch supplies sensible small-data defaults; every hyperparameter
    is overridable from ``[fusion.<name>]`` in ``config.toml``.
    """
    name = name.lower()

    if name == "svm":
        from sklearn.svm import SVC

        sub = cfg.get("svm", {})
        return SVC(
            C=float(sub.get("C", 1.0)),
            kernel=str(sub.get("kernel", "rbf")),
            gamma=sub.get("gamma", "scale"),
            class_weight="balanced",
            probability=False,
            random_state=seed,
        )

    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        sub = cfg.get("random_forest", {})
        # max_depth=0 in TOML means "no limit" → sklearn expects None.
        _max_depth = sub.get("max_depth", 0)
        _max_depth = None if (_max_depth in (None, 0)) else int(_max_depth)
        return RandomForestClassifier(
            n_estimators=int(sub.get("n_estimators", 500)),
            max_depth=_max_depth,
            min_samples_leaf=int(sub.get("min_samples_leaf", 2)),
            max_features=sub.get("max_features", "sqrt"),
            class_weight="balanced",
            n_jobs=int(sub.get("n_jobs", -1)),
            random_state=seed,
        )

    if name == "gradient_boosting":
        sub = cfg.get("gradient_boosting", {})
        lib = str(sub.get("library", "xgboost")).lower()

        if lib == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as e:
                raise ImportError(
                    "classifier='gradient_boosting' with library='xgboost' "
                    "requires the `xgboost` package.  Install via `pip install xgboost`."
                ) from e
            return XGBClassifier(
                n_estimators=int(sub.get("n_estimators", 400)),
                max_depth=int(sub.get("max_depth", 6)),
                learning_rate=float(sub.get("learning_rate", 0.05)),
                subsample=float(sub.get("subsample", 0.8)),
                colsample_bytree=float(sub.get("colsample_bytree", 0.8)),
                reg_lambda=float(sub.get("reg_lambda", 1.0)),
                # scale_pos_weight is applied externally based on train labels
                # so it adapts to the actual flattened-view training set.
                eval_metric="logloss",
                tree_method="hist",
                random_state=seed,
                n_jobs=int(sub.get("n_jobs", -1)),
            )

        if lib == "catboost":
            try:
                from catboost import CatBoostClassifier
            except ImportError as e:
                raise ImportError(
                    "classifier='gradient_boosting' with library='catboost' "
                    "requires the `catboost` package.  Install via `pip install catboost`."
                ) from e
            return CatBoostClassifier(
                iterations=int(sub.get("n_estimators", 400)),
                depth=int(sub.get("max_depth", 6)),
                learning_rate=float(sub.get("learning_rate", 0.05)),
                l2_leaf_reg=float(sub.get("reg_lambda", 3.0)),
                auto_class_weights="Balanced",
                loss_function="Logloss",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            )

        raise ValueError(
            f"Unknown gradient_boosting library: {lib!r}.  Use 'xgboost' or 'catboost'."
        )

    raise ValueError(
        f"Unknown classifier: {name!r}.  "
        "Choose from: mlp, svm, random_forest, gradient_boosting."
    )


# ── Reporting helper (shared) ─────────────────────────────────────────────────


def report_test(
    classifier_label: str,
    active_mods: list,
    preds: list,
    labels: list,
    extra_lines: list = None,
):
    test_pred_cnt = Counter(preds)
    test_label_cnt = Counter(labels)
    test_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    test_acc = accuracy_score(labels, preds)

    print(f"\n{'=' * 70}")
    print(f"Test Results — Multimodal Fusion ({' + '.join(active_mods)}) — {classifier_label}")
    print("=" * 70)
    print(f"[Test] Label distribution : {dict(test_label_cnt)}")
    print(f"[Test] Prediction dist.   : {dict(test_pred_cnt)}")
    print(f"[Test] Macro-F1  : {test_f1:.4f}")
    print(f"[Test] Accuracy  : {test_acc:.4f}")
    print()
    print(
        classification_report(
            labels, preds, target_names=["No A-H", "With A-H"], digits=4
        )
    )
    if extra_lines:
        for line in extra_lines:
            print(line)
    return test_f1, test_acc


# ── MLP training branch (consumes EarlyFusionPreprocessor output) ─────────────


def run_mlp(
    CFG: dict,
    pre: EarlyFusionPreprocessor,
    train_ds,
    val_ds,
    test_ds,
    active_mods: list,
    weights_path: Path,
):
    """MLP trained on the early-fused vector: StandardScaler → concat → PCA."""
    HIDDEN_DIMS = CFG["fusion"]["hidden_dims"]
    DROPOUT = CFG["fusion"]["dropout"]
    LABEL_SMOOTH = float(CFG["fusion"].get("label_smoothing", 0.1))
    MIXUP_ALPHA = float(CFG["fusion"].get("mixup_alpha", 0.4))

    EPOCHS = CFG["training"]["epochs"]
    BATCH_SIZE = CFG["training"]["batch_size"]
    LR = CFG["training"]["learning_rate"]
    WD = CFG["training"]["weight_decay"]
    NUM_WORKERS = CFG["training"]["num_workers"]
    GRAD_CLIP = CFG["training"]["grad_clip"]
    PATIENCE = CFG["training"]["early_stopping_patience"]

    DEVICE = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print(f"\n[MLP] Device              : {DEVICE}")
    print(f"[MLP] Fused input dim     : {pre.out_dim}")
    print(f"[MLP] Hidden dims         : {HIDDEN_DIMS}")
    print(f"[MLP] Dropout             : {DROPOUT}")
    print(f"[MLP] Label smoothing     : {LABEL_SMOOTH}")
    print(f"[MLP] Mixup alpha         : {MIXUP_ALPHA}")
    print(f"[MLP] Batch size          : {BATCH_SIZE}")
    print(f"[MLP] Epochs              : {EPOCHS}")
    print(f"[MLP] LR                  : {LR}")
    print(f"[MLP] Weight decay        : {WD}")
    print(f"[MLP] Patience            : {PATIENCE}")

    # Per-__getitem__ random-view dataset (train) / deterministic view 0 (val/test).
    train_td = _PreprocessedViewDataset(train_ds, pre, random_view=True)
    val_td   = _PreprocessedViewDataset(val_ds,   pre, random_view=False)
    test_td  = _PreprocessedViewDataset(test_ds,  pre, random_view=False)

    # Class-balanced sampler
    train_labels_list = train_ds.labels
    class_counts = np.bincount(train_labels_list, minlength=2)
    class_weights = 1.0 / class_counts.astype(np.float64)
    sample_weights = torch.tensor(
        [class_weights[l] for l in train_labels_list], dtype=torch.float64
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    print(
        f"[MLP] Class counts — No A-H: {class_counts[0]:,}  |  "
        f"With A-H: {class_counts[1]:,}"
    )

    # DataLoaders
    _drop_last = (len(train_td) % BATCH_SIZE) == 1
    train_loader = DataLoader(
        train_td, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, drop_last=_drop_last,
    )
    val_loader = DataLoader(val_td, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_td, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    print(
        f"[MLP] Train: {len(train_td):,} videos  |  "
        f"Val: {len(val_td):,}  |  Test: {len(test_td):,}"
    )

    # Model — single-vector input now (early fusion already concatenated).
    fusion_model = SimpleMLPFusionHead(
        input_dim=pre.out_dim, hidden_dims=HIDDEN_DIMS,
        num_classes=2, dropout=DROPOUT,
    ).to(DEVICE)

    trainable = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in fusion_model.parameters())
    print(f"[MLP] Architecture:\n{fusion_model}")
    print(f"[MLP] Parameters: {trainable:,} trainable  |  {total:,} total")

    criterion_train_hard = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    criterion_eval = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 1e-2
    )

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path = weights_path.parent / "fusion_training_curves.png"

    best_f1 = -1.0
    no_improve = 0
    history = {"tr_loss": [], "va_loss": [], "tr_f1": [], "va_f1": [], "va_acc": []}

    print("\n" + "=" * 70)
    print("Starting MLP fusion training ...")
    print("=" * 70)
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        fusion_model.train()
        tr_loss, tr_preds, tr_labels_ep = 0.0, [], []

        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS} train", leave=False)
        ):
            x, labels = batch
            x = x.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            if MIXUP_ALPHA > 0:
                mixed_x, soft_targets, _ = mixup_batch([x], labels, MIXUP_ALPHA, num_classes=2)
                train_logits = fusion_model(mixed_x[0])
                train_loss = soft_cross_entropy(train_logits, soft_targets)
            else:
                train_logits = fusion_model(x)
                train_loss = criterion_train_hard(train_logits, labels)

            train_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                fusion_model.parameters(), max_norm=GRAD_CLIP
            )
            optimizer.step()

            fusion_model.eval()
            with torch.no_grad():
                clean_logits = fusion_model(x)
                clean_loss = criterion_eval(clean_logits, labels)
            fusion_model.train()

            tr_loss += clean_loss.item() * len(labels)
            tr_preds.extend(clean_logits.argmax(1).cpu().tolist())
            tr_labels_ep.extend(labels.cpu().tolist())

            if epoch == 1 and batch_idx == 0:
                print(
                    f"\n  [Epoch 1, Batch 0] "
                    f"train_loss={train_loss.item():.4f}  "
                    f"clean_loss={clean_loss.item():.4f}  "
                    f"grad_norm={grad_norm:.4f}"
                )

        # Val
        fusion_model.eval()
        va_loss, va_preds, va_labels_ep = 0.0, [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch}/{EPOCHS} val  ", leave=False):
                x, labels = batch
                x = x.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = fusion_model(x)
                va_loss += criterion_eval(logits, labels).item() * len(labels)
                va_preds.extend(logits.argmax(1).cpu().tolist())
                va_labels_ep.extend(labels.cpu().tolist())

        scheduler.step()

        tr_f1 = f1_score(tr_labels_ep, tr_preds, average="macro", zero_division=0)
        va_f1 = f1_score(va_labels_ep, va_preds, average="macro", zero_division=0)
        va_acc = accuracy_score(va_labels_ep, va_preds)
        _tr_loss_avg = tr_loss / max(len(tr_labels_ep), 1)
        _va_loss_avg = va_loss / max(len(va_labels_ep), 1)

        history["tr_loss"].append(_tr_loss_avg)
        history["va_loss"].append(_va_loss_avg)
        history["tr_f1"].append(tr_f1)
        history["va_f1"].append(va_f1)
        history["va_acc"].append(va_acc)

        epoch_time = time.time() - epoch_start
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:>3}/{EPOCHS}  "
            f"tr_loss={_tr_loss_avg:.4f}  tr_f1={tr_f1:.4f}  "
            f"va_loss={_va_loss_avg:.4f}  va_f1={va_f1:.4f}  va_acc={va_acc:.4f}  "
            f"lr={cur_lr:.2e}  time={epoch_time:.0f}s"
        )
        print(
            f"         train preds: {dict(Counter(tr_preds))}  |  "
            f"val preds: {dict(Counter(va_preds))}"
        )

        if va_f1 > best_f1:
            best_f1 = va_f1
            # Persist BOTH the network state AND the fitted preprocessor so
            # inference is self-contained.
            torch.save(
                {"state_dict": fusion_model.state_dict(),
                 "input_dim": pre.out_dim,
                 "hidden_dims": HIDDEN_DIMS,
                 "dropout": DROPOUT},
                weights_path,
            )
            import joblib
            joblib.dump(pre, weights_path.with_suffix(".preproc.joblib"), compress=3)
            print(f"  ✓ New best val F1={best_f1:.4f} — saved {weights_path}")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  ✗ No improvement ({no_improve}/{PATIENCE})")
            if PATIENCE > 0 and no_improve >= PATIENCE:
                print(f"  ⛔ Early stopping at epoch {epoch}")
                break

    total_time = time.time() - train_start
    print(f"\nTraining done in {total_time / 60:.1f} min. Best val F1 = {best_f1:.4f}")
    plot_history(history, plot_path)

    # Test
    print("\n[Test] Loading best checkpoint and evaluating ...")
    ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    fusion_model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    fusion_model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test inference", leave=True):
            x, labels = batch
            x = x.to(DEVICE)
            preds = fusion_model(x).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    test_f1, test_acc = report_test("MLP", active_mods, all_preds, all_labels)
    print(f"  Trainable params       : {trainable:,}")
    print(f"  Best val F1            : {best_f1:.4f}")
    print(f"  Test F1                : {test_f1:.4f}")
    print(f"  Test accuracy          : {test_acc:.4f}")
    print(f"  Training time          : {total_time / 60:.1f} min")
    print(f"  Weights saved          : {weights_path}")
    print(f"  Preproc saved          : {weights_path.with_suffix('.preproc.joblib')}")
    print(f"  Curves saved           : {plot_path}")


# ── Classical (SVM / RF / GBM) training branch ────────────────────────────────


def run_classical(
    classifier_name: str,
    CFG: dict,
    pre: EarlyFusionPreprocessor,
    train_ds,
    val_ds,
    test_ds,
    active_mods: list,
    weights_path: Path,
):
    """
    Train an sklearn-style classifier (SVM / RF / GBM) on the EARLY-FUSED
    representation (StandardScaler per modality → concat → PCA).  Each
    training video contributes up to ``K_t * K_a * K_v`` rows via the view
    grid expansion (capped by ``[fusion] max_train_rows``).
    """
    import joblib

    SEED = CFG["training"]["seed"]
    fusion_cfg = CFG["fusion"]
    max_train_rows = int(fusion_cfg.get("max_train_rows", -1))

    # Build the early-fused design matrices.
    X_tr, y_tr = pre.transform_all_views(train_ds, max_rows=max_train_rows, seed=SEED)
    X_va = pre.transform_view0(val_ds)
    y_va = np.asarray(val_ds.labels, dtype=np.int64)
    X_te = pre.transform_view0(test_ds)
    y_te = np.asarray(test_ds.labels, dtype=np.int64)

    print(
        f"\n[{classifier_name}] X_train: {X_tr.shape}  "
        f"({X_tr.shape[0]:,} rows after view-expansion, fused dim = {pre.out_dim})"
    )
    print(f"[{classifier_name}] X_val  : {X_va.shape}")
    print(f"[{classifier_name}] X_test : {X_te.shape}")
    print(f"[{classifier_name}] y_train class balance: {dict(Counter(y_tr.tolist()))}")
    print(f"[{classifier_name}] y_val   class balance: {dict(Counter(y_va.tolist()))}")
    print(f"[{classifier_name}] y_test  class balance: {dict(Counter(y_te.tolist()))}")

    # Build classifier.
    clf = build_classical_classifier(classifier_name, fusion_cfg, SEED)
    print(f"[{classifier_name}] Estimator:\n  {clf}")

    # XGBoost-only: feed scale_pos_weight derived from train labels.
    fit_kwargs = {}
    if classifier_name == "gradient_boosting":
        lib = str(fusion_cfg.get("gradient_boosting", {}).get("library", "xgboost")).lower()
        if lib == "xgboost":
            pos = max(int((y_tr == 1).sum()), 1)
            neg = max(int((y_tr == 0).sum()), 1)
            clf.set_params(scale_pos_weight=neg / pos)
            print(
                f"[{classifier_name}] Applied scale_pos_weight = "
                f"{neg / pos:.3f}  (neg/pos in flat train set)"
            )

    # Fit.
    train_start = time.time()
    print(f"\n[{classifier_name}] Fitting ...")
    clf.fit(X_tr, y_tr, **fit_kwargs)
    train_time = time.time() - train_start
    print(f"[{classifier_name}] Fit done in {train_time:.1f}s")

    # Val + test prediction.
    va_preds = clf.predict(X_va).astype(int).tolist()
    va_f1 = f1_score(y_va, va_preds, average="macro", zero_division=0)
    va_acc = accuracy_score(y_va, va_preds)
    print(
        f"[{classifier_name}] Val  macro-F1: {va_f1:.4f}  |  acc: {va_acc:.4f}  "
        f"|  pred dist: {dict(Counter(va_preds))}"
    )

    te_preds = clf.predict(X_te).astype(int).tolist()
    test_f1, test_acc = report_test(
        classifier_name, active_mods, te_preds, y_te.tolist()
    )

    # Persist with joblib alongside the fitted preprocessor + metadata.
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = weights_path.with_suffix(".joblib")
    joblib.dump(
        {
            "classifier_name": classifier_name,
            "active_modalities": active_mods,
            "preprocessor": pre,       # StandardScaler(s) + optional PCA
            "estimator": clf,
            "input_dim": pre.out_dim,
        },
        out_path,
        compress=3,
    )
    print(f"  Weights saved          : {out_path}")
    print(f"  Val   F1               : {va_f1:.4f}")
    print(f"  Val   accuracy         : {va_acc:.4f}")
    print(f"  Test  F1               : {test_f1:.4f}")
    print(f"  Test  accuracy         : {test_acc:.4f}")
    print(f"  Fit time               : {train_time:.1f}s")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    from load_dataset import get_cached_multimodal_splits

    # ── Config ────────────────────────────────────────────────────────────────
    CFG = toml.load("config.toml")

    TEXT_DIM = CFG["fusion"]["text_dim"]
    AUDIO_DIM = CFG["fusion"]["audio_dim"]
    VIDEO_DIM = CFG["fusion"]["video_dim"]
    ACTIVE_MODS = CFG["fusion"].get("active_modalities", ["text", "audio", "video"])
    CLASSIFIER = str(CFG["fusion"].get("classifier", "mlp")).lower()

    _cache_dir_raw = CFG["fusion"].get("cache_dir", "multimodal/cache")
    CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(REPO_ROOT / _cache_dir_raw)))

    SEED = CFG["training"]["seed"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    FUSION_WEIGHTS = (
        REPO_ROOT
        / "multimodal"
        / CFG["output"]["weights_dir"]
        / CFG["output"]["weights_name"]
    )

    # ── Modality dims ─────────────────────────────────────────────────────────
    ALL_MOD_DIMS = {"text": TEXT_DIM, "audio": AUDIO_DIM, "video": VIDEO_DIM}
    MOD_DIMS = [ALL_MOD_DIMS[m] for m in ACTIVE_MODS]
    INPUT_DIM = sum(MOD_DIMS)

    print("=" * 70)
    print("Multimodal Fusion-Head Training — cache-driven")
    print("=" * 70)
    print(f"Classifier          : {CLASSIFIER}")
    print(f"Cache dir           : {CACHE_DIR}")
    print(f"Active modalities   : {ACTIVE_MODS}")
    print(f"Fusion input dim    : {' + '.join(str(d) for d in MOD_DIMS)} = {INPUT_DIM}")
    print(f"Seed                : {SEED}")
    print(f"Fusion weights path : {FUSION_WEIGHTS}")
    print("=" * 70)

    # ── Load cached datasets (used by both branches) ──────────────────────────
    print("\n[Cache] Loading multi-view embedding cache ...")
    # For classical classifiers we don't need per-__getitem__ random view —
    # build_classical_matrices reads ``ds.text/audio/video`` tensors directly.
    splits = get_cached_multimodal_splits(
        cache_dir=CACHE_DIR,
        train_random_view=(CLASSIFIER == "mlp"),
        expected_fingerprint=None,
    )
    train_ds, val_ds, test_ds = splits["train"], splits["val"], splits["test"]

    for ds_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        cnt = Counter(ds.labels)
        n = len(ds.labels)
        pct_pos = cnt.get(1, 0) / n * 100 if n > 0 else 0
        K_info = f"K_t={ds.K_t} K_a={ds.K_a} K_v={ds.K_v}"
        print(
            f"  [{ds_name:5s}]  {n:>5,} videos  |  "
            f"No A-H: {cnt.get(0, 0):,}  With A-H: {cnt.get(1, 0):,}  "
            f"({pct_pos:.1f}% positive)  |  {K_info}"
        )

    # ── Early-fusion preprocessor (StandardScaler per modality → concat → PCA) ──
    pca_components_raw = CFG["fusion"].get("pca_components", 0)
    if isinstance(pca_components_raw, str):
        if pca_components_raw.strip().lower() in ("", "none", "false", "0"):
            pca_components_cfg = None
        else:
            try:
                pca_components_cfg = float(pca_components_raw)
                if pca_components_cfg >= 1.0:
                    pca_components_cfg = int(pca_components_cfg)
            except ValueError:
                pca_components_cfg = None
    elif isinstance(pca_components_raw, bool):
        pca_components_cfg = None if not pca_components_raw else None
    elif isinstance(pca_components_raw, (int, float)):
        pca_components_cfg = pca_components_raw if pca_components_raw and pca_components_raw > 0 else None
    else:
        pca_components_cfg = None
    pca_whiten_cfg = bool(CFG["fusion"].get("pca_whiten", False))

    print("\n[Preproc] Fitting EarlyFusionPreprocessor on TRAIN ...")
    print(f"  StandardScaler per modality (fitted on all N*K train views)")
    print(f"  PCA components       : {pca_components_cfg!r} (whiten={pca_whiten_cfg})")
    pre = EarlyFusionPreprocessor(
        active_mods=ACTIVE_MODS,
        pca_components=pca_components_cfg,
        pca_whiten=pca_whiten_cfg,
        seed=SEED,
    ).fit(train_ds)
    print(f"  Concat raw dim       : {sum(MOD_DIMS)}")
    print(f"  Fused output dim     : {pre.out_dim}")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if CLASSIFIER == "mlp":
        run_mlp(
            CFG, pre, train_ds, val_ds, test_ds,
            active_mods=ACTIVE_MODS,
            weights_path=FUSION_WEIGHTS,
        )
    elif CLASSIFIER in ("svm", "random_forest", "gradient_boosting"):
        run_classical(
            CLASSIFIER, CFG, pre, train_ds, val_ds, test_ds,
            active_mods=ACTIVE_MODS,
            weights_path=FUSION_WEIGHTS,
        )
    else:
        raise ValueError(
            f"Unknown [fusion] classifier = {CLASSIFIER!r}.  "
            "Choose from: mlp, svm, random_forest, gradient_boosting."
        )

    print(f"\n{'=' * 70}")
    print("Multimodal Fusion Training Summary")
    print("=" * 70)
    print(f"  Classifier             : {CLASSIFIER}")
    print(f"  Active modalities      : {ACTIVE_MODS}")
    print(f"  Concat raw dim         : {' + '.join(str(d) for d in MOD_DIMS)} = {INPUT_DIM}")
    print(f"  Fused output dim       : {pre.out_dim}")
    print(f"  PCA components         : {pca_components_cfg!r} (whiten={pca_whiten_cfg})")
    print("=" * 70)


if __name__ == "__main__":
    main()
