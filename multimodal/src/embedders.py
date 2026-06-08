"""
All embedders:
  - Load pretrained HuggingFace weights then overlay a fine-tuned ``.pth``
    checkpoint (see per-class docstrings for key-remapping subtleties).
  - Freeze every parameter and call ``.eval()``.
  - Provide a ``.dim`` property for the output embedding size.

These classes used to live in ``multimodal/src/fusion_training.py``; they were
moved here so the *extraction* script can use them without pulling in the
fusion-head / training loop code.
"""

from pathlib import Path

import torch
import torch.nn as nn


# ── Text ──────────────────────────────────────────────────────────────────────

class TextEmbedder:
    """
    Wraps a frozen DistilBERT model and extracts the CLS token embedding.

    Key remap
    ---------
    The checkpoint saved by ``distilbert_training.py`` is a
    ``DistilBertForSequenceClassification`` state dict whose encoder keys are
    prefixed with ``distilbert.`` (e.g.
    ``distilbert.embeddings.word_embeddings.weight``).  ``DistilBertModel``
    expects the same keys *without* that prefix.  We remap before loading so
    the fine-tuned encoder weights are actually used.

    Output dim: ``hidden_size`` = 768.
    """

    def __init__(self, model_name: str, weights_path, device, max_length: int = 128):
        from transformers import DistilBertModel, AutoTokenizer

        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(device)

        if weights_path is not None and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=device, weights_only=True)
            remapped = {}
            skipped = []
            for k, v in state.items():
                if k.startswith("distilbert."):
                    remapped[k[len("distilbert."):]] = v
                else:
                    skipped.append(k)
            missing, unexpected = self.model.load_state_dict(remapped, strict=False)
            print(
                f"[TextEmbedder] Loaded {len(remapped)}/{len(state)} keys from "
                f"{weights_path}  (skipped head: {skipped}  "
                f"missing={missing}  unexpected={unexpected})"
            )
        else:
            print(
                f"[TextEmbedder] Weights not found at {weights_path} — "
                "using pretrained HuggingFace weights"
            )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size  # 768

    def embed(self, texts: list) -> torch.Tensor:
        """Return (N, 768) CLS embeddings for a list of strings."""
        with torch.no_grad():
            enc = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            return out.last_hidden_state[:, 0, :]  # (N, 768)


# ── Audio ─────────────────────────────────────────────────────────────────────

class AudioEmbedder:
    """
    Wraps a frozen ``Wav2Vec2ForSequenceClassification`` and extracts the
    projected embedding (before the final linear classifier).

    Output dim: ``classifier_proj_size`` = 256.
    """

    def __init__(
        self,
        model_name: str,
        weights_path,
        device,
        sample_rate: int = 16000,
        max_length_sec: float = 30.0,
    ):
        from transformers import (
            Wav2Vec2ForSequenceClassification,
            Wav2Vec2FeatureExtractor,
        )

        self.device = device
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_length_sec)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            ignore_mismatched_sizes=True,
        ).to(device)

        if weights_path is not None and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=device, weights_only=True)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            n_loaded = len(state) - len(unexpected)
            print(
                f"[AudioEmbedder] Loaded {n_loaded}/{len(state)} keys from {weights_path}"
                f"  (missing={len(missing)}, unexpected={len(unexpected)})"
            )
        else:
            print(
                f"[AudioEmbedder] Weights not found at {weights_path} — "
                "using pretrained HuggingFace weights"
            )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def dim(self) -> int:
        return self.model.config.classifier_proj_size  # 256

    def embed_from_encoded(self, encoded: dict) -> torch.Tensor:
        """
        Extract projected embedding from a pre-encoded batch dict.

        Parameters
        ----------
        encoded : dict with keys ``input_values`` (and optionally
                  ``attention_mask``), values are tensors already on device.

        Returns
        -------
        Tensor of shape (N, classifier_proj_size).
        """
        with torch.no_grad():
            outputs = self.model.wav2vec2(**encoded)
            hidden = outputs.last_hidden_state  # (N, T, hidden_size)

            if "attention_mask" in encoded:
                lengths = self.model._get_feat_extract_output_lengths(
                    encoded["attention_mask"].sum(-1)
                )
                # Vectorised padding mask (avoids a Python loop over the batch).
                B, T = hidden.shape[:2]
                ar = torch.arange(T, device=hidden.device).unsqueeze(0).expand(B, T)
                padding_mask = ar < lengths.to(hidden.device).long().unsqueeze(1)
                hidden = hidden * padding_mask.unsqueeze(-1).float()
                pooled = hidden.sum(1) / padding_mask.sum(1, keepdim=True).float().clamp(min=1)
            else:
                pooled = hidden.mean(1)

            projected = self.model.projector(pooled)
            return projected

    def embed_waveforms(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Convenience: run feature extractor → model on a batch of raw waveforms.

        Parameters
        ----------
        waveforms : Tensor of shape (B, num_samples) on CPU or any device.

        Returns
        -------
        Tensor of shape (B, classifier_proj_size).
        """
        # Feature extractor expects a list of 1-D numpy arrays / tensors.
        wf_list = [w.detach().cpu().numpy() for w in waveforms]
        enc = self.feature_extractor(
            wf_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_samples,
            truncation=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return self.embed_from_encoded(enc)


# ── Video ─────────────────────────────────────────────────────────────────────

class _SwinVideoClassifier(nn.Module):
    """Mirror of the architecture in ``video/src/swin_training.py``."""

    def __init__(self, model_name: str, num_labels: int = 2):
        super().__init__()
        from transformers import SwinForImageClassification

        self.swin = SwinForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        hidden_size = self.swin.config.hidden_size
        self.swin.classifier = nn.Identity()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return temporally mean-pooled embedding (B, hidden_size)."""
        B, T, C, H, W = pixel_values.shape
        flat = pixel_values.reshape(B * T, C, H, W)
        outputs = self.swin(pixel_values=flat)
        features = outputs.logits           # (B*T, hidden_size) via Identity
        features = features.view(B, T, -1)  # (B, T, hidden_size)
        return features.mean(dim=1)         # (B, hidden_size)


class VideoEmbedder:
    """
    Wraps a frozen ``_SwinVideoClassifier`` and extracts the temporally
    mean-pooled Swin backbone embedding (before the final linear classifier).

    Output dim: ``hidden_size`` = 768 (Swin-Tiny).
    """

    def __init__(self, model_name: str, weights_path, device):
        self.device = device
        self.model = _SwinVideoClassifier(model_name, num_labels=2).to(device)

        if weights_path is not None and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=device, weights_only=True)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            n_loaded = len(state) - len(unexpected)
            print(
                f"[VideoEmbedder] Loaded {n_loaded}/{len(state)} keys from {weights_path}"
                f"  (missing={len(missing)}, unexpected={len(unexpected)})"
            )
        else:
            print(
                f"[VideoEmbedder] Weights not found at {weights_path} — "
                "using pretrained HuggingFace weights"
            )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def dim(self) -> int:
        return self.model.swin.config.hidden_size  # 768

    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values : (B, T, C, H, W) — already on device

        Returns
        -------
        (B, hidden_size) temporal mean-pooled embedding
        """
        with torch.no_grad():
            return self.model(pixel_values)
