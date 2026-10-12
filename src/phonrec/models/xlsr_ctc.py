"""XLS-R (wav2vec 2.0) backbone + linear CTC head for phoneme recognition.

    waveform (16 kHz)
      -> CNN feature encoder          (part of XLS-R, frozen by default)
      -> feature projection + 24-layer Transformer (XLS-R-300M)
      -> dropout -> Linear(hidden, |vocab|) -> log_softmax
      -> CTC loss (training) / CTC greedy decoding (inference)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model

logger = logging.getLogger(__name__)

ENCODER_CONFIG_FILE = "encoder_config.json"
HEAD_CONFIG_FILE = "head_config.json"
WEIGHTS_FILE = "model.pt"


class XLSRPhonemeCTC(nn.Module):
    def __init__(
        self,
        encoder: Wav2Vec2Model,
        vocab_size: int,
        blank_id: int,
        final_dropout: float = 0.1,
        ctc_reduction: str = "mean",
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(final_dropout)
        self.head = nn.Linear(encoder.config.hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.final_dropout = final_dropout
        self.ctc_reduction = ctc_reduction
        self._feature_encoder_frozen = False

    # ------------------------------------------------------------ builders
    @classmethod
    def from_pretrained_backbone(cls, cfg, vocab_size: int, blank_id: int) -> "XLSRPhonemeCTC":
        """Load the pretrained XLS-R encoder and attach a freshly initialised head."""
        enc_cfg = Wav2Vec2Config.from_pretrained(
            cfg.pretrained_name,
            attention_dropout=cfg.attention_dropout,
            hidden_dropout=cfg.hidden_dropout,
            activation_dropout=cfg.activation_dropout,
            feat_proj_dropout=cfg.feat_proj_dropout,
            layerdrop=cfg.layerdrop,
            mask_time_prob=cfg.mask_time_prob,
            mask_time_length=cfg.mask_time_length,
            mask_feature_prob=cfg.mask_feature_prob,
            mask_feature_length=cfg.mask_feature_length,
            apply_spec_augment=(cfg.mask_time_prob > 0 or cfg.mask_feature_prob > 0),
        )
        encoder = Wav2Vec2Model.from_pretrained(cfg.pretrained_name, config=enc_cfg)
        model = cls(encoder, vocab_size, blank_id, cfg.final_dropout, cfg.ctc_reduction)
        if cfg.freeze_feature_encoder:
            model.freeze_feature_encoder()
        if cfg.gradient_checkpointing:
            model.enable_gradient_checkpointing()
        return model

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str | Path, map_location="cpu") -> "XLSRPhonemeCTC":
        """Rebuild a fine-tuned model offline (no hub download needed)."""
        ckpt_dir = Path(ckpt_dir)
        enc_cfg = Wav2Vec2Config.from_json_file(str(ckpt_dir / ENCODER_CONFIG_FILE))
        with open(ckpt_dir / HEAD_CONFIG_FILE) as f:
            head = json.load(f)
        model = cls(Wav2Vec2Model(enc_cfg), head["vocab_size"], head["blank_id"],
                    head.get("final_dropout", 0.0), head.get("ctc_reduction", "mean"))
        state = torch.load(ckpt_dir / WEIGHTS_FILE, map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model

    def save(self, ckpt_dir: str | Path) -> None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.config.to_json_file(str(ckpt_dir / ENCODER_CONFIG_FILE))
        with open(ckpt_dir / HEAD_CONFIG_FILE, "w") as f:
            json.dump({"vocab_size": self.vocab_size, "blank_id": self.blank_id,
                       "final_dropout": self.final_dropout,
                       "ctc_reduction": self.ctc_reduction}, f, indent=2)
        torch.save(self.state_dict(), ckpt_dir / WEIGHTS_FILE)

    # ------------------------------------------------------- freeze utils
    def freeze_feature_encoder(self) -> None:
        self.encoder.freeze_feature_encoder()
        self._feature_encoder_frozen = True

    def set_backbone_trainable(self, trainable: bool) -> None:
        """(Un)freeze everything in XLS-R except the CNN feature encoder if it is frozen."""
        for name, p in self.encoder.named_parameters():
            if self._feature_encoder_frozen and name.startswith("feature_extractor."):
                continue
            p.requires_grad = trainable

    def enable_gradient_checkpointing(self) -> None:
        try:  # non-reentrant checkpointing also works while the backbone is frozen
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:  # older transformers
            self.encoder.gradient_checkpointing_enable()

    def head_parameters(self):
        return list(self.head.parameters())

    def backbone_parameters(self):
        return list(self.encoder.parameters())

    # ------------------------------------------------------------ forward
    def output_lengths(self, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = self.encoder._get_feat_extract_output_lengths(attention_mask.sum(-1))
        return lengths.to(torch.long)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(input_values, attention_mask=attention_mask).last_hidden_state
        logits = self.head(self.dropout(hidden))
        log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T, V), fp32 for CTC stability
        out_lens = self.output_lengths(attention_mask)
        out = {"log_probs": log_probs, "output_lengths": out_lens}

        if labels is not None:
            if label_lengths is None:
                label_lengths = (labels >= 0).sum(-1)
            targets = labels[labels >= 0]
            with torch.autocast(device_type=log_probs.device.type, enabled=False):
                out["loss"] = F.ctc_loss(
                    log_probs.transpose(0, 1),
                    targets,
                    out_lens,
                    label_lengths,
                    blank=self.blank_id,
                    reduction=self.ctc_reduction,
                    zero_infinity=True,
                )
        return out
