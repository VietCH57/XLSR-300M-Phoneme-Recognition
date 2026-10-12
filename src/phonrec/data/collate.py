"""Batch padding."""

from __future__ import annotations

from typing import Dict, List

import torch


class CTCCollator:
    """Pads waveforms with 0 (+ attention mask) and labels with ``label_pad_id``."""

    def __init__(self, label_pad_id: int = -100):
        self.label_pad_id = label_pad_id

    def __call__(self, items: List[Dict]) -> Dict[str, torch.Tensor]:
        bsz = len(items)
        wav_lens = [len(it["input_values"]) for it in items]
        lab_lens = [len(it["labels"]) for it in items]
        max_wav, max_lab = max(wav_lens), max(max(lab_lens), 1)

        input_values = torch.zeros(bsz, max_wav, dtype=torch.float32)
        attention_mask = torch.zeros(bsz, max_wav, dtype=torch.long)
        labels = torch.full((bsz, max_lab), self.label_pad_id, dtype=torch.long)
        for b, it in enumerate(items):
            n = wav_lens[b]
            input_values[b, :n] = torch.from_numpy(it["input_values"])
            attention_mask[b, :n] = 1
            if lab_lens[b]:
                labels[b, : lab_lens[b]] = torch.tensor(it["labels"], dtype=torch.long)

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": labels,
            "label_lengths": torch.tensor(lab_lens, dtype=torch.long),
            "indices": torch.tensor([it["index"] for it in items], dtype=torch.long),
        }
