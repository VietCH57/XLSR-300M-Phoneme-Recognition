"""Evaluation / inference loop shared by train.py and infer.py."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from phonrec.data import CTCCollator, DynamicBatchSampler, PhonemeDataset, PhonemeTokenizer
from phonrec.decoding import ctc_greedy_decode
from phonrec.metrics import PERMeter
from phonrec.utils.misc import autocast_ctx

logger = logging.getLogger(__name__)


def make_eval_loader(ds: PhonemeDataset, max_batch_seconds: float, max_batch_size: int,
                     num_workers: int) -> DataLoader:
    sampler = DynamicBatchSampler(ds.durations, max_batch_seconds, max_batch_size, shuffle=False)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=CTCCollator(), num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=False)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    tokenizer: PhonemeTokenizer,
    device: torch.device,
    precision: str = "bf16",
    compute_loss: bool = True,
    has_reference: bool = True,
    keep_predictions: bool = False,
) -> Dict[str, Any]:
    """Run greedy CTC decoding over ``loader``.

    Returns {"meter": PERMeter, "loss": float|None, "metrics": {...}, "predictions": [...]}.
    """
    ds: PhonemeDataset = loader.dataset
    was_training = model.training
    model.eval()
    meter = PERMeter()
    loss_sum, loss_n = 0.0, 0
    preds: List[Dict[str, Any]] = []
    t0 = time.time()

    for batch in loader:
        inputs = batch["input_values"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device) if (compute_loss and has_reference) else None
        lab_lens = batch["label_lengths"].to(device) if labels is not None else None
        with autocast_ctx(device, precision):
            out = model(inputs, mask, labels=labels, label_lengths=lab_lens)
        if labels is not None:
            bsz = inputs.size(0)
            loss_sum += float(out["loss"]) * bsz
            loss_n += bsz

        hyps = ctc_greedy_decode(out["log_probs"], out["output_lengths"], tokenizer.blank_id)
        for idx, hyp_ids in zip(batch["indices"].tolist(), hyps):
            hyp = tokenizer.decode(hyp_ids)
            ref = tokenizer.decode(ds.ids[idx], skip_blank=False)
            dom = ds.domains[idx]
            rec: Dict[str, Any] = {"hyp": " ".join(hyp)}
            if has_reference:
                c = meter.update(ref, hyp, dom)
                rec.update(per=round(c.per, 4), sub=c.sub, dele=c.dele, ins=c.ins, ref_phones=c.ref_len)
            if keep_predictions:
                preds.append({"row": int(ds.df.at[idx, "row"]), "audio": ds.audio[idx], "domain": dom,
                              "ref": " ".join(ref), **rec})

    if was_training:
        model.train()

    loss: Optional[float] = loss_sum / loss_n if loss_n else None
    metrics: Dict[str, Any] = meter.flat_metrics() if has_reference else {}
    if loss is not None:
        metrics["loss"] = loss
    preds.sort(key=lambda r: r["row"])
    return {"meter": meter, "loss": loss, "metrics": metrics, "predictions": preds,
            "seconds": time.time() - t0}
