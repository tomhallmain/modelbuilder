"""
``encoder_finetune``: full fine-tune of a Hugging Face encoder with a one-logit head.

Loads ``AutoModelForSequenceClassification`` with ``num_labels=1`` and trains it with a
weighted binary cross-entropy (the per-row class × tier weights), AdamW, and linear warmup
then linear decay. Val is scored after every epoch (or every ``eval_every_steps``); the
best checkpoint by the selection metric is kept in CPU memory and restored at the end, and
training stops after ``patience`` evaluations without improvement.
"""

from __future__ import annotations

import contextlib
import math
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from mb.cancellation import check_cancel_event
from mb.models.text_backends._torch_common import (
    EarlyStopper,
    autocast_dtype,
    import_torch,
    import_transformers,
    seed_everything,
    weighted_bce,
)
from mb.models.text_backends.base import ProgressFn, TextBackend, TextFitContext, TextFitData
from mb.models.types import TextBackendType
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)


class EncoderFinetuneBackend(TextBackend):
    name = TextBackendType.ENCODER_FINETUNE
    neural = True
    default_model_id = "answerdotai/ModernBERT-base"
    default_lr = 3.0e-5
    default_epochs = 3
    default_batch_size = 256
    option_defaults = {
        # auto (bf16 on CUDA when supported) | bf16 | fp16 | fp32
        "precision": "auto",
        # 0 = evaluate on val once per epoch.
        "eval_every_steps": 0,
        "eval_batch_size": 512,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": False,
    }

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_id: Optional[str] = None
        self.max_length = 64
        self.device = "cpu"
        self.options: Dict[str, Any] = dict(self.option_defaults)

    def _load_pretrained(self, source: str) -> None:
        transformers = import_transformers()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(source)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(source, num_labels=1)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)

    def _encode(self, texts: Sequence[str]):
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

    def count_truncated(self, texts: Sequence[str]) -> Optional[int]:
        n = 0
        for start in range(0, len(texts), 10_000):
            enc = self.tokenizer(list(texts[start : start + 10_000]), truncation=False)
            n += sum(1 for ids in enc["input_ids"] if len(ids) > self.max_length)
        return n

    def fit(self, train: TextFitData, val: TextFitData, ctx: TextFitContext) -> Dict[str, Any]:
        torch = import_torch()
        transformers = import_transformers()
        seed_everything(ctx.seed)
        self.options = dict(ctx.options)
        self.model_id = ctx.model_id
        self.max_length = ctx.max_length
        self.device = ctx.device
        ctx.report(_("Loading {model}…").format(model=self.model_id), None)
        self._load_pretrained(self.model_id)
        if self.options["gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable()

        batch_size = int(ctx.batch_size)
        epochs = int(ctx.epochs)
        n = len(train.texts)
        steps_per_epoch = max(1, math.ceil(n / batch_size))
        total_steps = steps_per_epoch * epochs
        warmup = int(round(ctx.warmup_ratio * total_steps))
        eval_every = int(self.options["eval_every_steps"]) or steps_per_epoch

        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
        params = [
            {
                "params": [p for nm, p in self.model.named_parameters() if not nm.endswith(no_decay)],
                "weight_decay": ctx.weight_decay,
            },
            {
                "params": [p for nm, p in self.model.named_parameters() if nm.endswith(no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(params, lr=float(ctx.lr))
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
        amp = autocast_dtype(self.device, str(self.options["precision"]))
        scaler = torch.cuda.amp.GradScaler() if amp == torch.float16 else None

        labels = torch.as_tensor(train.labels, dtype=torch.float32)
        weights = torch.as_tensor(train.weights, dtype=torch.float32)
        rng = np.random.default_rng(ctx.seed)
        stopper = EarlyStopper(patience=ctx.patience)
        best_state: Optional[Dict[str, Any]] = None
        step = 0
        stop = False
        for epoch in range(epochs):
            order = rng.permutation(n)
            self.model.train()
            for b in range(steps_per_epoch):
                check_cancel_event(ctx.cancel_event)
                idx = order[b * batch_size : (b + 1) * batch_size]
                enc = self._encode([train.texts[i] for i in idx])
                t_idx = torch.from_numpy(idx)
                y = labels[t_idx].to(self.device)
                w = weights[t_idx].to(self.device)
                with _autocast(torch, amp):
                    logits = self.model(**enc).logits[:, 0]
                loss = weighted_bce(logits, y, w)
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.options["max_grad_norm"]))
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                step += 1
                if step % 20 == 0 or step == total_steps:
                    ctx.report(
                        _("Epoch {e}/{E} — step {s}/{S} — loss {l:.4f}").format(
                            e=epoch + 1, E=epochs, s=step, S=total_steps, l=loss.item()
                        ),
                        step / total_steps,
                    )
                if step % eval_every == 0 or step == total_steps:
                    probs = self.predict_proba(val.texts, cancel_event=ctx.cancel_event)
                    metric = ctx.selection_metric(probs)
                    improved = stopper.update(metric, step, step / steps_per_epoch)
                    logger.info(
                        "encoder_finetune: step %d epoch %.2f val metric %.6f%s",
                        step,
                        step / steps_per_epoch,
                        metric,
                        " (best)" if improved else "",
                    )
                    if improved:
                        best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}
                    self.model.train()
                    if stopper.should_stop:
                        stop = True
                        break
            if stop:
                logger.info("encoder_finetune: early stop after %d evaluations without improvement", stopper.bad_evals)
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return {"steps": step, "steps_per_epoch": steps_per_epoch, **stopper.summary()}

    def predict_proba(
        self,
        texts: Sequence[str],
        *,
        cancel_event: Optional[threading.Event] = None,
        progress: Optional[ProgressFn] = None,
    ) -> np.ndarray:
        torch = import_torch()
        bs = int(self.options["eval_batch_size"])
        amp = autocast_dtype(self.device, str(self.options["precision"]))
        out = np.empty(len(texts), dtype=np.float64)
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(texts), bs):
                check_cancel_event(cancel_event)
                chunk = texts[start : start + bs]
                enc = self._encode(chunk)
                with _autocast(torch, amp):
                    logits = self.model(**enc).logits[:, 0]
                out[start : start + len(chunk)] = torch.sigmoid(logits.float()).cpu().numpy()
                if progress is not None and (start // bs) % 50 == 0:
                    progress(_("Scoring…"), min(1.0, (start + len(chunk)) / max(len(texts), 1)))
        return out

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_length": self.max_length,
            "options": self.options,
            "base_revision": getattr(self.model.config, "_commit_hash", None),
        }

    def save_weights(self, model_dir: Path) -> None:
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

    @classmethod
    def load_weights(cls, model_dir: Path, meta: Dict[str, Any], device: str) -> EncoderFinetuneBackend:
        b = cls()
        b.model_id = meta.get("model_id")
        b.max_length = int(meta.get("max_length", 64))
        b.options = {**cls.option_defaults, **(meta.get("options") or {})}
        b.device = device
        b._load_pretrained(str(model_dir))
        b.model.eval()
        return b


def _autocast(torch, dtype):
    return torch.autocast(device_type="cuda", dtype=dtype) if dtype is not None else contextlib.nullcontext()


BACKEND = EncoderFinetuneBackend
