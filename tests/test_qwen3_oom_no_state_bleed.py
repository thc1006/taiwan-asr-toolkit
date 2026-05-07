# -*- coding: utf-8 -*-
"""Regression test for v0.5.5 H3: Qwen3 OOM retry must NOT mutate
self.cfg.batch_size.

Pre-v0.5.5 behaviour (qwen3.py:272 and :379):

    cur_bs = max(1, cur_bs // 2)
    pending_idxs = [...]
    self.cfg.batch_size = cur_bs       # <-- writes back to instance config

Effect: in a multi-file run `asr-qwen3 a.mp3 b.mp3 c.mp3`, if file `a` OOMs at
batch=48 and recovers at batch=24, files `b` and `c` then start at batch=24
even when they would have fit at 48. State bleed across files in the same
transcribe() / transcribe_files() session.

The fix is to keep `cur_bs` strictly local — drop the `self.cfg.batch_size = cur_bs`
write-back line. The retry/redistribute logic continues to use `cur_bs` locally.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from taiwan_asr.qwen3 import HwConfig, Qwen3ASR


def _cpu_cfg(batch_size: int = 48) -> HwConfig:
    return HwConfig(
        device="cpu", dtype=torch.float32, batch_size=batch_size,
        chunk_sec=4.0, attn_impl="eager", compile_mode=None, desc="test",
    )


def _instance(batch_size: int = 48) -> Qwen3ASR:
    asr = Qwen3ASR(_cpu_cfg(batch_size), s2tw_enabled=False, aligner_enabled=False)
    asr.model = MagicMock()
    return asr


@pytest.mark.fast
def test_single_file_oom_retry_keeps_self_cfg_unchanged(monkeypatch):
    """transcribe(): OOM-then-success on a single file must leave self.cfg.batch_size alone."""
    asr = _instance(batch_size=48)
    original = asr.cfg.batch_size

    _aud = (np.zeros(int(3 * 16000), dtype=np.float32), 16000)
    fake_chunks = [
        {"start": 0.0, "end": 3.0, "audio": _aud},
        {"start": 3.0, "end": 6.0, "audio": _aud},
        {"start": 6.0, "end": 9.0, "audio": _aud},
    ]
    asr.vad = MagicMock(_using="test")
    asr.vad.speech_chunks = MagicMock(return_value=fake_chunks)

    call_count = {"n": 0}
    def fake_batch(audios, lang):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise torch.cuda.OutOfMemoryError("forced OOM (test)")
        from types import SimpleNamespace
        return [SimpleNamespace(text="hi", language="zh") for _ in audios]
    monkeypatch.setattr(asr, "_transcribe_batch", fake_batch)

    audio = np.zeros(16000 * 9, dtype=np.float32)
    with patch("taiwan_asr.qwen3.AudioIO") as audio_io:
        audio_io.decode_to_array.return_value = (audio, 9.0)
        asr.transcribe("any.wav")

    assert asr.cfg.batch_size == original, (
        f"self.cfg.batch_size leaked from {original} to {asr.cfg.batch_size}. "
        f"OOM retry must keep cur_bs local; subsequent files would inherit "
        f"the reduced batch even when they would have fit."
    )


@pytest.mark.fast
def test_pool_oom_retry_keeps_self_cfg_unchanged(monkeypatch):
    """transcribe_files(): same invariant for the cross-file pool path."""
    asr = _instance(batch_size=48)
    original = asr.cfg.batch_size

    fake_chunks = [
        {"start": 0.0, "end": 3.0, "audio": (np.zeros(int(3 * 16000), dtype=np.float32), 16000)}
        for _ in range(4)
    ]
    asr.vad = MagicMock(_using="test")
    asr.vad.speech_chunks = MagicMock(return_value=fake_chunks)

    call_count = {"n": 0}
    def fake_batch(audios, lang):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise torch.cuda.OutOfMemoryError("forced OOM (test)")
        from types import SimpleNamespace
        return [SimpleNamespace(text="hi", language="zh") for _ in audios]
    monkeypatch.setattr(asr, "_transcribe_batch", fake_batch)

    audio = np.zeros(16000 * 12, dtype=np.float32)
    with patch("taiwan_asr.qwen3.AudioIO") as audio_io:
        audio_io.decode_to_array.return_value = (audio, 12.0)
        asr.transcribe_files(["a.wav", "b.wav"])

    assert asr.cfg.batch_size == original, (
        f"Pool path: self.cfg.batch_size leaked from {original} to "
        f"{asr.cfg.batch_size}. The retry/redistribute logic should use a "
        f"local cur_bs only."
    )
