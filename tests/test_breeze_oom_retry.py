# -*- coding: utf-8 -*-
"""Regression test for v0.5.5 S2: Breeze backend must retry with halved
batch_size on torch.cuda.OutOfMemoryError, mirroring qwen3.py:262-273.

Pre-v0.5.5: breeze.py:277-283 made the inference call unguarded; any OOM
propagated to the user, killing the whole transcription job. qwen3 had
graceful retry; breeze did not. Asymmetric.

The retry must use a LOCAL counter, NOT self.cfg.batch_size — see H3 for
the related qwen3 state-bleed bug.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("faster_whisper")

from taiwan_asr.breeze import FasterWhisperBackend, HwConfig
from taiwan_asr.common import S2TW


def _cpu_cfg(batch_size: int = 32) -> HwConfig:
    return HwConfig(
        device="cpu", cuda_idx=0, dtype=torch.float32, ct2_compute="float32",
        batch_size=batch_size, cpu_threads=1, num_workers=1,
        chunk_sec=4.0, pad_sec=0.4, desc="test cpu",
    )


def _backend(batch_size: int = 32) -> FasterWhisperBackend:
    b = FasterWhisperBackend(cfg=_cpu_cfg(batch_size), beam=1, s2tw=S2TW(False))
    b.model = MagicMock()
    b.batched = MagicMock()
    b.vad = MagicMock(_using="test")
    fake_chunks = [{"start": 0.0, "end": 3.0}, {"start": 4.0, "end": 7.0}]
    b.vad.speech_chunks = MagicMock(return_value=fake_chunks)
    return b


@pytest.mark.fast
def test_oom_retries_with_halved_batch():
    """First call raises OOM, second succeeds — backend must retry not raise."""
    backend = _backend(batch_size=32)
    fake_info = MagicMock(language="zh", language_probability=1.0)

    call_count = {"n": 0}
    def fake_transcribe(audio, batch_size, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
        return iter([]), fake_info
    backend.batched.transcribe.side_effect = fake_transcribe

    audio = np.zeros(16000 * 8, dtype=np.float32)
    with patch("taiwan_asr.breeze.AudioIO") as audio_io:
        audio_io.decode_to_array.return_value = (audio, 8.0)
        out, sw = backend.transcribe("any.wav")

    assert call_count["n"] == 2, (
        f"Backend should retry once after OOM; called {call_count['n']} times. "
        f"Pre-v0.5.5 it raised on the first OOM with no retry."
    )
    # Inspect the second call's batch_size — must be smaller than the first
    first_bs = backend.batched.transcribe.call_args_list[0].kwargs["batch_size"]
    second_bs = backend.batched.transcribe.call_args_list[1].kwargs["batch_size"]
    assert second_bs < first_bs, (
        f"Retry must use a smaller batch_size; first={first_bs}, second={second_bs}"
    )


@pytest.mark.fast
def test_oom_retry_does_not_mutate_self_cfg_batch_size():
    """H3-equivalent for Breeze: cur_bs is local, must not bleed into self.cfg."""
    backend = _backend(batch_size=32)
    original = backend.cfg.batch_size
    fake_info = MagicMock(language="zh", language_probability=1.0)

    call_count = {"n": 0}
    def fake_transcribe(audio, batch_size, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise torch.cuda.OutOfMemoryError("CUDA OOM")
        return iter([]), fake_info
    backend.batched.transcribe.side_effect = fake_transcribe

    audio = np.zeros(16000 * 8, dtype=np.float32)
    with patch("taiwan_asr.breeze.AudioIO") as audio_io:
        audio_io.decode_to_array.return_value = (audio, 8.0)
        backend.transcribe("any.wav")

    assert backend.cfg.batch_size == original, (
        f"cur_bs must be local; self.cfg.batch_size leaked from {original} "
        f"to {backend.cfg.batch_size}. Multi-file runs would inherit a shrunken "
        f"batch even when later files would have fit at the original."
    )


@pytest.mark.fast
def test_oom_at_batch_1_eventually_raises():
    """If even batch_size=1 OOMs, propagate — don't infinite-loop."""
    backend = _backend(batch_size=4)
    backend.batched.transcribe.side_effect = torch.cuda.OutOfMemoryError("CUDA OOM")

    audio = np.zeros(16000 * 8, dtype=np.float32)
    with patch("taiwan_asr.breeze.AudioIO") as audio_io:
        audio_io.decode_to_array.return_value = (audio, 8.0)
        with pytest.raises(torch.cuda.OutOfMemoryError):
            backend.transcribe("any.wav")
