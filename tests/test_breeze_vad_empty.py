# -*- coding: utf-8 -*-
"""Regression test for v0.5.5: Breeze must NOT crash when VAD returns 0 segments.

Pre-v0.5.5 behaviour (the bug confirmed on music/3.m4a + a synthetic anullsrc wav):

    if n_chunks == 0:
        common["clip_timestamps"] = "0"      # <-- string, not list-of-dict
    ...
    self.batched.transcribe(audio, ..., clip_timestamps="0")
    # faster-whisper internals iterate the str char-by-char and call .items()
    # on it -> AttributeError: 'str' object has no attribute 'items'

The qwen3 backend handles the same situation correctly:

    if n == 0:
        sw.lap("ASR")
        return [], sw

These tests lock the symmetric behaviour into Breeze.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("faster_whisper")

from taiwan_asr.breeze import FasterWhisperBackend, HwConfig
from taiwan_asr.common import S2TW, Stopwatch


def _cpu_cfg() -> HwConfig:
    return HwConfig(
        device="cpu",
        cuda_idx=0,
        dtype=torch.float32,
        ct2_compute="float32",
        batch_size=1,
        cpu_threads=1,
        num_workers=1,
        chunk_sec=4.0,
        pad_sec=0.4,
        desc="test cpu",
    )


def _backend_with_mocked_runtime() -> FasterWhisperBackend:
    """Build a FasterWhisperBackend whose model + batched + VAD are all mocked."""
    backend = FasterWhisperBackend(
        cfg=_cpu_cfg(), beam=1, s2tw=S2TW(False), use_manual_vad=True,
    )
    backend.model = MagicMock()
    backend.batched = MagicMock()
    backend.vad = MagicMock()
    backend.vad._using = "test"
    return backend


@pytest.mark.fast
def test_empty_vad_returns_empty_segments_not_crash():
    """When VAD finds zero speech segments, transcribe must return ([], Stopwatch),
    NOT crash inside faster-whisper with `'str' object has no attribute 'items'`.
    """
    backend = _backend_with_mocked_runtime()
    backend.vad.speech_chunks = MagicMock(return_value=[])  # silent audio

    silent = np.zeros(16000 * 30, dtype=np.float32)  # 30 s of zeros
    with patch("taiwan_asr.breeze.AudioIO") as audio_io:
        audio_io.decode_to_array.return_value = (silent, 30.0)
        out, sw = backend.transcribe("any_path.wav")

    assert out == [], (
        f"Empty-VAD must yield empty segment list (got {len(out)} segments). "
        "Pre-v0.5.5 this raised AttributeError instead."
    )
    assert isinstance(sw, Stopwatch)


@pytest.mark.fast
def test_empty_vad_does_not_call_inference_pipeline():
    """When VAD is empty, we must short-circuit BEFORE calling the batched
    pipeline. Calling .transcribe with empty/string clip_timestamps is the
    exact code path that crashed in v0.5.4.
    """
    backend = _backend_with_mocked_runtime()
    backend.vad.speech_chunks = MagicMock(return_value=[])

    silent = np.zeros(16000 * 30, dtype=np.float32)
    with patch("taiwan_asr.breeze.AudioIO") as audio_io:
        audio_io.decode_to_array.return_value = (silent, 30.0)
        backend.transcribe("any_path.wav")

    backend.batched.transcribe.assert_not_called()
    backend.model.transcribe.assert_not_called()


@pytest.mark.fast
def test_non_empty_vad_still_calls_pipeline():
    """Sanity: the empty-VAD short-circuit must NOT break the normal path."""
    backend = _backend_with_mocked_runtime()
    fake_chunks = [{"start": 0.0, "end": 5.0}, {"start": 6.0, "end": 12.0}]
    backend.vad.speech_chunks = MagicMock(return_value=fake_chunks)
    # batched.transcribe returns (iterator, info)
    fake_info = MagicMock(language="zh", language_probability=1.0)
    backend.batched.transcribe.return_value = (iter([]), fake_info)

    audio_arr = np.zeros(16000 * 12, dtype=np.float32)
    with patch("taiwan_asr.breeze.AudioIO") as audio_io:
        audio_io.decode_to_array.return_value = (audio_arr, 12.0)
        out, sw = backend.transcribe("any_path.wav")

    assert backend.batched.transcribe.called, (
        "Non-empty VAD must still call the inference pipeline."
    )
    # The clip_timestamps kwarg should be the list-of-dicts form for batched.
    kwargs = backend.batched.transcribe.call_args.kwargs
    assert isinstance(kwargs.get("clip_timestamps"), list), (
        f"Batched pipeline must receive list-of-dict clip_timestamps, "
        f"got {type(kwargs.get('clip_timestamps')).__name__}."
    )
    assert all(isinstance(c, dict) for c in kwargs["clip_timestamps"])
