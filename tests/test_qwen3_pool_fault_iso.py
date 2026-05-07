# -*- coding: utf-8 -*-
"""Regression test for v0.5.5 M5: qwen3 transcribe_files must isolate per-file
decode failures.

Pre-v0.5.5 (qwen3.py:333-339):

    for fi, src in enumerate(paths):
        audio, dur = AudioIO.decode_to_array(src, sr=sr)   # <- raises here
        chunks = self.vad.speech_chunks(audio)
        ...

If ffmpeg fails on one corrupt file in a 10-file pool, AudioIO raises
RuntimeError and the whole pool aborts — losing the work on the 9 good
files. Compare to the single-file CLI path (qwen3.py:454-457) which
prints "找不到 X" and continues.

Fix: wrap the per-file decode+VAD in try/except, log warning, leave the
file's slot in out_per_file as [], and proceed with the rest of the pool.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from taiwan_asr.qwen3 import HwConfig, Qwen3ASR


def _instance() -> Qwen3ASR:
    cfg = HwConfig(
        device="cpu", dtype=torch.float32, batch_size=4, chunk_sec=4.0,
        attn_impl="eager", compile_mode=None, desc="test",
    )
    asr = Qwen3ASR(cfg, s2tw_enabled=False, aligner_enabled=False)
    asr.model = MagicMock()
    return asr


@pytest.mark.fast
def test_pool_continues_when_one_file_decode_fails():
    """Bad file in middle of pool: good files still produce output."""
    asr = _instance()

    fake_audio = np.zeros(16000 * 4, dtype=np.float32)
    audio_chunk = (np.zeros(int(3 * 16000), dtype=np.float32), 16000)
    fake_chunks = [{"start": 0.0, "end": 3.0, "audio": audio_chunk}]
    asr.vad = MagicMock(_using="test")
    asr.vad.speech_chunks = MagicMock(return_value=fake_chunks)

    def fake_decode(src, sr=16000):
        if "bad" in src:
            raise RuntimeError(f"ffmpeg 解碼失敗: {src}")
        return fake_audio, 4.0
    with patch("taiwan_asr.qwen3.AudioIO") as audio_io:
        audio_io.decode_to_array.side_effect = fake_decode
        from types import SimpleNamespace
        asr._transcribe_batch = MagicMock(
            return_value=[SimpleNamespace(text="hi", language="zh")]
        )
        result, sw = asr.transcribe_files(["good_a.wav", "bad_b.wav", "good_c.wav"])

    assert "good_a.wav" in result and "bad_b.wav" in result and "good_c.wav" in result, (
        f"All 3 paths must appear in result dict; got {list(result.keys())}"
    )
    assert result["bad_b.wav"] == [], "Failed file slot must be empty"
    assert isinstance(result["good_a.wav"], list)
    assert isinstance(result["good_c.wav"], list)


@pytest.mark.fast
def test_pool_aborts_only_when_all_files_fail():
    """If every single file fails, return all-empty dict (no crash)."""
    asr = _instance()
    asr.vad = MagicMock(_using="test")
    asr.vad.speech_chunks = MagicMock(return_value=[])

    with patch("taiwan_asr.qwen3.AudioIO") as audio_io:
        audio_io.decode_to_array.side_effect = RuntimeError("ffmpeg fail (test)")
        result, sw = asr.transcribe_files(["a.wav", "b.wav"])

    assert result == {"a.wav": [], "b.wav": []}, (
        f"All-fail pool must yield all-empty mapping, got {result!r}"
    )
