# -*- coding: utf-8 -*-
"""Regression test for v0.5.5 M2: Breeze must length-sort VAD chunks before
handing them to BatchedInferencePipeline, mirroring qwen3's pool batching.

Pre-v0.5.5: breeze.py:256-259 fed clip_timestamps in original VAD time order.
With cudnn.benchmark=True (set by init_torch), each new mel-spectrogram
dimension triggered a cuDNN kernel rebuild — wall-clock spikes mid-stream
on long sparse audio with very uneven chunk lengths.

The fix sorts chunks by duration ascending before passing to faster-whisper.
Because each clip is decoded independently (condition_on_previous_text=False),
reordering does not affect transcription quality — but the output segment
list MUST still be time-sorted by start before returning to the caller.
"""
from __future__ import annotations
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("faster_whisper")

from taiwan_asr.breeze import FasterWhisperBackend, HwConfig
from taiwan_asr.common import S2TW


def _backend() -> FasterWhisperBackend:
    cfg = HwConfig(
        device="cpu", cuda_idx=0, dtype=torch.float32, ct2_compute="float32",
        batch_size=1, cpu_threads=1, num_workers=1,
        chunk_sec=28.0, pad_sec=0.4, desc="test",
    )
    b = FasterWhisperBackend(cfg=cfg, beam=1, s2tw=S2TW(False))
    b.model = MagicMock()
    b.batched = MagicMock()
    b.vad = MagicMock(_using="test")
    return b


@pytest.mark.fast
def test_clip_timestamps_passed_to_batched_are_length_sorted():
    """clip_timestamps argument should be sorted by chunk duration ascending."""
    backend = _backend()
    # Chunks in ARBITRARY time order, with uneven durations
    chunks = [
        {"start": 0.0,  "end": 25.0},  # 25 s
        {"start": 30.0, "end": 31.0},  # 1 s
        {"start": 40.0, "end": 50.0},  # 10 s
        {"start": 60.0, "end": 65.0},  # 5 s
    ]
    backend.vad.speech_chunks = MagicMock(return_value=chunks)
    fake_info = MagicMock(language="zh", language_probability=1.0)
    backend.batched.transcribe.return_value = (iter([]), fake_info)

    audio = np.zeros(16000 * 70, dtype=np.float32)
    with patch("taiwan_asr.breeze.AudioIO") as audio_io:
        audio_io.decode_to_array.return_value = (audio, 70.0)
        backend.transcribe("any.wav")

    kwargs = backend.batched.transcribe.call_args.kwargs
    cts = kwargs.get("clip_timestamps")
    assert isinstance(cts, list) and all(isinstance(c, dict) for c in cts)
    durations = [c["end"] - c["start"] for c in cts]
    assert durations == sorted(durations), (
        f"clip_timestamps must be length-sorted ascending; got durations={durations}. "
        f"Pre-v0.5.5 they came in original VAD time order, causing cuDNN cache thrash."
    )


@pytest.mark.fast
def test_output_segments_are_time_sorted_after_length_sort():
    """Even though we feed faster-whisper out-of-time-order chunks, the
    Segment list returned to the caller MUST be sorted by .start."""
    backend = _backend()
    chunks = [
        {"start": 0.0,  "end": 25.0},
        {"start": 30.0, "end": 31.0},
        {"start": 40.0, "end": 50.0},
    ]
    backend.vad.speech_chunks = MagicMock(return_value=chunks)
    # faster-whisper would return segments in some order — simulate length-sort
    # output (shortest first) to verify our re-sort step
    fake_info = MagicMock(language="zh", language_probability=1.0)
    fake_segments = [
        # Shortest chunk's segment (came back first because length-sorted)
        MagicMock(start=30.0, end=31.0, text="b", words=None, avg_logprob=0.0, no_speech_prob=0.0),
        # Mid chunk
        MagicMock(start=40.0, end=50.0, text="c", words=None, avg_logprob=0.0, no_speech_prob=0.0),
        # Longest chunk
        MagicMock(start=0.0,  end=25.0, text="a", words=None, avg_logprob=0.0, no_speech_prob=0.0),
    ]
    backend.batched.transcribe.return_value = (iter(fake_segments), fake_info)

    audio = np.zeros(16000 * 60, dtype=np.float32)
    with patch("taiwan_asr.breeze.AudioIO") as audio_io:
        audio_io.decode_to_array.return_value = (audio, 60.0)
        out, sw = backend.transcribe("any.wav")

    starts = [s.start for s in out]
    assert starts == sorted(starts), (
        f"Output segments must be time-sorted by .start, got {starts}. "
        f"Length-sorted input ordering must NOT leak into the output."
    )
