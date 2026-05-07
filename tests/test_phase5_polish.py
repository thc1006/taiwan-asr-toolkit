# -*- coding: utf-8 -*-
"""Phase 5 batch: L2 (path normalization), L4 (Stopwatch dup labels),
L5 (PUNCT_RE Unicode ellipsis variants), L6 (diarize error message).

L3 (cer_eval wer == cer redundant key) is documented in-place via comment;
no behavioural change so no test added.
"""
from __future__ import annotations
import re

import pytest

from taiwan_asr.common import Stopwatch


# ── L4: Stopwatch.lap should warn on duplicate label ─────────────────────

@pytest.mark.fast
def test_stopwatch_lap_warns_on_duplicate_label(capsys):
    """Pre-v0.5.5 lap() silently allowed duplicate labels; .get(prefix) only
    returned the first match. Now the second lap with the same label emits
    a warning so callers notice the collision."""
    sw = Stopwatch()
    sw.lap("ASR")
    sw.lap("ASR")  # duplicate
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "ASR" in combined and ("duplicate" in combined.lower() or "重複" in combined), (
        f"Stopwatch must warn when the same label is laps twice; got:\n{combined!r}"
    )


# ── L5: PUNCT_RE must include Unicode ellipsis variants ─────────────────

@pytest.mark.fast
def test_punct_re_strips_unicode_ellipsis_variants():
    """U+22EF (⋯) and U+2027 (‧) are common in Apple Voice Memos / Chinese
    typography. CER between transcripts that differ only in ellipsis style
    should be 0; pre-v0.5.5 they were not stripped."""
    from taiwan_asr.cer_eval import _PUNCT_RE
    samples = [
        "你好⋯⋯世界",
        "你好‧世界",
        "你好…世界",
    ]
    cleaned = [_PUNCT_RE.sub("", s) for s in samples]
    assert all(c == "你好世界" for c in cleaned), (
        f"PUNCT_RE must strip ⋯ (U+22EF), ‧ (U+2027), and … uniformly; got {cleaned}"
    )


# ── L2: qwen3 path normalization keys ────────────────────────────────────

@pytest.mark.fast
def test_qwen3_pool_normalizes_path_keys():
    """transcribe_files must key by the resolved path so duplicates like
    './a.mp3' and 'a.mp3' don't appear as two slots in the result dict."""
    from taiwan_asr.qwen3 import _normalize_pool_key
    a = _normalize_pool_key("./music/a.mp3")
    b = _normalize_pool_key("music/a.mp3")
    assert a == b, (
        f"Path normalizer must collapse equivalent forms; got {a!r} vs {b!r}"
    )


# ── L6: diarize error msg points to actual model_id ──────────────────────

@pytest.mark.fast
def test_diarize_error_message_uses_actual_model_id():
    """Pre-v0.5.5 the gated-error message hardcoded pyannote/speaker-diarization-3.1
    even when the actual default model_id was tensorlake/...  Resulting users
    would 'Agree' on the wrong repo. Now the message uses the live model_id."""
    from taiwan_asr.diarize import _format_gated_error_message
    msg = _format_gated_error_message("tensorlake/speaker-diarization-3.1")
    assert "tensorlake/speaker-diarization-3.1" in msg, (
        f"Error message must reference the model_id the caller actually passed; got:\n{msg}"
    )
    msg2 = _format_gated_error_message("pyannote/speaker-diarization-3.1")
    assert "pyannote/speaker-diarization-3.1" in msg2
