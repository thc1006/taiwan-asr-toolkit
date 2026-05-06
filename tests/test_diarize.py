# -*- coding: utf-8 -*-
"""TDD: 講者分離 (pyannote.audio)
紅燈期望:
  1. diarize 模組存在
  2. assign_speakers(segments, diar_segments) 正確 — 用 segment 中點時戳對齊
  3. 缺 HF token 或缺 model 時優雅 fail (回傳原 segments,no crash)
  4. Segment dataclass 有 speaker_id 欄位
"""
from __future__ import annotations
import pytest


@pytest.mark.fast
def test_diarize_module_exists():
    import diarize
    assert hasattr(diarize, "assign_speakers"), "diarize.assign_speakers 不存在"


@pytest.mark.fast
def test_segment_has_speaker_id_field():
    from _asr_common import Segment
    s = Segment(start=0.0, end=1.0, text="hi")
    assert hasattr(s, "speaker_id"), "Segment 缺 speaker_id 欄位"
    assert s.speaker_id is None, "預設應為 None (向後相容)"


@pytest.mark.fast
def test_segment_to_line_with_speaker():
    from _asr_common import Segment
    s = Segment(start=0.0, end=1.0, text="哈囉", speaker_id="SPEAKER_01")
    line = s.to_line()
    assert "SPEAKER_01" in line, f"to_line 應含 speaker_id,實際:{line!r}"


@pytest.mark.fast
def test_segment_to_line_without_speaker_unchanged():
    """無 speaker_id 時格式不變 (向後相容)。"""
    from _asr_common import Segment
    s = Segment(start=0.0, end=1.0, text="哈囉")
    line = s.to_line()
    assert "SPEAKER" not in line, f"無 speaker_id 時不應有 SPEAKER 字串:{line!r}"


@pytest.mark.fast
def test_assign_speakers_by_midpoint():
    """每個 ASR 段用中點時戳對齊 diar 段。"""
    from _asr_common import Segment
    from diarize import assign_speakers

    asr_segs = [
        Segment(0.0, 2.0, "段一"),
        Segment(2.5, 5.0, "段二"),
        Segment(5.5, 8.0, "段三"),
    ]
    # diar 結果 (start, end, speaker)
    diar = [
        (0.0, 3.0, "SPEAKER_00"),
        (3.0, 6.0, "SPEAKER_01"),
        (6.0, 9.0, "SPEAKER_00"),
    ]
    out = assign_speakers(asr_segs, diar)
    assert out[0].speaker_id == "SPEAKER_00"  # mid=1.0 → 0-3
    assert out[1].speaker_id == "SPEAKER_01"  # mid=3.75 → 3-6
    assert out[2].speaker_id == "SPEAKER_00"  # mid=6.75 → 6-9


@pytest.mark.fast
def test_assign_speakers_empty_diar_returns_unchanged():
    """空 diar 結果應回傳原 segments (no crash)。"""
    from _asr_common import Segment
    from diarize import assign_speakers
    asr_segs = [Segment(0.0, 1.0, "test")]
    out = assign_speakers(asr_segs, [])
    assert len(out) == 1
    assert out[0].text == "test"
    # speaker_id 應仍是 None
    assert out[0].speaker_id is None
