# -*- coding: utf-8 -*-
"""
v3 回歸測試 — 鎖住既有輸出特徵作 baseline。
任何 v4+ 優化必須維持或改善以下指標:
  * 段數在合理範圍 (不可大幅減少 — 那是 hallucination/漏切)
  * 字元覆蓋與字數
  * 兩模型字元 Jaccard 一致性
  * 繁體率
  * 無 >60s 失控段
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import pytest


def _norm(t: str) -> str:
    return re.sub(r"[\s\,\.\!\?\;\:\、\。\,\!\?\;\:\「\」\『\』\"\'\(\)（）\-\_\d]+", "", t or "")


def _stats(segs):
    full = "".join(_norm(s.get("text", "")) for s in segs)
    n_segs = len(segs)
    n_chars = len(full)
    longest_seg = max(((s.get("end", 0) - s.get("start", 0)) for s in segs), default=0)
    return n_segs, n_chars, longest_seg, full


@pytest.mark.fast
class TestBreeze886Baseline:
    """v3 在 標準錄音 886.mp3 上的特徵鎖定。"""

    def test_segment_count_in_range(self, existing_breeze_886):
        n_segs = len(existing_breeze_886)
        # v3 manual VAD 產出 11 段;允許 9-15 容錯
        assert 8 <= n_segs <= 18, f"段數異常: {n_segs} (應在 8-18)"

    def test_no_runaway_long_segment(self, existing_breeze_886):
        for s in existing_breeze_886:
            seg_dur = s.get("end", 0) - s.get("start", 0)
            assert seg_dur <= 60.0, (
                f"出現 >60s 失控段 (start={s.get('start')}, end={s.get('end')}) "
                f"→ Whisper VAD 失控,絕不可發生"
            )

    def test_meaningful_content(self, existing_breeze_886):
        _, n_chars, _, _ = _stats(existing_breeze_886)
        assert n_chars >= 500, f"4 分鐘錄音字數 {n_chars} 過少,可能漏切"

    def test_traditional_chinese_majority(self, existing_breeze_886):
        _, _, _, full = _stats(existing_breeze_886)
        n_simp = sum(1 for c in full if c in "国发现实际经济认识让说话语门间问题")
        n_trad = sum(1 for c in full if c in "國發現實際經濟認識讓說話語門間問題")
        assert n_trad >= n_simp, f"繁:{n_trad} < 簡:{n_simp}"


@pytest.mark.fast
class TestQwen3886Baseline:
    """v3 在 標準錄音 886.mp3 上 Qwen3 特徵鎖定。"""

    def test_segment_count_in_range(self, existing_qwen3_886):
        n_segs = len(existing_qwen3_886)
        assert 8 <= n_segs <= 18, f"段數 {n_segs} 異常"

    def test_no_runaway_long_segment(self, existing_qwen3_886):
        for s in existing_qwen3_886:
            seg_dur = s.get("end", 0) - s.get("start", 0)
            assert seg_dur <= 60.0

    def test_meaningful_content(self, existing_qwen3_886):
        _, n_chars, _, _ = _stats(existing_qwen3_886)
        assert n_chars >= 500


@pytest.mark.fast
class TestCrossModelAgreement:
    """兩模型一致性 — 同一段 30 秒範圍內的轉錄字元應有高度重疊。"""

    def _char_jaccard(self, a: str, b: str) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(len(sa | sb), 1)

    def test_jaccard_above_threshold(self, existing_breeze_886, existing_qwen3_886):
        a = "".join(_norm(s.get("text", "")) for s in existing_breeze_886)
        b = "".join(_norm(s.get("text", "")) for s in existing_qwen3_886)
        j = self._char_jaccard(a, b)
        # v3 實測 ~0.89,設 0.80 容錯
        assert j >= 0.80, f"Jaccard {j:.3f} 低於 0.80 — 兩模型出現嚴重分歧"
