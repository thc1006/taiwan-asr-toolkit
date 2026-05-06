# -*- coding: utf-8 -*-
"""TDD: CER (Character Error Rate) 評估工具
紅燈期望:
  1. cer_eval 模組存在,提供 compute_cer / compute_metrics
  2. compute_cer('我們吃飯', '我們吃完飯') 約等於 0.25 (1 insertion / 4 chars)
  3. compute_metrics 回傳 dict 包含 cer, hits, substitutions, deletions, insertions
  4. 對中文做正規化 (去標點、空白、簡轉繁) 再比
"""
from __future__ import annotations
import pytest


@pytest.mark.fast
def test_cer_eval_module_exists():
    from taiwan_asr import cer_eval
    assert hasattr(cer_eval, "compute_cer")


@pytest.mark.fast
def test_compute_cer_simple_insertion():
    from taiwan_asr.cer_eval import compute_cer
    # ref: '我們吃飯' (4 chars), hyp: '我們吃完飯' (5 chars) — 1 insertion
    cer = compute_cer("我們吃飯", "我們吃完飯")
    assert abs(cer - 0.25) < 0.05, f"CER 應約 0.25, 實際 {cer}"


@pytest.mark.fast
def test_compute_cer_perfect_match():
    from taiwan_asr.cer_eval import compute_cer
    assert compute_cer("這是測試", "這是測試") == 0.0


@pytest.mark.fast
def test_compute_cer_normalizes_punctuation():
    from taiwan_asr.cer_eval import compute_cer
    cer = compute_cer("這是測試。", "這是測試,")
    assert cer == 0.0, f"標點差異不應計入 CER, 實際 {cer}"


@pytest.mark.fast
def test_compute_cer_normalizes_simplified_to_traditional():
    from taiwan_asr.cer_eval import compute_cer
    # 簡體 ref vs 繁體 hyp 應視為相同
    cer = compute_cer("我们吃饭", "我們吃飯")
    assert cer == 0.0, f"簡繁差異應被正規化, 實際 {cer}"


@pytest.mark.fast
def test_compute_metrics_returns_breakdown():
    from taiwan_asr.cer_eval import compute_metrics
    m = compute_metrics("我們吃飯", "我們吃完飯")
    assert "cer" in m and "insertions" in m and "deletions" in m
    assert "substitutions" in m and "hits" in m
    assert m["insertions"] >= 1
    assert m["substitutions"] == 0
    assert m["deletions"] == 0


@pytest.mark.fast
def test_cer_handles_empty_ref():
    from taiwan_asr.cer_eval import compute_cer
    cer = compute_cer("", "abc")
    assert isinstance(cer, float)
