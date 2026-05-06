# -*- coding: utf-8 -*-
"""TDD: glossary 載入器
紅燈期望:
  1. _asr_common 模組必須有 load_glossary(path) 函式
  2. 讀現有 glossary.txt 應返回非空 list
  3. 跳過 # 註解與空行
  4. 「延三舍」必須在預設 glossary 中
"""
from __future__ import annotations
import textwrap
from pathlib import Path
import pytest


@pytest.mark.fast
def test_load_glossary_function_exists():
    from taiwan_asr.common import load_glossary  # 紅燈
    assert callable(load_glossary)


@pytest.mark.fast
def test_load_glossary_skips_comments_and_blanks(tmp_path):
    from taiwan_asr.common import load_glossary
    p = tmp_path / "g.txt"
    p.write_text(textwrap.dedent("""
        # 這行是註解
        延三舍

        # 又一個註解
        圓山宿舍
        男一
    """).strip(), encoding="utf-8")
    terms = load_glossary(str(p))
    assert terms == ["延三舍", "圓山宿舍", "男一"]


@pytest.mark.fast
def test_load_glossary_default_file_has_ntu_terms(project_root):
    from taiwan_asr.common import load_glossary
    g_path = project_root / "glossary.txt"
    if not g_path.is_file():
        pytest.skip("glossary.txt 不存在")
    terms = load_glossary(str(g_path))
    assert len(terms) >= 20, f"glossary 詞太少: {len(terms)}"
    assert "延三舍" in terms, "預設 glossary 必含 延三舍"
    assert "圓山宿舍" in terms or "圓山" in terms
    assert "祝福二組" in terms


@pytest.mark.fast
def test_load_glossary_handles_missing_file():
    from taiwan_asr.common import load_glossary
    out = load_glossary("/path/does/not/exist.txt")
    # 不應 crash,應返回空 list
    assert out == []


@pytest.mark.fast
def test_load_glossary_dedupe_preserves_order():
    from taiwan_asr.common import load_glossary
    import tempfile, os
    fd, p = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    Path(p).write_text("延三舍\n圓山\n延三舍\n男一\n", encoding="utf-8")
    try:
        terms = load_glossary(p)
        assert terms == ["延三舍", "圓山", "男一"], "重複詞應保留首次出現順序"
    finally:
        os.remove(p)
