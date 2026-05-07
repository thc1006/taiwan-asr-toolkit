# -*- coding: utf-8 -*-
"""v0.5.3: glossary is now packaged into the wheel (src/taiwan_asr/data/).

These tests lock that:
  1. `taiwan_asr.common.builtin_glossary_path()` resolves to a real file
  2. `load_glossary("builtin")` reads it and produces the expected NTU terms
  3. The root `glossary.txt` (kept for clone-workflow back-compat) stays
     byte-for-byte synced with the packaged copy
"""
from __future__ import annotations
from pathlib import Path
import pytest


@pytest.mark.fast
def test_builtin_glossary_path_resolves():
    from taiwan_asr.common import builtin_glossary_path
    p = builtin_glossary_path()
    assert p.is_file(), f"packaged glossary not found at {p}"
    assert p.suffix == ".txt"
    # Must live inside the taiwan_asr package
    assert "taiwan_asr" in str(p) and "data" in str(p)


@pytest.mark.fast
def test_load_glossary_builtin_keyword_loads_packaged_terms():
    from taiwan_asr.common import load_glossary
    terms = load_glossary("builtin")
    assert len(terms) >= 20, f"packaged glossary too small: {len(terms)} terms"
    assert "研三舍" in terms
    assert "祝福二組" in terms
    assert "圓山宿舍" in terms or "圓山" in terms


@pytest.mark.fast
def test_load_glossary_builtin_is_case_insensitive():
    from taiwan_asr.common import load_glossary
    a = load_glossary("BUILTIN")
    b = load_glossary("Builtin")
    c = load_glossary("builtin")
    assert a == b == c
    assert len(a) >= 20


@pytest.mark.fast
def test_root_glossary_synced_with_packaged(project_root):
    """Root glossary.txt is kept for clone-workflow users; it MUST stay
    byte-identical to the packaged copy. If you edit one, sync the other.
    """
    root = project_root / "glossary.txt"
    packaged = project_root / "src" / "taiwan_asr" / "data" / "ntu_glossary.txt"
    if not (root.is_file() and packaged.is_file()):
        pytest.skip("one of root/packaged glossary missing; sync check N/A")
    assert root.read_text(encoding="utf-8") == packaged.read_text(encoding="utf-8"), (
        "glossary.txt at repo root has drifted from src/taiwan_asr/data/ntu_glossary.txt.\n"
        "Sync them with: cp src/taiwan_asr/data/ntu_glossary.txt glossary.txt"
    )
