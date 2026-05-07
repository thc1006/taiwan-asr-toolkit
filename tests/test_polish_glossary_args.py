# -*- coding: utf-8 -*-
"""Regression test for v0.5.5 S3+M4: asr-polish --glossary-file builtin must
actually load the packaged NTU glossary, not silently return [].

Pre-v0.5.5 behaviour (polish.py:260-266 reimplemented its own loader):

    if args.glossary_file:
        gp = Path(args.glossary_file)
        if gp.is_file():        # Path("builtin").is_file() -> False
            extra_glossary += [...]   # so this branch never runs

Result: --glossary-file builtin silently degraded to no extra glossary, with
no warning, no error. README:194 documented this exact invocation.
"""
from __future__ import annotations
from argparse import Namespace

import pytest

from taiwan_asr.polish import _resolve_extra_glossary


@pytest.mark.fast
def test_builtin_keyword_loads_packaged_glossary():
    """The 'builtin' magic value must resolve to the packaged NTU glossary."""
    args = Namespace(glossary="", glossary_file="builtin")
    terms = _resolve_extra_glossary(args)
    assert len(terms) >= 30, (
        f"--glossary-file builtin should pull in the packaged NTU glossary "
        f"(>=30 terms expected), got {len(terms)}."
    )
    # Spot-check known dorm names that ship in src/taiwan_asr/data/ntu_glossary.txt
    for canary in ("研三舍", "祝福二組", "住輔組"):
        assert canary in terms, (
            f"Packaged glossary must contain canary term {canary!r}; "
            f"got first 5: {terms[:5]}"
        )


@pytest.mark.fast
def test_builtin_case_insensitive():
    """Case variants of the magic value should work too (matches common.load_glossary)."""
    for variant in ("BUILTIN", "Builtin", "BuiltIn"):
        args = Namespace(glossary="", glossary_file=variant)
        terms = _resolve_extra_glossary(args)
        assert len(terms) > 0, f"variant {variant!r} should also resolve to packaged terms"


@pytest.mark.fast
def test_explicit_file_path_still_works(tmp_path):
    """Regression: a real file path must still load normally (no behavior break)."""
    g = tmp_path / "my_terms.txt"
    g.write_text("# my comment\n甲乙\n丙丁\n  # indented comment\n戊己\n", encoding="utf-8")
    args = Namespace(glossary="", glossary_file=str(g))
    terms = _resolve_extra_glossary(args)
    assert "甲乙" in terms and "丙丁" in terms and "戊己" in terms
    # The lstripped # comment must NOT leak through (M4 fix)
    assert "# indented comment" not in terms
    assert "  # indented comment" not in terms


@pytest.mark.fast
def test_inline_glossary_combines_with_file(tmp_path):
    """--glossary 'a,b,c' + --glossary-file builtin should concatenate."""
    args = Namespace(glossary="自訂A,自訂B", glossary_file="builtin")
    terms = _resolve_extra_glossary(args)
    assert "自訂A" in terms and "自訂B" in terms
    assert "研三舍" in terms  # from builtin


@pytest.mark.fast
def test_missing_file_returns_empty_quietly():
    """Non-existent file path -> [] silently (current contract preserved)."""
    args = Namespace(glossary="", glossary_file="/no/such/file.txt")
    terms = _resolve_extra_glossary(args)
    assert terms == []


@pytest.mark.fast
def test_no_glossary_args_returns_empty():
    """Neither --glossary nor --glossary-file -> []."""
    args = Namespace(glossary="", glossary_file="")
    terms = _resolve_extra_glossary(args)
    assert terms == []
