# -*- coding: utf-8 -*-
"""Regression test for v0.5.5 M3: polish.py must pick pad_token_id correctly.

Pre-v0.5.5 behaviour (polish.py:184):
    pad_token_id=self.tok.eos_token_id,

Qwen2.5 / Qwen3 tokenizers have a SEPARATE pad token (e.g. <|fim_pad|>) and
eos token (<|endoftext|>). Hard-coding pad := eos:
  - emits HF UserWarning ("Setting pad_token_id to eos_token_id...")
  - is correct enough for batch=1 single-sequence calls (current usage)
  - silently corrupts attention masks if anyone extends polish to batched
    generation (window>1 across multiple sequences).

The fix uses tok.pad_token_id when available, falling back to eos only when
the tokenizer truly lacks a pad token.
"""
from __future__ import annotations
from types import SimpleNamespace

import pytest

from taiwan_asr.polish import _choose_pad_token_id


@pytest.mark.fast
def test_uses_pad_when_tokenizer_has_distinct_pad():
    """Qwen-style tokenizer: distinct pad and eos -> use pad."""
    tok = SimpleNamespace(pad_token_id=42, eos_token_id=7)
    assert _choose_pad_token_id(tok) == 42


@pytest.mark.fast
def test_falls_back_to_eos_when_pad_is_none():
    """Older tokenizers without an explicit pad token -> fall back to eos."""
    tok = SimpleNamespace(pad_token_id=None, eos_token_id=7)
    assert _choose_pad_token_id(tok) == 7


@pytest.mark.fast
def test_falls_back_to_eos_when_pad_attr_missing():
    """Tokenizer without pad_token_id attribute at all -> still works."""
    tok = SimpleNamespace(eos_token_id=99)
    assert _choose_pad_token_id(tok) == 99


@pytest.mark.fast
def test_handles_zero_pad_id():
    """pad_token_id == 0 is a valid token id (e.g. some BPE tokenizers).
    Don't conflate 0 with falsy/None — must NOT fall back to eos here."""
    tok = SimpleNamespace(pad_token_id=0, eos_token_id=7)
    assert _choose_pad_token_id(tok) == 0
