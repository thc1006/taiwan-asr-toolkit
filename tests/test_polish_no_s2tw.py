# -*- coding: utf-8 -*-
"""Regression test for v0.5.5 M8: asr-polish must accept --no-s2tw, matching
the breeze and qwen3 CLIs which already had it.

Pre-v0.5.5: polish.py had no --no-s2tw flag; Qwen3Polisher hard-coded
S2TW(True) at line 116. README:188 listed --no-s2tw without scoping it,
so users running `asr-polish ... --no-s2tw` got an argparse error.
"""
from __future__ import annotations
import inspect
from argparse import ArgumentParser, Namespace
from unittest.mock import patch

import pytest


@pytest.mark.fast
def test_polish_argparser_has_no_s2tw_flag():
    """Build the polish CLI argparser, confirm --no-s2tw is registered."""
    from taiwan_asr.polish import build_argparser
    ap = build_argparser()
    args = ap.parse_args(["dummy.json", "--no-s2tw"])
    assert getattr(args, "no_s2tw", False) is True


@pytest.mark.fast
def test_polish_argparser_default_keeps_s2tw_enabled():
    """Without --no-s2tw the default should be False (i.e. s2tw stays enabled)."""
    from taiwan_asr.polish import build_argparser
    ap = build_argparser()
    args = ap.parse_args(["dummy.json"])
    assert getattr(args, "no_s2tw", False) is False


@pytest.mark.fast
def test_qwen3polisher_constructor_accepts_s2tw_enabled_param():
    """The polisher class itself must accept an s2tw_enabled flag so the CLI
    can plumb the flag value in. Pre-v0.5.5 it was hard-coded True."""
    from taiwan_asr.polish import Qwen3Polisher
    sig = inspect.signature(Qwen3Polisher.__init__)
    assert "s2tw_enabled" in sig.parameters, (
        "Qwen3Polisher.__init__ must accept s2tw_enabled to honor --no-s2tw."
    )
