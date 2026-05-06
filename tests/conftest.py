# -*- coding: utf-8 -*-
"""共用 pytest fixtures。"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
import pytest

# 把專案根目錄加到 sys.path,讓 tests 可以 import 主腳本模組
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def clip_30s(fixtures_dir) -> Path:
    """30 秒台灣繁中音訊 (從 標準錄音 886 截前 30s)"""
    p = fixtures_dir / "clip_30s.wav"
    if not p.is_file():
        pytest.skip(f"fixture 缺失: {p}")
    return p


@pytest.fixture(scope="session")
def music_886(project_root) -> Path:
    """完整 4-min 台灣繁中錄音"""
    p = project_root / "music" / "標準錄音 886.mp3"
    if not p.is_file():
        pytest.skip(f"音檔缺失: {p}")
    return p


@pytest.fixture(scope="session")
def music_884(project_root) -> Path:
    p = project_root / "music" / "標準錄音 884.mp3"
    if not p.is_file():
        pytest.skip(f"音檔缺失: {p}")
    return p


@pytest.fixture(scope="session")
def existing_breeze_886(project_root) -> dict:
    """v3 已產生的 Breeze 結果作 golden,供回歸比較"""
    p = project_root / "transcripts" / "breeze" / "標準錄音 886_breeze.json"
    if not p.is_file():
        pytest.skip(f"v3 baseline 缺: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def existing_qwen3_886(project_root) -> dict:
    p = project_root / "transcripts" / "qwen3" / "標準錄音 886_qwen3.json"
    if not p.is_file():
        pytest.skip(f"v3 baseline 缺: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
