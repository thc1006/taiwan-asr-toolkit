# -*- coding: utf-8 -*-
"""Lock the v0.5.1 Blackwell VRAM-tier batch table as a regression test.

Tier table (sm_120 / sm_100):
    >= 140 GB -> Qwen3 batch=128, Breeze batch=96   (B100 / B200 datacenter)
    >= 70 GB  -> Qwen3 batch=96,  Breeze batch=64   (RTX Pro 6000 workstation)
    >= 30 GB  -> Qwen3 batch=48,  Breeze batch=32   (RTX 5090 — unchanged from 0.5.0)
    smaller   -> Qwen3 batch=32,  Breeze batch=16
"""
from __future__ import annotations

import pytest


class _FakeProps:
    """Minimal stand-in for torch.cuda.get_device_properties() return value."""
    def __init__(self, total_bytes: int) -> None:
        self.total_memory = total_bytes


def _patch_blackwell(monkeypatch, vram_gb: float) -> None:
    """Pretend we are on a Blackwell GPU (cc=(12,0)) with the given VRAM."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: f"Mock Blackwell {vram_gb:.0f}GB", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (12, 0), raising=False)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda i=0: _FakeProps(int(vram_gb * 1024 ** 3)),
        raising=False,
    )
    # bf16 always supported on Blackwell
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True, raising=False)


@pytest.mark.fast
@pytest.mark.parametrize(
    "vram_gb,expected_qwen3,expected_breeze",
    [
        (192.0, 128, 96),    # B100 / B200 datacenter
        (140.0, 128, 96),    # exact lower bound of the >=140 GB tier
        (96.0,  96,  64),    # RTX Pro 6000 workstation (Colab Pro+ option)
        (70.0,  96,  64),    # exact lower bound of the >=70 GB tier
        (32.0,  48,  32),    # RTX 5090 — must remain unchanged from v0.5.0
        (30.0,  48,  32),    # exact lower bound of the >=30 GB tier
        (16.0,  32,  16),    # hypothetical smaller Blackwell variant
    ],
)
def test_blackwell_vram_tier_batch_table(monkeypatch, vram_gb, expected_qwen3, expected_breeze):
    _patch_blackwell(monkeypatch, vram_gb)

    from taiwan_asr.breeze import detect_hw as detect_breeze
    from taiwan_asr.qwen3 import detect_hw as detect_qwen3

    cfg_br = detect_breeze(fast=False)
    cfg_q3 = detect_qwen3()

    assert cfg_br.batch_size == expected_breeze, (
        f"Breeze batch on {vram_gb}GB Blackwell: expected {expected_breeze}, got {cfg_br.batch_size}"
    )
    assert cfg_q3.batch_size == expected_qwen3, (
        f"Qwen3 batch on {vram_gb}GB Blackwell: expected {expected_qwen3}, got {cfg_q3.batch_size}"
    )


@pytest.mark.fast
def test_rtx_5090_path_unchanged(monkeypatch):
    """Defensive: explicit guard that v0.5.0 RTX 5090 user behavior was preserved."""
    _patch_blackwell(monkeypatch, 32.0)
    from taiwan_asr.breeze import detect_hw as detect_breeze
    from taiwan_asr.qwen3 import detect_hw as detect_qwen3
    assert detect_breeze(fast=False).batch_size == 32, "RTX 5090 Breeze batch must stay 32 (v0.5.0 baseline)"
    assert detect_qwen3().batch_size == 48, "RTX 5090 Qwen3 batch must stay 48 (v0.5.0 baseline)"


@pytest.mark.fast
def test_blackwell_uses_bfloat16_dtype_at_all_vram_tiers(monkeypatch):
    """All Blackwell tiers should use bf16 (their tensor cores are native)."""
    import torch
    for vram in (16.0, 32.0, 96.0, 192.0):
        _patch_blackwell(monkeypatch, vram)
        from taiwan_asr.breeze import detect_hw as detect_breeze
        from taiwan_asr.qwen3 import detect_hw as detect_qwen3
        assert detect_breeze(fast=False).dtype is torch.bfloat16, f"Breeze on {vram}GB Blackwell should be bf16"
        assert detect_qwen3().dtype is torch.bfloat16, f"Qwen3 on {vram}GB Blackwell should be bf16"
