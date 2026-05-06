# -*- coding: utf-8 -*-
"""
TDD: Qwen3 --no-aligner 旗標
紅燈期望:
  1. Qwen3ASR.__init__ 應有 aligner_enabled 參數 (預設 True)
  2. 旗標 --no-aligner 應在 main() 中被解析
  3. aligner_enabled=False 時,from_pretrained 不傳 forced_aligner/_kwargs
  4. 預設行為 (aligner=True) 不變,向後相容
"""
from __future__ import annotations
import inspect
import re
from pathlib import Path
import pytest


@pytest.mark.fast
def test_qwen3asr_has_aligner_enabled_param():
    """Qwen3ASR.__init__ 必須有 aligner_enabled 參數。"""
    from qwen3_asr import Qwen3ASR
    sig = inspect.signature(Qwen3ASR.__init__)
    assert "aligner_enabled" in sig.parameters, (
        "Qwen3ASR.__init__ 缺 aligner_enabled 參數"
    )


@pytest.mark.fast
def test_qwen3asr_aligner_default_is_true():
    """aligner_enabled 預設必須是 True (向後相容,準度為先)。"""
    from qwen3_asr import Qwen3ASR
    sig = inspect.signature(Qwen3ASR.__init__)
    p = sig.parameters["aligner_enabled"]
    assert p.default is True, f"預設應為 True (準度為先),實際: {p.default}"


@pytest.mark.fast
def test_no_aligner_cli_flag_exists():
    """qwen3_asr.py 主程式必須有 --no-aligner 旗標。"""
    src = Path(__file__).parent.parent / "qwen3_asr.py"
    code = src.read_text(encoding="utf-8")
    # 需要在 argparse 中註冊
    assert '--no-aligner' in code, "缺少 --no-aligner CLI 旗標"


@pytest.mark.fast
def test_qwen3_load_skips_aligner_kwargs_when_disabled(monkeypatch):
    """關閉 aligner 時, from_pretrained 不可帶 forced_aligner/_kwargs。"""
    from qwen3_asr import Qwen3ASR, HwConfig

    captured_kwargs = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, **kwargs):
            captured_kwargs.update(kwargs)
            obj = cls()
            obj.model = obj  # 滿足 inner = getattr(...,'model',None) or self.model
            return obj
        def transcribe(self, **kwargs):
            return []

    # 攔截 qwen_asr import
    import sys
    fake_module = type(sys)("qwen_asr")
    fake_module.Qwen3ASRModel = FakeModel
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_module)

    cfg = HwConfig(device="cpu", compile_mode=None)
    asr = Qwen3ASR(cfg, aligner_enabled=False)
    asr.load()
    assert "forced_aligner" not in captured_kwargs, (
        "aligner 關閉時不應傳 forced_aligner"
    )
    assert "forced_aligner_kwargs" not in captured_kwargs, (
        "aligner 關閉時不應傳 forced_aligner_kwargs"
    )


@pytest.mark.fast
def test_qwen3_load_includes_aligner_when_enabled(monkeypatch):
    """預設 (aligner=True) 必須傳 forced_aligner。"""
    from qwen3_asr import Qwen3ASR, HwConfig

    captured_kwargs = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, **kwargs):
            captured_kwargs.update(kwargs)
            obj = cls()
            obj.model = obj
            return obj
        def transcribe(self, **kwargs):
            return []

    import sys
    fake_module = type(sys)("qwen_asr")
    fake_module.Qwen3ASRModel = FakeModel
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_module)

    cfg = HwConfig(device="cpu", compile_mode=None)
    asr = Qwen3ASR(cfg)  # 預設 aligner_enabled=True
    asr.load()
    assert captured_kwargs.get("forced_aligner") == Qwen3ASR.ALIGNER, (
        "預設應載入 forced_aligner"
    )
