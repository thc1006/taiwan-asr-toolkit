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
    from taiwan_asr.qwen3 import Qwen3ASR
    sig = inspect.signature(Qwen3ASR.__init__)
    assert "aligner_enabled" in sig.parameters, (
        "Qwen3ASR.__init__ 缺 aligner_enabled 參數"
    )


@pytest.mark.fast
def test_qwen3asr_aligner_default_is_true():
    """aligner_enabled 預設必須是 True (向後相容,準度為先)。"""
    from taiwan_asr.qwen3 import Qwen3ASR
    sig = inspect.signature(Qwen3ASR.__init__)
    p = sig.parameters["aligner_enabled"]
    assert p.default is True, f"預設應為 True (準度為先),實際: {p.default}"


@pytest.mark.fast
def test_no_aligner_cli_flag_exists():
    """qwen3_asr.py 主程式必須有 --no-aligner 旗標。"""
    src = Path(__file__).parent.parent / "src/taiwan_asr/qwen3.py"
    code = src.read_text(encoding="utf-8")
    # 需要在 argparse 中註冊
    assert '--no-aligner' in code, "缺少 --no-aligner CLI 旗標"


@pytest.mark.fast
def test_qwen3_load_skips_aligner_kwargs_when_disabled(monkeypatch):
    """關閉 aligner 時, from_pretrained 不可帶 forced_aligner/_kwargs。"""
    from taiwan_asr.qwen3 import Qwen3ASR, HwConfig

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
    from taiwan_asr.qwen3 import Qwen3ASR, HwConfig

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


# ─────────────────────────────────────────────────────────────────────────
# v0.5.4 regression: _transcribe_batch must NOT request return_time_stamps
# when no aligner is loaded — qwen_asr raises ValueError otherwise.
# ─────────────────────────────────────────────────────────────────────────


def _instance_with_mocked_model(*, aligner_enabled: bool):
    """Build a Qwen3ASR whose .model is a MagicMock, bypassing real load()."""
    from unittest.mock import MagicMock
    from types import SimpleNamespace
    import torch
    from taiwan_asr.qwen3 import HwConfig, Qwen3ASR

    cfg = HwConfig(
        device="cpu", dtype=torch.float32, batch_size=1, chunk_sec=4.0,
        attn_impl="eager", compile_mode=None, desc="test cpu",
    )
    asr = Qwen3ASR(cfg, s2tw_enabled=False, aligner_enabled=aligner_enabled)
    asr.model = MagicMock()
    asr.model.transcribe.return_value = [SimpleNamespace(text="測試", language="zh")]
    return asr


@pytest.mark.fast
def test_no_aligner_passes_return_time_stamps_false():
    """v0.5.4 fix: --no-aligner -> return_time_stamps=False at call time."""
    pytest.importorskip("torch")
    import numpy as np
    asr = _instance_with_mocked_model(aligner_enabled=False)
    dummy = (np.zeros(int(0.5 * 16000), dtype=np.float32), 16000)
    asr._transcribe_batch([dummy], "Chinese")

    asr.model.transcribe.assert_called_once()
    kwargs = asr.model.transcribe.call_args.kwargs
    assert kwargs.get("return_time_stamps") is False, (
        "Bug regression: --no-aligner must pass return_time_stamps=False to "
        "qwen_asr.transcribe; passing True triggers ValueError when no "
        "forced_aligner is loaded (this was the v0.5.3 Colab crash)."
    )


@pytest.mark.fast
def test_with_aligner_passes_return_time_stamps_true():
    """Default path (aligner loaded): keep word-level timestamps."""
    pytest.importorskip("torch")
    import numpy as np
    asr = _instance_with_mocked_model(aligner_enabled=True)
    dummy = (np.zeros(int(0.5 * 16000), dtype=np.float32), 16000)
    asr._transcribe_batch([dummy], "Chinese")

    kwargs = asr.model.transcribe.call_args.kwargs
    assert kwargs.get("return_time_stamps") is True, (
        "When the aligner IS loaded we still want character-level timestamps."
    )


@pytest.mark.fast
def test_warmup_no_aligner_does_not_raise():
    """The 2-second silence warmup must not blow up under --no-aligner.

    Pre-v0.5.4 the warmup raised ValueError and was silently swallowed by a
    bare except in load(), masking the bug until a real file came in.
    """
    pytest.importorskip("torch")
    import numpy as np
    asr = _instance_with_mocked_model(aligner_enabled=False)
    dummy = (np.zeros(int(2 * 16000), dtype=np.float32), 16000)
    out = asr._transcribe_batch([dummy], "Chinese")
    assert asr.model.transcribe.called
    assert out is not None
