# -*- coding: utf-8 -*-
"""TDD: Breeze 接收 glossary 並注入 initial_prompt / hotwords
紅燈期望:
  1. FasterWhisperBackend.__init__ 接受 glossary 參數 (list[str], 預設 [])
  2. CLI --glossary-file 旗標存在
  3. 給 glossary 時, transcribe 內 common['initial_prompt'] 含至少 5 個詞
  4. transcribe 內 common['hotwords'] 為 glossary 詞 join 字串
  5. 不給 glossary 時行為與目前一致 (向後相容)
"""
from __future__ import annotations
import inspect
from pathlib import Path
import pytest


@pytest.mark.fast
def test_faster_whisper_backend_accepts_glossary():
    from taiwan_asr.breeze import FasterWhisperBackend
    sig = inspect.signature(FasterWhisperBackend.__init__)
    assert "glossary" in sig.parameters, "FasterWhisperBackend.__init__ 缺 glossary 參數"


@pytest.mark.fast
def test_glossary_default_is_empty_list():
    from taiwan_asr.breeze import FasterWhisperBackend
    sig = inspect.signature(FasterWhisperBackend.__init__)
    p = sig.parameters["glossary"]
    # 預設 [] 或 None,允許任一,但行為應等同無 glossary
    assert p.default in (None, [], ()), f"預設應為空 (向後相容):{p.default}"


@pytest.mark.fast
def test_breeze_glossary_file_cli_flag():
    src = Path(__file__).parent.parent / "src/taiwan_asr/breeze.py"
    code = src.read_text(encoding="utf-8")
    assert "--glossary-file" in code, "缺 --glossary-file CLI 旗標"


@pytest.mark.fast
def test_breeze_initial_prompt_has_glossary_terms(monkeypatch):
    """注入 glossary → initial_prompt 必須含 ≥5 個詞,且 hotwords 帶字串。"""
    import sys
    import numpy as np
    from taiwan_asr.breeze import FasterWhisperBackend, HwConfig
    from taiwan_asr.common import S2TW

    captured = {}

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            class _Info: language = "zh"; language_probability = 1.0
            return iter([]), _Info()

    class FakeBatched:
        def __init__(self, model=None): pass
        def transcribe(self, audio, batch_size=8, **kwargs):
            captured.update(kwargs)
            class _Info: language = "zh"; language_probability = 1.0
            return iter([]), _Info()

    fw_module = type(sys)("faster_whisper")
    fw_module.WhisperModel = FakeModel
    fw_module.BatchedInferencePipeline = FakeBatched
    monkeypatch.setitem(sys.modules, "faster_whisper", fw_module)

    # mock AudioIO + VAD
    from taiwan_asr.common import AudioIO, SileroVAD
    monkeypatch.setattr(
        AudioIO, "decode_to_array",
        staticmethod(lambda src, sr=16000: (np.zeros(16000, dtype=np.float32), 1.0)),
    )
    monkeypatch.setattr(
        SileroVAD, "speech_chunks",
        lambda self, audio: [{"audio": (audio, 16000), "start": 0.0, "end": 1.0}],
    )

    cfg = HwConfig(device="cpu")
    glossary = ["延三舍", "圓山宿舍", "祝福二組", "男一", "女一", "課指組"]
    backend = FasterWhisperBackend(cfg, beam=1, s2tw=S2TW(False),
                                    use_manual_vad=True, glossary=glossary)
    # 直接植入 mock,跳過 _ensure_ct2 / load
    backend.model = FakeModel()
    backend.batched = FakeBatched()

    backend.transcribe("dummy.wav", language="zh")

    init_prompt = captured.get("initial_prompt", "")
    n_in_prompt = sum(1 for t in glossary if t in init_prompt)
    assert n_in_prompt >= 5, (
        f"initial_prompt 應含 ≥5 個 glossary 詞,實際含 {n_in_prompt}\n"
        f"prompt = {init_prompt!r}"
    )
    # hotwords 也應傳遞 (faster-whisper >=1.0 支援)
    hot = captured.get("hotwords", "")
    assert hot, "hotwords 參數未傳遞"
    assert any(t in hot for t in glossary), f"hotwords 不含 glossary 詞:{hot!r}"


@pytest.mark.fast
def test_breeze_no_glossary_backward_compat(monkeypatch):
    """無 glossary 時行為不變 (initial_prompt 仍存在但不額外加詞)。"""
    import sys
    import numpy as np
    from taiwan_asr.breeze import FasterWhisperBackend, HwConfig
    from taiwan_asr.common import S2TW

    captured = {}

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            class _Info: language = "zh"; language_probability = 1.0
            return iter([]), _Info()
    class FakeBatched:
        def __init__(self, model=None): pass
        def transcribe(self, audio, batch_size=8, **kwargs):
            captured.update(kwargs)
            class _Info: language = "zh"; language_probability = 1.0
            return iter([]), _Info()
    fw_module = type(sys)("faster_whisper")
    fw_module.WhisperModel = FakeModel
    fw_module.BatchedInferencePipeline = FakeBatched
    monkeypatch.setitem(sys.modules, "faster_whisper", fw_module)

    from taiwan_asr.common import AudioIO, SileroVAD
    monkeypatch.setattr(
        AudioIO, "decode_to_array",
        staticmethod(lambda src, sr=16000: (np.zeros(16000, dtype=np.float32), 1.0)),
    )
    monkeypatch.setattr(
        SileroVAD, "speech_chunks",
        lambda self, audio: [{"audio": (audio, 16000), "start": 0.0, "end": 1.0}],
    )

    cfg = HwConfig(device="cpu")
    backend = FasterWhisperBackend(cfg, beam=1, s2tw=S2TW(False), use_manual_vad=True)
    backend.model = FakeModel()
    backend.batched = FakeBatched()
    backend.transcribe("dummy.wav", language="zh")

    init_prompt = captured.get("initial_prompt", "")
    # 至少要有原本的繁體提示 (向後相容)
    assert init_prompt and "繁體中文" in init_prompt, (
        f"無 glossary 時 initial_prompt 應仍含原繁體提示:{init_prompt!r}"
    )
