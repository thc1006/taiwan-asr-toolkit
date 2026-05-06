# -*- coding: utf-8 -*-
"""
TDD: 多檔 chunk pool batching (Qwen3)
紅燈期望:
  1. Qwen3ASR.transcribe_files(paths, language) 方法存在,回傳 dict[path -> List[Segment]]
  2. 跨檔 chunk pool 後,每檔 segment 一定是依照原檔的時間順序
  3. main() 有 --no-pool 旗標 (預設 pool 開,多檔時自動 batch 共用)
"""
from __future__ import annotations
import inspect
import re
from pathlib import Path
import pytest


@pytest.mark.fast
def test_qwen3_has_transcribe_files_method():
    """Qwen3ASR 必須有 transcribe_files 方法支援多檔 pool。"""
    from qwen3_asr import Qwen3ASR
    assert hasattr(Qwen3ASR, "transcribe_files"), (
        "Qwen3ASR 缺少 transcribe_files(paths, language) 方法"
    )


@pytest.mark.fast
def test_transcribe_files_signature():
    """簽名:transcribe_files(self, paths, language=...)"""
    from qwen3_asr import Qwen3ASR
    sig = inspect.signature(Qwen3ASR.transcribe_files)
    params = list(sig.parameters.keys())
    assert "paths" in params
    assert "language" in params


@pytest.mark.fast
def test_no_pool_cli_flag_exists():
    """qwen3_asr.py 必須有 --no-pool 旗標。"""
    src = Path(__file__).parent.parent / "qwen3_asr.py"
    code = src.read_text(encoding="utf-8")
    assert '--no-pool' in code, "缺少 --no-pool CLI 旗標"


@pytest.mark.fast
def test_pool_returns_dict_keyed_by_path(monkeypatch):
    """transcribe_files 回傳值必須是 {path: [Segment]} 結構。"""
    from qwen3_asr import Qwen3ASR, HwConfig
    import sys, numpy as np

    # 用 monkeypatch 攔截真實模型,只測試 pool 邏輯
    class FakeResult:
        def __init__(self, text, lang="Chinese"):
            self.text = text
            self.language = lang
    class FakeModel:
        @classmethod
        def from_pretrained(cls, **kwargs):
            obj = cls(); obj.model = obj; return obj
        def transcribe(self, audio, language=None, return_time_stamps=True):
            # 對每個 chunk 回傳一個假 result;依 audio 長度區分
            if not isinstance(audio, list):
                audio = [audio]
            return [FakeResult(f"段{len(a[0])}") for a in audio]

    fake_module = type(sys)("qwen_asr")
    fake_module.Qwen3ASRModel = FakeModel
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_module)

    # Stub VAD: 給每檔回傳兩段
    from _asr_common import SileroVAD
    def fake_chunks(self, audio):
        return [
            {"audio": (np.zeros(8000, dtype=np.float32), 16000), "start": 0.0, "end": 0.5},
            {"audio": (np.zeros(16000, dtype=np.float32), 16000), "start": 0.5, "end": 1.5},
        ]
    monkeypatch.setattr(SileroVAD, "speech_chunks", fake_chunks)

    # Stub AudioIO.decode_to_array
    from _asr_common import AudioIO
    monkeypatch.setattr(
        AudioIO, "decode_to_array",
        staticmethod(lambda src, sr=16000: (np.zeros(int(2*16000), dtype=np.float32), 2.0)),
    )

    cfg = HwConfig(device="cpu", compile_mode=None, batch_size=4)
    asr = Qwen3ASR(cfg, aligner_enabled=False).load()

    paths = ["fake1.mp3", "fake2.mp3"]
    out, sw = asr.transcribe_files(paths, language="zh")

    assert isinstance(out, dict)
    assert set(out.keys()) == set(paths)
    for p, segs in out.items():
        assert len(segs) == 2, f"{p}: 應有 2 段,實際 {len(segs)}"
        # 段必須按 start 排序
        starts = [s.start for s in segs]
        assert starts == sorted(starts), f"{p}: 段未按時序排"
