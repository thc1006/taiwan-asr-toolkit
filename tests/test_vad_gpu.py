# -*- coding: utf-8 -*-
"""
TDD: GPU VAD
紅燈期望: SileroVAD 接受 device 參數,GPU/CPU 結果等價但 GPU 在長音檔較快。
"""
from __future__ import annotations
import os
import time
import inspect
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────
# 紅燈測試 1: device 參數必須存在
# ─────────────────────────────────────────────────────────────
@pytest.mark.fast
def test_silero_vad_accepts_device_param():
    """SileroVAD 必須有 device 建構參數。"""
    from taiwan_asr.common import SileroVAD
    sig = inspect.signature(SileroVAD.__init__)
    assert "device" in sig.parameters, "SileroVAD 必須有 device 參數 (cpu/cuda:0/auto)"


# ─────────────────────────────────────────────────────────────
# 測試 2: device='auto' 經實測選 cpu (Silero 模型太小,GPU 開銷反超)
# ─────────────────────────────────────────────────────────────
@pytest.mark.medium
def test_silero_vad_auto_picks_cpu_for_small_model():
    """device='auto' 應選 cpu (ONNX SIMD 對 Silero 這種小模型最快)。"""
    from taiwan_asr.common import SileroVAD
    vad = SileroVAD(device="auto")
    audio = np.zeros(int(0.5 * 16000), dtype=np.float32)
    vad.speech_chunks(audio)
    assert vad._using == "onnx", \
        f"auto 模式應選 ONNX CPU (對小模型最快),實際: {vad._using}"


# ─────────────────────────────────────────────────────────────
# 紅燈測試 3: GPU 與 CPU 在 30s 真實音訊輸出等價
# ─────────────────────────────────────────────────────────────
@pytest.mark.medium
def test_gpu_vad_matches_cpu_vad(clip_30s, cuda_available):
    """GPU/CPU VAD 對同一音訊產生段數一致、邊界誤差 <50ms。"""
    if not cuda_available:
        pytest.skip("cuda 不可用")
    from taiwan_asr.common import SileroVAD, AudioIO
    audio, _ = AudioIO.decode_to_array(str(clip_30s))

    cpu = SileroVAD(device="cpu")
    cpu_chunks = cpu.speech_chunks(audio)

    gpu = SileroVAD(device="cuda:0")
    gpu_chunks = gpu.speech_chunks(audio)

    assert len(cpu_chunks) == len(gpu_chunks), (
        f"段數不一致: CPU={len(cpu_chunks)} GPU={len(gpu_chunks)}\n"
        f"CPU: {[(c['start'],c['end']) for c in cpu_chunks]}\n"
        f"GPU: {[(c['start'],c['end']) for c in gpu_chunks]}"
    )
    for i, (a, b) in enumerate(zip(cpu_chunks, gpu_chunks)):
        assert abs(a["start"] - b["start"]) < 0.05, \
            f"段 {i} start: CPU={a['start']:.3f} GPU={b['start']:.3f}"
        assert abs(a["end"] - b["end"]) < 0.05, \
            f"段 {i} end: CPU={a['end']:.3f} GPU={b['end']:.3f}"


# ─────────────────────────────────────────────────────────────
# 測試 4: 純資訊 — 量出 CPU/GPU 各別速度,不主張誰較快
# (實測 Silero 太小,GPU 反慢;此測試僅紀錄供分析)
# ─────────────────────────────────────────────────────────────
@pytest.mark.medium
def test_vad_speed_informational(music_886, cuda_available):
    """測量 CPU 與 GPU VAD 速度供記錄,不下結論。"""
    if not cuda_available:
        pytest.skip("cuda 不可用,僅 GPU 路徑被 skip")
    from taiwan_asr.common import SileroVAD, AudioIO
    audio, _ = AudioIO.decode_to_array(str(music_886))

    # 預熱
    SileroVAD(device="cpu").speech_chunks(audio)
    SileroVAD(device="cuda:0").speech_chunks(audio)

    cpu = SileroVAD(device="cpu")
    t0 = time.perf_counter(); cpu.speech_chunks(audio); t_cpu = time.perf_counter() - t0
    gpu = SileroVAD(device="cuda:0")
    t0 = time.perf_counter(); gpu.speech_chunks(audio); t_gpu = time.perf_counter() - t0
    print(f"\n[info] VAD 4-min audio:  CPU(onnx)={t_cpu*1000:.0f}ms  "
          f"GPU(torch)={t_gpu*1000:.0f}ms — auto 預設 cpu")
    # 唯一斷言:兩者都能跑完
    assert t_cpu > 0 and t_gpu > 0


# ─────────────────────────────────────────────────────────────
# 紅燈測試 5: 向後相容 — 不指定 device 仍可運作 (預設 auto/cpu)
# ─────────────────────────────────────────────────────────────
@pytest.mark.fast
def test_silero_vad_default_still_works():
    from taiwan_asr.common import SileroVAD
    audio = np.zeros(int(0.5 * 16000), dtype=np.float32)
    vad = SileroVAD()  # 不傳 device
    chunks = vad.speech_chunks(audio)
    assert isinstance(chunks, list)
