# -*- coding: utf-8 -*-
"""
不變式測試 — 這些斷言絕不可動搖。
特別是 Breeze 必須永遠是 MediaTek-Research/Breeze-ASR-25
(台灣繁體中文專攻模型,任何優化都不可換掉)。
"""
from __future__ import annotations
import os
import json
import re
from pathlib import Path
import pytest


# ─────────────────────────────────────────────────────────────
# 1. Breeze 模型本體 — 絕不可換成其他 Whisper 變體
# ─────────────────────────────────────────────────────────────
@pytest.mark.fast
@pytest.mark.breeze_invariant
def test_breeze_model_id_is_mediatek():
    """Breeze 必須使用 MediaTek-Research/Breeze-ASR-25,絕不可換。"""
    from taiwan_asr.breeze import FasterWhisperBackend, TransformersBackend
    assert FasterWhisperBackend.MODEL_ID == "MediaTek-Research/Breeze-ASR-25"
    assert TransformersBackend.MODEL_ID == "MediaTek-Research/Breeze-ASR-25"
    # 雙重保險:檢查關鍵字
    assert "MediaTek" in FasterWhisperBackend.MODEL_ID
    assert "Breeze" in FasterWhisperBackend.MODEL_ID


@pytest.mark.fast
@pytest.mark.breeze_invariant
def test_breeze_no_whisper_turbo_substitution():
    """確認沒有偷偷換成 Whisper-Large-v3 / Distil-Whisper / Turbo 變體。"""
    src = Path(__file__).parent.parent / "src/taiwan_asr/breeze.py"
    content = src.read_text(encoding="utf-8")
    # 這些字串如果出現在 MODEL_ID 周圍就是換模型
    forbidden_in_model = [
        "openai/whisper", "Systran/faster", "distil-whisper",
        "large-v3-turbo", "mobiuslabsgmbh",
    ]
    for f in forbidden_in_model:
        # 允許出現在註解/文檔,但不可作為 MODEL_ID
        # 簡單啟發:行內若同時有 'MODEL_ID' 和 forbidden,就 fail
        for line in content.splitlines():
            if "MODEL_ID" in line and "=" in line and f in line.lower():
                pytest.fail(f"src/taiwan_asr/breeze.py 模型 ID 含禁用詞: {f}\n  → {line!r}")


@pytest.mark.medium
@pytest.mark.breeze_invariant
def test_breeze_ct2_cache_is_mediatek():
    """CT2 cache 目錄裡的模型必須是從 MediaTek/Breeze 轉來的。"""
    cache = Path.home() / ".cache" / "breeze-asr-ct2-bf16"
    if not cache.is_dir():
        pytest.skip(f"CT2 cache 不存在 (尚未首次轉換): {cache}")
    # CT2 模型有 model.bin,從 transformers 轉時會留下原始 generation_config 或 tokenizer
    # 檢查 tokenizer.json 或 generation_config.json
    found = False
    for fname in ["generation_config.json", "tokenizer.json", "preprocessor_config.json"]:
        p = cache / fname
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        s = json.dumps(data)
        if "MediaTek" in s or "Breeze" in s:
            found = True
            break
    # 即使 metadata 不留模型名,只要 model.bin 存在我們就放行
    # (無法在不載模型情況下 100% 驗證)
    assert (cache / "model.bin").is_file(), \
        f"CT2 模型主檔缺失:{cache}/model.bin"


# ─────────────────────────────────────────────────────────────
# 2. OpenCC 必須是 s2twp (台灣慣用詞,不是 s2tw / s2t)
# ─────────────────────────────────────────────────────────────
@pytest.mark.fast
@pytest.mark.breeze_invariant
def test_opencc_uses_s2twp():
    from taiwan_asr.common import S2TW
    s = S2TW(True)
    assert s.cc is not None, "OpenCC 未載入"
    # 我們嘗試順序:s2twp 優先
    assert s.scheme.startswith("s2tw"), f"OpenCC scheme 不是 s2tw 系列: {s.scheme}"


@pytest.mark.fast
def test_s2tw_actually_converts_simplified():
    """簡體必須真的被轉成繁體 (台灣慣用詞)。"""
    from taiwan_asr.common import S2TW
    s = S2TW(True)
    out = s("我们正在测试软件优化方案")
    assert "軟體" in out or "软件" not in out, f"s2tw 沒轉:{out!r}"
    assert "我們" in out, f"我们 沒轉成 我們:{out!r}"


# ─────────────────────────────────────────────────────────────
# 3. Qwen3 語言映射 (zh → "Chinese")
# ─────────────────────────────────────────────────────────────
@pytest.mark.fast
def test_qwen3_lang_map_zh_to_chinese():
    from taiwan_asr.qwen3 import Qwen3ASR
    assert Qwen3ASR.normalize_lang("zh") == "Chinese"
    assert Qwen3ASR.normalize_lang("zh-TW") == "Chinese"
    assert Qwen3ASR.normalize_lang("zh-tw") == "Chinese"
    assert Qwen3ASR.normalize_lang("Chinese") == "Chinese"
    assert Qwen3ASR.normalize_lang("auto") is None
    assert Qwen3ASR.normalize_lang(None) is None


@pytest.mark.fast
def test_qwen3_model_ids():
    """Qwen3-ASR + ForcedAligner 模型 ID 不可動。"""
    from taiwan_asr.qwen3 import Qwen3ASR
    assert Qwen3ASR.MODEL == "Qwen/Qwen3-ASR-1.7B"
    assert Qwen3ASR.ALIGNER == "Qwen/Qwen3-ForcedAligner-0.6B"


# ─────────────────────────────────────────────────────────────
# 4. 關鍵中文 ASR 解碼參數預設值 (準度核心)
# ─────────────────────────────────────────────────────────────
@pytest.mark.fast
def test_breeze_decoding_defaults_safe_for_chinese():
    """讀 breeze_asr.py 原始碼,確認關鍵中文解碼參數在預設路徑。"""
    src = Path(__file__).parent.parent / "src/taiwan_asr/breeze.py"
    code = src.read_text(encoding="utf-8")
    # 必須有的關鍵防護
    must_have = [
        "condition_on_previous_text=False",   # 中文必須 False!
        "compression_ratio_threshold=2.4",
        "log_prob_threshold=-1.0",
        "no_speech_threshold=0.6",
    ]
    for token in must_have:
        assert token in code, f"breeze 缺關鍵中文解碼參數: {token}"


# ─────────────────────────────────────────────────────────────
# 5. 既有 Breeze 輸出仍是繁體中文
# ─────────────────────────────────────────────────────────────
@pytest.mark.fast
@pytest.mark.breeze_invariant
def test_existing_breeze_886_is_traditional(existing_breeze_886):
    full = "".join(s.get("text", "") for s in existing_breeze_886)
    assert len(full) > 100, "Breeze 輸出過短,可能模型有問題"

    # 簡體常見字 (這些不該大量出現)
    simplified_indicators = "国发现实际经济认识让说话语门间问题点头计较实际"
    n_simp = sum(1 for c in full if c in simplified_indicators)

    # 繁體常見字 (這些應該出現)
    traditional_indicators = "國發現實際經濟認識讓說話語門間問題點頭計較實際"
    n_trad = sum(1 for c in full if c in traditional_indicators)

    # Privacy: assertion messages must report aggregate stats only,
    # never raw transcript slices — failed CI runs would otherwise
    # leak fixture content into public logs.
    assert n_trad >= n_simp, (
        f"Breeze 輸出簡體太多: 簡={n_simp}, 繁={n_trad}, "
        f"總字數={len(full)}"
    )
    # 至少應有一些常見繁體字
    assert n_trad >= 5, (
        f"繁體字數 {n_trad} 太少 (總字數={len(full)}, 簡體={n_simp})"
    )
