# -*- coding: utf-8 -*-
"""驗證 hot-word 在 886 上真正修字 (鎖為 regression)。
前提:transcripts/breeze/標準錄音 886_breeze.json 是「啟用 glossary」的版本。
若未來重跑 Breeze 沒帶 --glossary-file,此測試會失敗,提醒使用者。
"""
import json
from pathlib import Path
import pytest


@pytest.mark.fast
def test_886_breeze_uses_yan3_not_yuan3(existing_breeze_886):
    """正確的詞是「研三」(研究生宿舍三),Hot-word 應修正 圓三/延三 → 研三。"""
    full = "".join(s.get("text", "") for s in existing_breeze_886)
    n_yan3 = full.count("研三")
    n_yuan3 = full.count("圓三")
    assert n_yan3 >= 1, (
        f"啟用 glossary 後 886 應含『研三』(NTU 研究生宿舍),實際前 200 字: {full[:200]}\n"
        f"請重跑: python3 breeze_asr.py 'music/標準錄音 886.mp3' --glossary-file glossary.txt"
    )
    assert n_yan3 >= n_yuan3, (
        f"研三 出現次數應 >= 圓三 (hot-word 應抑制錯字)\n"
        f"研三={n_yan3}, 圓三={n_yuan3}"
    )


@pytest.mark.fast
def test_886_breeze_no_2zu_simplified_族(existing_breeze_886):
    """祝福二族 / 住輔二族 (族字錯) 應被矯正為 二組。"""
    full = "".join(s.get("text", "") for s in existing_breeze_886)
    # 「二族」這個錯字應被縮減
    n_zuzu = full.count("二族")
    n_zuzu_correct = full.count("二組")
    assert n_zuzu_correct >= n_zuzu, (
        f"二組 應 >= 二族 (組織名應為 組 不是 族),實際 二組={n_zuzu_correct} 二族={n_zuzu}"
    )
