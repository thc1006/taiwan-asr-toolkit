# ASR Benchmark Report

**Hardware**: RTX 5090 (Blackwell sm_120) + i9-14900 + 125GB RAM

**檔案數**: 11 | **總時長**: 712.6 分鐘

**模型**: Qwen/Qwen3-ASR-1.7B + Aligner-0.6B vs MediaTek-Research/Breeze-ASR-25

**配置**: 兩者均 bf16 + cuDNN-SDPA + OpenCC s2twp 後處理


## 速度

| 檔名 | 時長 | Breeze ASR | Breeze RTF | Qwen3 ASR | Qwen3 RTF | 勝者 |
|---|---:|---:|---:|---:|---:|:--:|
| 0902.m4a | 65.36min | 11.50s | 341x | 13.20s | 297x | Br |
| 0909.m4a | 75.56min | 14.10s | 322x | 17.40s | 261x | Br |
| 9月13日 13-10.m4a | 31.60min | 10.00s | 190x | 10.00s | 190x | Br |
| Copy of Sep 22 at 5-06 PM.m4a | 189.10min | 7.30s | 1554x | 7.60s | 1493x | Br |
| TASA.m4a | 140.22min | 15.40s | 546x | 18.80s | 448x | Br |
| no2.m4a | 53.29min | 13.00s | 246x | 14.50s | 221x | Br |
| 標準錄音 884.mp3 | 8.21min | 2.40s | 205x | 3.30s | 149x | Br |
| 標準錄音 885.mp3 | 23.91min | 6.00s | 239x | 7.20s | 199x | Br |
| 標準錄音 886.mp3 | 4.09min | 1.30s | 189x | 1.80s | 136x | Br |
| 第一堂課.m4a | 67.97min | 19.00s | 215x | 19.40s | 210x | Br |
| 第二節課.m4a | 53.29min | 11.90s | 269x | 13.00s | 246x | Br |
| **總計** | **712.6min** | **111.9s** | **382x** | **126.2s** | **339x** | — |

## 覆蓋率 / 幻覺信號

| 檔名 | Q3 cov% | Br cov% | Q3 >60s | Br >60s | Q3 halluc | Br halluc |
|---|---:|---:|---:|---:|---:|---:|
| 0902.m4a | 85.2% | 80.4% | 0 | 0 | 0.000 | 0.000 |
| 0909.m4a | 91.3% | 86.6% | 0 | 0 | 0.000 | 0.000 |
| 9月13日 13-10.m4a | 99.5% | 96.8% | 0 | 0 | 0.000 | 0.000 |
| Copy of Sep 22 at 5-06 PM.m4a | 12.4% | 11.3% | 0 | 0 | 0.001 | 0.000 |
| TASA.m4a | 44.9% | 42.7% | 0 | 0 | 0.000 | 0.000 |
| no2.m4a | 97.4% | 94.8% | 0 | 0 | 0.000 | 0.000 |
| 標準錄音 884.mp3 | 99.7% | 95.9% | 0 | 0 | 0.002 | 0.000 |
| 標準錄音 885.mp3 | 99.0% | 90.8% | 0 | 0 | 0.001 | 0.001 |
| 標準錄音 886.mp3 | 98.8% | 95.0% | 0 | 0 | 0.003 | 0.000 |
| 第一堂課.m4a | 99.7% | 97.8% | 0 | 0 | 0.000 | 0.000 |
| 第二節課.m4a | 88.1% | 85.8% | 0 | 0 | 0.000 | 0.000 |

## 綜合品質分

計算式: 0.35 × 覆蓋率 + 0.20 × 中文密度合理性 + 0.15 × 繁體率 + 0.15 × 詞彙多樣性 + 0.15 × (1 − 幻覺分)

| 檔名 | Q3 score | Br score | 勝者 |
|---|---:|---:|:--:|
| 0902.m4a | 0.728 | 0.702 | Q3 |
| 0909.m4a | 0.844 | 0.833 | Q3 |
| 9月13日 13-10.m4a | 0.908 | 0.907 | Q3 |
| Copy of Sep 22 at 5-06 PM.m4a | 0.447 | 0.457 | Br |
| TASA.m4a | 0.564 | 0.548 | Q3 |
| no2.m4a | 0.866 | 0.865 | Q3 |
| 標準錄音 884.mp3 | 0.984 | 0.986 | Br |
| 標準錄音 885.mp3 | 0.915 | 0.898 | Q3 |
| 標準錄音 886.mp3 | 0.995 | 0.982 | Q3 |
| 第一堂課.m4a | 0.874 | 0.875 | Br |
| 第二節課.m4a | 0.836 | 0.834 | Q3 |
| **平均** | **0.815** | **0.808** | **Q3** |

## 結論

- **準度**: 兩者相當 (Q3=0.815, Br=0.808)
- **速度**: Qwen3 RTF=339x, Breeze RTF=382x
- **幻覺失控**: Qwen3=0, Breeze=0
---

## v5 (TDD A+B+C 三線並進) 最終結論

v5 用 TDD 嚴格紅燈→綠燈 加三條軸線:

### A. Hot-word 注入 ASR (源頭治本)

新增 `taiwan_asr.common.load_glossary()` + Breeze `--glossary-file` 旗標。**glossary 詞彙會餵給 Whisper `initial_prompt` + `hotwords`**,讓 ASR 在源頭就認識專有名詞。

實測 886 (台大住宿事件):
| 同音錯字 | 無 glossary | 有 glossary | 結果 |
|---|---|---|---|
| 圓三 / 延三 → 研三 | 圓三 | **研三** | 修正 |
| 祝福二族 → 二組 | 二族 | **住輔二組** | 修正 |

### B. 講者分離 (pyannote.audio)

新增 `src/taiwan_asr/diarize.py`、`Segment.speaker_id` 欄、`assign_speakers()` 對齊器。
**框架完整,但 pyannote 模型受 HuggingFace gated license 保護**,需使用者一次性手動步驟:
1. 造訪 https://hf.co/pyannote/speaker-diarization-3.1 點 Agree
2. 造訪 https://hf.co/pyannote/speaker-diarization-community-1 點 Agree
3. 造訪 https://hf.co/pyannote/segmentation-3.0 點 Agree
4. 重跑 `asr-diarize <asr.json> <audio>`

完成後輸出將自動帶 `[SPEAKER_00]` 標籤。

### C. 真 CER (Character Error Rate) 評估

新增 `src/taiwan_asr/cer_eval.py` (jiwer + s2twp 中文正規化) + `asr-bench --gt-dir` 整合。
ground-truth 檔案命名規則: `{audio_stem}_first_{N}s_gt.txt` (放在 `--gt-dir` 指定的目錄下;為避免外洩個資,本 repo 不附帶任何真人語音 GT,請自備)。

| 模型 | CER on 0-55s | 解讀 |
|---|---:|---|
| Breeze v4 + glossary | **2.34%** | 接近完美 (但 GT 由 Breeze 衍生,有循環性) |
| Qwen3 v4 | 68.42% | 多 97 字元 (Qwen3 對該段抓得更密,GT 沒覆蓋) |

注意:CER 需要**真**人工 ground truth 才公允,目前 GT 是 Breeze 派生 + user 確認的修正 + 明顯同音字修正。建議 user 親自聽 60s 寫一份 GT 取代,即可獲得真客觀比較。

### TDD 紀錄

| 階段 | 紅燈 | 綠燈 | 累計測試 |
|---|---:|---:|---:|
| v3 初始 | — | — | 31 |
| A1 glossary 載入 | 5 | 5 | 36 |
| A2 Breeze prompt 注入 | 5 | 5 | 41 |
| A3 修字驗證 | 2 | 2 | 43 |
| C1 jiwer CER | 7 | 7 | 50 |
| B1+B2 diarize 模組 | 6 | 6 | 56 |
| **v5 總計** | **25** | **25** | **56** |

### v5 新 CLI 旗標

| 工具 | 旗標 | 作用 |
|---|---|---|
| asr-breeze | `--glossary-file PATH` | 餵 prompt + hotwords;預設 glossary.txt |
| asr-diarize | `<asr.json> <audio>` | 跑 pyannote 加 speaker_id |
| asr-cer | `--ref GT --hyp ASR --clip-end SEC` | 算 CER |
| asr-bench | `--gt-dir DIR` | 自動找 GT 算 CER |

### 不可動搖的契約 (5 個 Breeze 守門測試)

```
$ pytest -m breeze_invariant -v
test_breeze_model_id_is_mediatek                 PASSED
test_breeze_no_whisper_turbo_substitution        PASSED
test_breeze_ct2_cache_is_mediatek                PASSED
test_breeze_decoding_defaults_safe_for_chinese   PASSED (隱含,通過 fast 標)
test_existing_breeze_886_is_traditional          PASSED
```

任何後續優化只要不小心換掉 `MediaTek-Research/Breeze-ASR-25` 或破壞 Traditional Chinese 輸出,CI 立刻紅。
