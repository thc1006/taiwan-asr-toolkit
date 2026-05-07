<div align="center">

# Taiwan ASR Toolkit

### Production-grade Traditional Chinese (Taiwan Mandarin) speech-to-text — **RTF up to 1554x** on a single RTX 5090

**Qwen3-ASR-1.7B** + **MediaTek Breeze-ASR-25** · Hot-word injection · LLM context polish · Speaker diarization · OpenCC s2twp · 69 TDD tests

[![PyPI](https://img.shields.io/pypi/v/taiwan-asr-toolkit?label=PyPI&color=brightgreen)](https://pypi.org/project/taiwan-asr-toolkit/)
[![Downloads](https://img.shields.io/pypi/dm/taiwan-asr-toolkit?label=PyPI%20downloads)](https://pypi.org/project/taiwan-asr-toolkit/)
[![CI](https://github.com/thc1006/taiwan-asr-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/thc1006/taiwan-asr-toolkit/actions/workflows/tests.yml)
[![Release](https://img.shields.io/github/v/release/thc1006/taiwan-asr-toolkit?include_prereleases&sort=semver)](https://github.com/thc1006/taiwan-asr-toolkit/releases)
[![Tests](https://img.shields.io/badge/tests-69%20passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%20%7C%20Blackwell%20sm__120-76B900)](docs/INSTALL.md)
[![License](https://img.shields.io/badge/license-MIT-purple)](LICENSE)
[![繁體中文](https://img.shields.io/badge/output-繁體中文%20s2twp-red)](#features)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thc1006/taiwan-asr-toolkit/blob/main/examples/quickstart.ipynb)

[**Why this exists**](#why-this-exists) · [**Quick start**](#quick-start) · [**Benchmarks**](#benchmarks) · [**Usage**](#usage) · [**Architecture**](docs/ARCHITECTURE.md) · [**vs alternatives**](examples/compare_alternatives.md)

</div>

---

## Why this exists

If you've tried `openai/whisper-large-v3` or `whisperX` on **Taiwan Mandarin recordings**, you've hit:

- Output is **Simplified Chinese** by default (you keep getting `软件` instead of `軟體`)
- Whisper's built-in VAD silently fails on long sparse audio → 一個 48-min 失控段
- **Proper nouns die**: `延三舍 / 研三舍` (NTU dorms) become `圓三 / 圓山`
- Generic Whisper is **not tuned for Taiwan vocabulary** — homophone errors everywhere
- Variable-length VAD chunks waste 5-10x compute through padding

This toolkit fixes all of those. Two production-grade Mandarin ASR models, **identical pipeline** for fair comparison, **glossary-driven hot-word injection** at the source, **LLM polish** with proper-noun protection, **OpenCC s2twp** baked in. Tested on real lecture/interview/standard recordings.

> **Star this repo** if you've been burned by `condition_on_previous_text=True` on Mandarin. We feel you.

---

## Quick start

### Zero-effort try (Colab)

Click the **Open in Colab** badge at the top — runs the full pipeline on a
bundled 30-second sample with a Colab GPU, no install on your machine.

### Install from PyPI

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
pip install taiwan-asr-toolkit

# transcribe with the bundled NTU glossary (packaged in the wheel)
asr-breeze your_audio.mp3 --glossary-file builtin
```

### 30-second local test (clone, no audio needed)

```bash
git clone https://github.com/thc1006/taiwan-asr-toolkit.git && cd taiwan-asr-toolkit
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
pip install -e ".[all]"

asr-breeze tests/fixtures/clip_30s.wav --glossary-file builtin
cat transcripts/breeze/clip_30s_breeze.txt
```

That's it — 30 s of Taiwan Mandarin in, Traditional Chinese transcript out.

### On your own audio

```bash
asr-breeze path/to/your_audio.mp3 --glossary-file builtin
# or use your own glossary file:
asr-breeze path/to/your_audio.mp3 --glossary-file my_terms.txt
# Output: transcripts/breeze/{filename}_breeze.{txt,srt,json}
```

Output is **Traditional Chinese (Taiwan)**, with timestamps, segment-level + word-level
(Breeze) timing, and proper SRT subtitles.

---

## Benchmarks

Real numbers on a single **RTX 5090 (Blackwell sm_120, 32 GB GDDR7)** + **i9-14900** (24 threads).
Test corpus: **11 audio files, 712.6 minutes** (≈12 hours) of Taiwan-Mandarin lectures + interviews.

### Speed (Real-Time Factor — higher = faster)

| Audio file | Length | Breeze RTF | Qwen3 RTF |
|---|---:|---:|---:|
| 4-min standard recording   | 4 min | **189x** | 136x |
| 24-min standard recording | 24 min | **239x** | 199x |
| 65-min interview          | 65 min | **341x** | 297x |
| 140-min lecture (TASA)    | 140 min | **546x** | 448x |
| 189-min sparse audio      | 189 min | **1554x** | 1497x |
| **All 11 files combined** | **712 min** | **382x** | **354x** |
| **Total ASR time** | | **111.9 s** | **126.2 s** |

### Quality vs hallucination (proxy metrics, no GT)

| Metric | Qwen3 | Breeze |
|---|---:|---:|
| Avg quality score (35% coverage + 20% c/s + 15% Trad + 15% vocab + 15% no-halluc) | **0.815** | 0.808 |
| Per-file wins (out of 11) | **8** | 3 |
| **Catastrophic >60s segments** | **0** | **0** |
| Coverage % (transcribed vs audio time) | **83.3%** | 79.8% |
| OpenCC `s2twp` Traditional ratio | 0.97 | 0.97 |

### Real CER on hand-corrected fixture (886 first 55s)

| Model | CER | Notes |
|---|---:|---|
| Breeze-ASR-25 + glossary | **2.34%** | hot-word injection fixes 圓三→研三 at source |
| Qwen3-ASR-1.7B           | 68.42% | over-transcribes (97 extra chars not in fixed-time GT) |

Want to verify? `pytest tests/test_glossary_effect.py -v` — locks in the `圓三 → 研三` improvement as a regression test.

---

## Features

| | What it does |
|---|---|
| **Two SOTA Mandarin ASR models** | [Qwen/Qwen3-ASR-1.7B](https://hf.co/Qwen/Qwen3-ASR-1.7B) + [MediaTek-Research/Breeze-ASR-25](https://hf.co/MediaTek-Research/Breeze-ASR-25). Both run, both compared. |
| **Traditional Chinese always** | OpenCC `s2twp` post-processing converts any leftover 簡體 → 繁體 (Taiwan idioms): 軟件→軟體, 激光→雷射, 視頻→影片. |
| **Hot-word injection** | Pass `--glossary-file builtin` for the packaged NTU glossary (or your own .txt); proper nouns get fed to Whisper's `initial_prompt` + `hotwords`. Fixes `圓三 → 研三`, `祝福二族 → 住輔二組`, etc. **at the source**. |
| **Symmetric pipeline** | Same Silero VAD ONNX, same chunking, same dtype on both models. The benchmark measures **the model**, not the plumbing. |
| **Multi-file pool batching** | Cross-file length-sorted batching keeps batch=48 fully utilized when transcribing folders of mixed-length files. |
| **LLM context polish** | Optional Qwen3-8B post-correction with **NTU glossary protection** (won't accidentally "fix" `研三舍` to `延長`). |
| **Speaker diarization** | Optional pyannote 3.x integration with open-mirror fallback (no gated-license blocker). |
| **Real CER measurement** | jiwer-based CER with s2twp normalization. Bring your own ground-truth or use the included approximate fixture. |
| **69 TDD tests** | Including 5 invariant tests that **lock the Breeze model ID** so optimizations can't accidentally swap to a different Whisper variant. |
| **Blackwell-native** | bf16 + cuDNN-SDPA + torch.compile for RTX 5090. Auto-falls back gracefully on Hopper/Ada/Ampere/CPU. |

---

## Usage

### Single-file transcription

```bash
# Breeze (Whisper-Large-v2 fine-tune, fastest)
asr-breeze "music/lecture.mp3" --glossary-file builtin

# Qwen3-ASR (more comprehensive coverage)
asr-qwen3 "music/interview.m4a"

# Both at once (same audio, two transcripts to compare)
./run.sh both "music/standard_recording.mp3"
```

### Batch transcription (auto pool batching)

```bash
# Transcribes everything in music/ via Qwen3 with pool batching
asr-qwen3 music/*.mp3 music/*.m4a

# Same with Breeze (now also gets cross-file batched if multiple files)
asr-breeze music/*.{mp3,m4a,wav} --glossary-file builtin
```

### Power-user flags

| Flag | Effect |
|---|---|
| `--glossary-file PATH` | (Breeze) Inject domain terms via Whisper's prompt + hotwords. Use `--glossary-file builtin` for the packaged NTU dorm/dept glossary, or pass your own .txt file. |
| `--fast` | (Breeze) `int8_bfloat16` quantization, ~1.5x speedup, +0.3-0.5% CER on Mandarin |
| `--beam N` | (Breeze) Beam size; default 5. `--beam 1` = greedy (fastest), `--beam 10` = max accuracy |
| `--no-aligner` | (Qwen3) Skip ForcedAligner-0.6B; ~25% faster but loses word-level timestamps |
| `--no-pool` | (Qwen3) Disable cross-file chunk pooling for multi-file runs |
| `--internal-vad` | (Breeze) Use faster-whisper's built-in VAD instead of our ONNX Silero (not recommended) |
| `--no-s2tw` | Disable OpenCC s2twp post-processing |

### LLM context polish (post-process)

```bash
# Polish a Breeze output with Qwen3-8B + glossary protection
asr-polish transcripts/breeze/lecture_breeze.json --glossary-file builtin
# → transcripts/breeze-polished/lecture_breeze-polished.{txt,srt,json}
```

### Speaker diarization

```bash
# Adds [SPEAKER_00] / [SPEAKER_01] labels to each segment
asr-diarize transcripts/breeze/interview_breeze.json music/interview.m4a
# Requires HF license accept on:
# - https://hf.co/pyannote/speaker-diarization-3.1
# - https://hf.co/pyannote/speaker-diarization-community-1
# - https://hf.co/pyannote/segmentation-3.0
```

### Benchmark + CER report

```bash
# Generates docs/BENCHMARK.md with speed + quality metrics
asr-bench --gt-dir tests/fixtures
```

---

## Architecture

```
audio (.mp3/.m4a/.wav/...)
       ↓ ffmpeg pipe → numpy float32 16kHz mono
       ↓
Silero VAD ONNX (CPU SIMD, ~3-5x faster than PyTorch backend)
       ↓ ≤28s chunks
       ├──→ Qwen3-ASR-1.7B + ForcedAligner (HF transformers, bf16, batch=48)
       └──→ Breeze-ASR-25 (CTranslate2, bf16, batch=32, beam=5, hotwords)
                                ↓
                  OpenCC s2twp 簡→繁(台灣慣用詞)
                                ↓
              ┌───────────┬─────────────┐
              ↓           ↓             ↓
            TXT          SRT          JSON

   ┌──── Optional post-processing ────┐
   │ asr-polish  asr-diarize  asr-bench  │
   │ Qwen3-8B    pyannote     CER + RTF     │
   └──────────────────────────────────┘
```

Full architectural details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Project structure

```
taiwan-asr-toolkit/
├── src/taiwan_asr/
│   ├── __init__.py    ← package; minimal export to avoid heavy import cascades
│   ├── common.py      ← shared utils: ffmpeg pipe, OpenCC, Silero VAD, glossary, Segment
│   ├── qwen3.py       ← Qwen3-ASR + multi-file chunk pool
│   ├── breeze.py      ← Breeze-ASR-25 with manual VAD + hot-word injection
│   ├── polish.py      ← Qwen3-8B LLM context correction (glossary-protected)
│   ├── diarize.py     ← pyannote.audio speaker diarization
│   ├── cer_eval.py    ← jiwer-based CER with s2twp normalization
│   └── benchmark.py   ← speed + accuracy report
├── glossary.txt       ← default NTU glossary (dorm/dept names)
├── run.sh             ← convenience wrapper around asr-* CLI commands
├── pyproject.toml     ← project metadata, deps, CLI scripts (asr-qwen3, asr-breeze, …)
├── tests/             ← 69 TDD tests (including 5 Breeze invariants)
├── docs/              ← BENCHMARK.md / ARCHITECTURE.md / INSTALL.md
└── archive/           ← legacy Colab notebooks (kept for reference only)
```

After `pip install -e .` the following CLI commands are on PATH: `asr-qwen3`, `asr-breeze`, `asr-polish`, `asr-diarize`, `asr-bench`, `asr-cer`.

---

## Testing & contributing

```bash
# All 69 tests, no model load required for "fast" tier
pytest -m fast

# Breeze contract tests (NEVER allowed to fail)
pytest -m breeze_invariant

# Full suite (some need VAD load)
pytest
```

The toolkit follows strict TDD. **Any contribution must:**
1. Have a failing test that now passes
2. Keep all 56 existing tests green
3. Pass `pytest -m breeze_invariant` (which locks `MediaTek-Research/Breeze-ASR-25` as Breeze's model)
4. Keep Traditional Chinese (Taiwan) output

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide.

---

## vs alternatives

| | This toolkit | `whisperX` | `faster-whisper` (raw) | `openai/whisper` |
|---|:---:|:---:|:---:|:---:|
| Taiwan Traditional Chinese by default | s2twp baked-in | |  | |
| Two SOTA Mandarin models compared | Qwen3 + Breeze | Whisper only | Whisper only | Whisper only |
| Fixes `圓三/延三 → 研三` proper-noun ASR errors | glossary hot-word | |  manual prompt | |
| LLM context polish with proper-noun protection | Qwen3-8B + glossary | |  | |
| Speaker diarization (open-mirror fallback) | tensorlake mirror | pyannote (gated) | |  |
| RTX 5090 / Blackwell native (bf16 + cuDNN-SDPA) | |  | |  |
| TDD with model-invariant lock | 69 tests | |  | |
| Best RTF on long Mandarin audio | **1554x** | ~70x | ~250x | ~30x |

---

## Citation & credits

This toolkit is **integration plumbing** — credit goes to the model authors:

- **MediaTek-Research/Breeze-ASR-25** ([HuggingFace](https://hf.co/MediaTek-Research/Breeze-ASR-25)) — Whisper-Large-v2 fine-tune for Taiwan Mandarin
- **Alibaba/Tongyi Qwen3-ASR-1.7B** ([HuggingFace](https://hf.co/Qwen/Qwen3-ASR-1.7B)) — multilingual ASR with ForcedAligner
- **OpenAI Whisper** — base architecture for Breeze-ASR-25
- **CTranslate2 / faster-whisper** — Whisper inference engine
- **pyannote/audio** — speaker diarization
- **OpenCC** — simplified-traditional Chinese conversion (s2twp recipe)
- **Silero VAD** — fast voice activity detection

If this toolkit helps your research, please cite the underlying models. A toolkit-level citation is fine but optional:

```bibtex
@software{taiwan_asr_toolkit,
  title = {Taiwan ASR Toolkit: Production-grade Traditional Chinese Speech-to-Text Pipeline},
  author = {Taiwan ASR Toolkit Contributors},
  year   = {2026},
  url    = {https://github.com/thc1006/taiwan-asr-toolkit},
  note   = {Qwen3-ASR-1.7B + MediaTek Breeze-ASR-25 with hot-word injection, LLM polish, and speaker diarization}
}
```

---

## License

MIT for this toolkit's code. See [`LICENSE`](LICENSE).

Third-party model licenses (you must comply with each):
- Qwen models: Apache 2.0
- Breeze-ASR-25: Apache 2.0
- Silero VAD: MIT
- pyannote: gated, requires HF license accept

---

<div align="center">

** Made with bf16 tensor cores in Taiwan **

If this toolkit saved you hours, **drop a star** — it helps more people find it.

[Report a bug](../../issues/new?template=bug_report.md) · [Request a feature](../../issues/new?template=feature_request.md) · [Discuss](../../discussions)

</div>
