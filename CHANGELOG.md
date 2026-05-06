# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and
[Semantic Versioning](https://semver.org/).

## [0.5.0] – 2026-05-07

Initial public release. Combines five iterations of internal optimization (v1–v5).

### Added — three independent feature axes
- **A. Hot-word injection at ASR source** (`breeze_asr.py --glossary-file`)
  - `load_glossary()` in `_asr_common.py`
  - Glossary terms feed Whisper's `initial_prompt` + faster-whisper `hotwords`
  - Demonstrably fixes `圓三 → 研三` (NTU graduate dorm) and `祝福二族 → 住輔二組` on real audio
- **B. Speaker diarization** (`diarize.py`)
  - pyannote.audio 4.x integration with `tensorlake/speaker-diarization-3.1` open mirror
  - `Segment.speaker_id` field added (backward-compatible default `None`)
  - `assign_speakers()` aligns ASR segments with diarization turns by max time-overlap
- **C. Real CER measurement** (`cer_eval.py`)
  - jiwer-based CER + WER with NFKC + punctuation removal + s2twp normalization
  - `benchmark.py --gt-dir` integrates ground-truth comparison
  - Default ground-truth fixture for 標準錄音 886 first 55s

### Added — performance and quality
- Multi-file chunk pool batching for Qwen3 (`transcribe_files`, default-on for multi-file)
- `--no-aligner` flag for Qwen3 (~25% speedup, loses word-level timestamps)
- LLM context polish (`polish.py`) using Qwen3-8B with NTU glossary protection
- Length-sorted batching across files
- Manual ONNX Silero VAD on Breeze side via `clip_timestamps` (replaces faster-whisper internal VAD)
- Breeze + Qwen3 symmetric pipeline (same VAD, same dtype, same batch policy) for fair comparison
- `--internal-vad` flag for Breeze fallback

### Added — testing
- 56 pytest tests (47 fast + 4 medium + 5 breeze-invariant + 0 slow)
- Breeze invariant suite locks `MODEL_ID == "MediaTek-Research/Breeze-ASR-25"` so optimizations can't accidentally swap models
- v3 baseline regression tests that lock segment counts, no >60s segments, Traditional ratio

### Added — documentation & infra
- `README.md` with SEO-optimized hero, benchmarks, comparison table
- `docs/ARCHITECTURE.md` with pipeline diagram and design rationale
- `docs/INSTALL.md` with uv-recommended install + Blackwell tuning
- `docs/BENCHMARK.md` with full v2→v3→v4→v5 evolution
- `llms.txt` (https://llmstxt.org standard) for AI agent retrieval
- `AGENTS.md` with contracts for AI coding assistants
- `CONTRIBUTING.md` with TDD workflow
- `pyproject.toml` (modern, replaces requirements.txt + pytest.ini)
- `.gitignore` excludes audio (`music/`, `transcripts/`) and caches
- GitHub Actions CI for fast tests
- Issue templates and PR template

### Performance milestones (RTX 5090 + i9-14900, 712 min audio total)

| Metric | Result |
|---|---:|
| Breeze RTF (combined) | **382×** |
| Qwen3 RTF (combined) | **354×** |
| Breeze RTF (best single file, sparse audio) | **1554×** |
| Qwen3 RTF (best single file) | **1497×** |
| Catastrophic >60s hallucinated segments | **0** (both models, all 11 files) |
| Cross-model character Jaccard agreement | 0.76 |
| Quality score average (Q3 / Br) | 0.815 / 0.808 |

### Known issues / deferred
- Pyannote diarization requires user to manually accept HF licenses on three model pages (one-time, free)
- Real CER GT is approximate (Breeze-derived); user should hand-transcribe a 60s clip for unbiased CER
- flash-attn 3 source build not yet integrated (potential +10–20% on Qwen3 path)
- TensorRT-LLM Whisper engine not yet evaluated (potential +80–100% on Breeze path)

### Hardware/software baseline
- NVIDIA RTX 5090 (Blackwell sm_120, 32 GB GDDR7)
- Intel Core i9-14900 (24 threads)
- CUDA 12.8, cuDNN 9.10, PyTorch 2.9.1+cu128
- Python 3.13, Ubuntu 24.04
