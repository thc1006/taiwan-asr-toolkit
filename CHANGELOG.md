# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and
[Semantic Versioning](https://semver.org/).

## [0.5.4] – 2026-05-07

### Fixed
- **`asr-qwen3 --no-aligner` no longer crashes.** `_transcribe_batch`
  was hardcoding `return_time_stamps=True`, which the underlying
  `qwen_asr` library refuses when the ForcedAligner is not loaded
  (`ValueError: return_time_stamps=True requires forced_aligner …`).
  It now mirrors `aligner_enabled`, so `--no-aligner` saves the
  expected ~25 % time + ~600 MB download without the warmup or first
  real call exploding. Regression test in
  `tests/test_qwen3_no_aligner.py`.

### Changed (privacy)
- **Removed all voice fixtures from the repo.** `tests/fixtures/clip_30s.wav`
  and the matching ground-truth text are no longer tracked. The wheel never
  shipped them (only `taiwan_asr/data/*.txt` is in `package-data`), so PyPI
  installs are unaffected. Users now bring their own audio. Existing
  audio-dependent tests (`test_vad_gpu.py`, `test_glossary_effect.py`)
  `pytest.skip()` cleanly when no fixture is provided locally.
- **Colab quickstart notebook now opens a real file picker.** Cell 3
  detects Google Colab and uses `google.colab.files.upload()`; later cells
  parameterize by `audio_path` / `audio_stem`, so any uploaded audio works
  end-to-end. In local Jupyter the cell raises a friendly message asking
  you to set `audio_path` manually.
- `examples/compare_alternatives.md` removed. Its technical content
  (`condition_on_previous_text=False` rationale, OpenCC `s2twp` mapping,
  hot-word injection mechanics) lives in README.md and `docs/BENCHMARK.md`.

### Caveats
- v0.5.0 – v0.5.3 GitHub Releases were rebuilt at this version's tag SHAs
  to scrub the audio fixture from auto-generated source archives.
  PyPI artifacts at those versions remain unchanged (they were already
  voice-free).

## [0.5.3] – 2026-05-07

### Added
- The default NTU glossary now ships **inside the wheel** as
  `src/taiwan_asr/data/ntu_glossary.txt`. PyPI users (`pip install
  taiwan-asr-toolkit`) no longer need to download `glossary.txt` from
  GitHub separately to get hot-word injection.
- New magic value: `--glossary-file builtin` — resolves to the packaged
  default glossary regardless of CWD. Works for both pip-installed and
  cloned setups.
- New helper `taiwan_asr.common.builtin_glossary_path()` (also reachable
  via `from taiwan_asr.common import builtin_glossary_path`).
- 4 new tests in `tests/test_builtin_glossary.py` (test count 65 -> 69):
  resolution, builtin keyword load, case-insensitivity, and a sync guard
  that flags drift between root `glossary.txt` and the packaged copy.
- `[tool.setuptools.package-data]` now includes `data/*.txt`.

### Compat
- Root `glossary.txt` is kept for the existing clone-and-edit workflow.
  A test enforces it stays byte-identical to the packaged copy. Maintainer
  edits go in `src/taiwan_asr/data/ntu_glossary.txt`, then `cp` to root.
- Existing `--glossary-file glossary.txt` invocations from cloned repos
  continue to work unchanged.

## [0.5.2] – 2026-05-07

Quality / compatibility / docs polish patch in response to a self-review of
the v0.5.0 -> v0.5.1 diff. No user-visible behavior changes.

### Added
- `tests/test_blackwell_vram_tier.py`: 9 new tests (7 parametrized boundary
  tests + 1 explicit RTX 5090 baseline regression guard + 1 dtype guard)
  that lock the v0.5.1 Blackwell VRAM-tier batch table. Test count
  56 -> 65; all `@fast`, no GPU required (uses `monkeypatch` on
  `torch.cuda.*`).
- `examples/README.md`: navigation page for the examples/ directory with
  the Open-in-Colab badge and a short description of `quickstart.ipynb`.
- `examples/quickstart.ipynb`: dedicated markdown warning cell before the
  optional Qwen3 demo, calling out the ~5 GB additional model download
  (notebook now has 10 cells, was 9).
- CHANGELOG v0.5.1 retroactive `### Caveats` note: the >= 70 GB and
  >= 140 GB Blackwell tier batch sizes are extrapolated from the 32 GB
  RTX 5090 measurement, not field-tuned, with `--batch N` override
  guidance for empirical fine-tuners.

### Changed
- `pyproject.toml`: lower `torch>=2.7` to `torch>=2.5` (and `torchaudio` to
  match). The only post-2.5 API the toolkit calls
  (`torch.backends.cuda.enable_cudnn_sdp`) is already wrapped in try/except
  in `init_torch()`, so older torch falls back gracefully. Colab as of
  mid-2026 preinstalls torch 2.5-2.7, so the previous lower bound was
  forcing unnecessary upgrades.
- `.github/workflows/tests.yml`: dropped the redundant `pip install
  qwen-asr faster-whisper ctranslate2 silero-vad pydub` line; those come
  transitively via `pip install -e ".[eval,dev]"`.
- `docs/PROMOTION.md`: prepended a "public for transparency, not user
  docs" header making clear this is the maintainer's internal launch
  checklist, not a marketing playbook for users to follow.

### Fixed
- `docs/PROMOTION.md`: dropped a stray emoji that survived the v0.5.0
  emoji-cleanup commit.
- `examples/README.md`: removed brittle "9-cell" claim about
  `quickstart.ipynb` (cell count may drift; the description doesn't
  benefit from naming a number).

## [0.5.1] – 2026-05-07

### Added
- VRAM-tier-aware batch sizing on Blackwell (sm_120 / sm_100). Auto-detected
  per the actual VRAM the GPU reports:
    - >= 140 GB (B100 / B200): Qwen3 batch 128, Breeze batch 96
    - >= 70 GB  (RTX Pro 6000, 96 GB): Qwen3 batch 96, Breeze batch 64
    - >= 30 GB  (RTX 5090, 32 GB): Qwen3 batch 48, Breeze batch 32 (unchanged from 0.5.0)
    - smaller variants: Qwen3 batch 32, Breeze batch 16
  Smaller-GPU paths (Hopper, Ada, Ampere, Turing, Volta, CPU) unchanged.
- `examples/quickstart.ipynb` Colab notebook redesigned to be GPU-aware:
  full runtime table including RTX Pro 6000 and B100 / B200, footnote noting
  that single-30s-clip RTF is dominated by per-call overhead so the bigger
  GPUs only pull decisively ahead on multi-file batch jobs.
- `docs/PROMOTION.md`: internal launch playbook for PyPI, HN Show HN,
  Twitter, Reddit, HF model-page Discussions, and awesome-list submissions.

### Fixed
- `docs/BENCHMARK.md` stale `_asr_common.load_glossary()` reference now
  points at the post-refactor module path `taiwan_asr.common.load_glossary()`.

### Caveats
- The batch sizes for the >= 70 GB and >= 140 GB tiers (RTX Pro 6000, B100,
  B200) are **extrapolated** from the 32 GB RTX 5090 measurement, not
  field-tuned. They are conservative defaults that should fit comfortably
  in VRAM, but may not be the exact Pareto-optimal value. If you are using
  a Pro 6000 / B-series card and find a better batch size empirically,
  please open an issue with your numbers — or override at the CLI:
  ```bash
  asr-breeze --batch 80 audio.mp3   # override auto-detected batch=64 on Pro 6000
  asr-qwen3  --batch 80 audio.mp3
  ```

## [0.5.0] – 2026-05-07

Initial public release. Combines five iterations of internal optimization (v1–v5).

### Added — three independent feature axes
- **A. Hot-word injection at ASR source** (`asr-breeze --glossary-file`)
  - `load_glossary()` in `src/taiwan_asr/common.py`
  - Glossary terms feed Whisper's `initial_prompt` + faster-whisper `hotwords`
  - Demonstrably fixes `圓三 → 研三` (NTU graduate dorm) and `祝福二族 → 住輔二組` on real audio
- **B. Speaker diarization** (`src/taiwan_asr/diarize.py`)
  - pyannote.audio 4.x integration with `tensorlake/speaker-diarization-3.1` open mirror
  - `Segment.speaker_id` field added (backward-compatible default `None`)
  - `assign_speakers()` aligns ASR segments with diarization turns by max time-overlap
- **C. Real CER measurement** (`src/taiwan_asr/cer_eval.py`)
  - jiwer-based CER + WER with NFKC + punctuation removal + s2twp normalization
  - `asr-bench --gt-dir` integrates ground-truth comparison
  - Default ground-truth fixture for 標準錄音 886 first 55s

### Added — performance and quality
- Multi-file chunk pool batching for Qwen3 (`transcribe_files`, default-on for multi-file)
- `--no-aligner` flag for Qwen3 (~25% speedup, loses word-level timestamps)
- LLM context polish (`src/taiwan_asr/polish.py`) using Qwen3-8B with NTU glossary protection
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
| Breeze RTF (combined) | **382x** |
| Qwen3 RTF (combined) | **354x** |
| Breeze RTF (best single file, sparse audio) | **1554x** |
| Qwen3 RTF (best single file) | **1497x** |
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
