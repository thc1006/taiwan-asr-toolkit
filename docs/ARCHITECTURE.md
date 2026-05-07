# Architecture

## High-level pipeline

```
                ┌────────────────────────────────────────┐
input audio ──▶ │ ffmpeg pipe → numpy float32 16k mono   │ (AudioIO.decode_to_array)
(.mp3/.m4a/.    └────────────────┬───────────────────────┘
 wav/.mp4/...)                   │
                                 ▼
                ┌────────────────────────────────────────┐
                │ Silero VAD (ONNX CPU SIMD)             │ ≤28s chunks, 0.4s pad
                │ → list[{audio, start, end}]            │
                └────────────────┬───────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                                     ▼
    ┌──────────────────────┐              ┌──────────────────────────┐
    │ Qwen3-ASR-1.7B       │              │ Breeze-ASR-25            │
    │ + ForcedAligner-0.6B │              │ (Whisper-Large-v2 FT)    │
    │ HF transformers      │              │ via faster-whisper /     │
    │ bf16, sdpa, batch=48 │              │ CTranslate2 bf16,       │
    │ length-sorted        │              │ batch=32, beam=5         │
    │ + multi-file pool    │              │ + clip_timestamps        │
    │ + warmup             │              │ + initial_prompt/hotword │
    └─────────┬────────────┘              └─────────┬────────────────┘
              │                                     │
              ▼                                     ▼
    ┌──────────────────────────────────────────────────────────┐
    │ OpenCC s2twp 簡→繁(台灣慣用詞)                              │
    └────────────────────┬─────────────────────────────────────┘
                         │
                ┌────────┼─────────────┐
                ▼        ▼             ▼
            ┌─────┐ ┌──────────┐ ┌──────────────┐
            │ TXT │ │   SRT    │ │ JSON (full) │
            └─────┘ └──────────┘ └──────────────┘

       Optional post-processing
           ─────────────────────
   ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
   │ asr-polish         │    │ asr-diarize        │    │ asr-bench          │
   │ Qwen3-8B context   │    │ pyannote 3.x       │    │ CER vs ground truth│
   │ correction         │    │ → speaker_id       │    │ + cross-model agree│
   │ (glossary-aware)   │    │                    │    │                    │
   └────────────────────┘    └────────────────────┘    └────────────────────┘
```

## Module responsibilities

| Module | Purpose |
|---|---|
| `src/taiwan_asr/common.py` | Env init, ffmpeg pipe IO, OpenCC s2twp, Silero VAD, length-sorted batching, Stopwatch, glossary loader, `Segment` dataclass |
| `src/taiwan_asr/qwen3.py`   | Qwen3-ASR-1.7B + ForcedAligner-0.6B; multi-file chunk pool; `--no-aligner` |
| `src/taiwan_asr/breeze.py` | Breeze-ASR-25 (Whisper-Large-v2 FT); manual VAD + clip_timestamps; `--glossary-file` for hot-word injection; `--fast` for int8 quant |
| `src/taiwan_asr/polish.py`      | Qwen3-8B LLM context-correction with NTU glossary protection (won't touch proper nouns) |
| `src/taiwan_asr/diarize.py`     | pyannote.audio speaker diarization; assigns `speaker_id` to each ASR segment by max time-overlap |
| `src/taiwan_asr/cer_eval.py`    | Character Error Rate / WER, using jiwer + s2twp normalization for fair simplified-vs-traditional comparison |
| `src/taiwan_asr/benchmark.py`   | Full speed + accuracy report (RTF, coverage, hallucination signals, cross-model agreement, optional CER) |

## Why these design choices

### 1. Same VAD on both sides (fair benchmark)

Earlier versions had Qwen3 use our Silero VAD while Breeze used faster-whisper's internal VAD.
Different VAD policies → different chunking → biased comparison. v3+ enforces the **same Silero ONNX
chunker** on both sides so any speed/quality difference is the model itself, not pipeline plumbing.

### 2. `condition_on_previous_text=False` (Breeze)

Whisper's default `condition_on_previous_text=True` causes runaway repetition cascades on Mandarin
far worse than English. This is the **single most important** Chinese-ASR setting.

### 3. Length-sorted batching (Qwen3)

VAD produces highly variable chunk lengths (0.4s–28s). Naive batching pads to the longest in batch
→ 5–10x compute waste. Sorting chunks by length descending and grouping similar-length together
brings padding waste below 2x. Free 1.5–3x speedup, zero accuracy cost.

### 4. Multi-file chunk pool (Qwen3 v4)

When transcribing a folder of mixed-length files, per-file batches under-fill on short files
(an 8-min file with 21 chunks doesn't fill batch=48). Pooling chunks across all input files into
one length-sorted stream keeps every batch fully utilized.

### 5. Hot-word at the source (Breeze v5)

ASR errors on proper nouns like 「研三舍」(NTU graduate dorm) cascade — once mis-transcribed
to 「圓三」, downstream LLM polish has no signal to recover. Feeding the glossary to Whisper's
`initial_prompt` + `hotwords` boosts the right tokens at decode time, **fixing errors at the source**.
The polish step then handles only the remaining rare cases.

### 6. CT2 over HF transformers (Breeze)

CTranslate2's compiled kernels outperform HF's eager-mode bf16 on Whisper architecture by ~30%.
This is why Breeze ends up faster than Qwen3 in v3+ benchmarks despite Qwen3 being a smaller model.

## Test architecture (TDD)

```
109 tests · 4 priority tiers
─────────────────────────────
@breeze_invariant    (5) ← never-fail; protects MediaTek-Research/Breeze-ASR-25 model id
@fast              (104) ← no model load, ~9s (includes the 5 invariants)
@medium              (4) ← VAD only
@slow                (1) ← model-loading e2e (intentionally not in default suite)
```

Run subsets:
```bash
pytest -m breeze_invariant     # contract tests
pytest -m fast                 # quick check during dev
pytest                         # full default
```
