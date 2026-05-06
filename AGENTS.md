# Guide for AI coding agents

You are working on the **Taiwan ASR Toolkit**, a Traditional-Chinese speech-to-text pipeline. This file gives you the contracts and idioms you must respect.

## The single non-negotiable contract

```python
# src/taiwan_asr/breeze.py
class FasterWhisperBackend:
    MODEL_ID = "MediaTek-Research/Breeze-ASR-25"   # never change this
class TransformersBackend:
    MODEL_ID = "MediaTek-Research/Breeze-ASR-25"   # never change this
```

The Breeze model is **the entire reason this project exists**. It's a Whisper-Large-v2 fine-tune
specifically for Taiwan Mandarin. Replacing it with `openai/whisper-large-v3-turbo`,
`distil-whisper`, `Systran/faster-whisper-*`, or any other Whisper variant — even if faster —
**defeats the toolkit's purpose**.

This is enforced by 5 tests:

```bash
pytest -m breeze_invariant -v   # MUST be 5 passed
```

Before you propose any optimization, run those tests. After you implement, run them again.

## Key idioms

### 1. The Breeze + Qwen3 symmetry rule

When changing pipeline plumbing (VAD parameters, batch logic, decoding params),
apply the same change to **both** `src/taiwan_asr/breeze.py` and `src/taiwan_asr/qwen3.py`. The toolkit's
benchmark legitimacy depends on identical pipelines.

If you can only apply a change to one (e.g. faster-whisper has `hotwords` but qwen-asr doesn't),
document it explicitly in the docstring and in `docs/ARCHITECTURE.md`.

### 2. Decoding-parameter floor (Mandarin-specific)

Never default to:
- `condition_on_previous_text=True` — Mandarin runaway repetition cascades. Always False.
- `temperature=0.0` only — without the fallback chain, hallucinations on hard chunks fail silently.
- `compression_ratio_threshold` higher than 2.4 or `log_prob_threshold` lower than -1.0 — the gating thresholds detect hallucination.

Test [test_breeze_decoding_defaults_safe_for_chinese](tests/test_invariants.py) reads source code
and ensures these are present.

### 3. Output is always Traditional Chinese (Taiwan)

Every text-producing path goes through `S2TW(True)` (OpenCC `s2twp` recipe). If you add a new
text source, apply `s2tw(text)` before saving.

### 4. Glossary respects proper nouns

`src/taiwan_asr/polish.py` MUST NOT correct entries in the glossary. The default glossary contains NTU dorm
names like `研三舍`, `延三舍`, `男一` etc. The user previously had `延三` ASR-mistranscribed,
the LLM "fixed" it to `延長`, which destroyed the meaning. Don't repeat that.

### 5. Language code mapping (Qwen3)

`qwen-asr` 0.0.6 only accepts full English names like `"Chinese"`, not BCP-47 codes like `"zh"`.
The mapping is in `Qwen3ASR.LANG_MAP`. If you hit a `ValueError: Unsupported language: ...`,
extend `LANG_MAP`, don't hard-code the BCP-47 code.

## How to add a feature (TDD)

```
1. Write a test that fails because the feature doesn't exist
2. Run pytest — confirm RED
3. Implement minimum to pass the test
4. Run pytest — confirm GREEN
5. Verify pytest -m breeze_invariant still passes
6. Update docs/CHANGELOG.md
```

Concrete pattern from this codebase: see `tests/test_glossary.py` (RED for missing
`load_glossary`) → `src/taiwan_asr/common.py` `load_glossary` impl → GREEN.

## Performance gotchas

- **GPU VAD is slower than CPU ONNX** for Silero — model too small. Use `device='cpu'` (the default).
- **`cudnn.benchmark=True`** is set in `init_torch()` — fine for length-sorted batches (limited unique shapes); could thrash if you process truly random shapes.
- **`PYTORCH_CUDA_ALLOC_CONF` deprecated** in PyTorch 2.9+. We set both that and `PYTORCH_ALLOC_CONF` in `init_env()`.
- **Conda ffmpeg lacks libsoxr** — `AudioIO.decode_to_array` falls back to `swr` automatically.
- **torchaudio ABI** must match torch (e.g., torch 2.9.1 → torchaudio 2.9.1; torchaudio 2.11 will SIGABRT).

## When you can't run code (model downloads, GPU absent)

- `pytest -m fast` runs in ~1 second with no GPU and no model download. Always run this first.
- `pytest -m breeze_invariant` is also fast and has no model dependency.
- Don't try to invoke `src/taiwan_asr/qwen3.py` or `src/taiwan_asr/breeze.py` end-to-end without GPU + ~5GB model cache.

## Commit conventions

This repo uses **clean Conventional Commits without AI signatures**. Do **NOT** add
`Co-Authored-By: Claude` or any AI attribution to commit messages. Commits are signed by
the human committer (`thc1006 <84045975+thc1006@users.noreply.github.com>`).

Examples of acceptable commit messages:

```
feat(breeze): inject glossary into hotwords + initial_prompt
fix(qwen3): map zh-TW to "Chinese" for qwen-asr 0.0.6 API
test: add 5 invariant tests locking Breeze model id
docs: explain why GPU VAD is slower than CPU ONNX for Silero
```

## When in doubt

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for design rationale. The "Why" sections
explain choices that will look strange without context (e.g., why CT2 over HF transformers,
why our own VAD instead of faster-whisper's, why batch=48 on Qwen3 but batch=32 on Breeze).
