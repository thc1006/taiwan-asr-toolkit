# Contributing to Taiwan ASR Toolkit

Thanks for your interest! This project follows strict TDD and a small set of
**non-negotiable contracts**. Read these before opening a PR.

## Non-negotiable contracts

### 1. Breeze model is `MediaTek-Research/Breeze-ASR-25`

The Breeze-ASR-25 model is the entire reason this toolkit exists. It's a
Whisper-Large-v2 fine-tune specifically for Taiwan Mandarin. **Replacing it with
any other Whisper variant — even if faster — defeats the purpose.**

This is enforced by [`tests/test_invariants.py`](tests/test_invariants.py)
under `@pytest.mark.breeze_invariant`. Any PR that breaks these tests will be
rejected.

```bash
pytest -m breeze_invariant -v   # MUST be 5 passed
```

### 2. Output is always Traditional Chinese (Taiwan)

OpenCC `s2twp` post-processing is mandatory on all text-producing paths. Tests
verify the existing `886_breeze.json` output is majority-Traditional.

### 3. Mandarin decoding floor

These Whisper params are battle-tested for Mandarin:
- `condition_on_previous_text=False` (CRITICAL — True causes runaway repetition)
- `compression_ratio_threshold=2.4`
- `log_prob_threshold=-1.0`
- `no_speech_threshold=0.6`
- `temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` fallback chain

Don't weaken these in defaults. New flags can opt-in to alternatives.

## Development workflow (TDD)

```
1. Open an issue describing the change
2. Write a failing test that exercises the new behavior
3. Run pytest — confirm RED
4. Implement minimum to pass the test
5. Run pytest — confirm GREEN
6. Verify pytest -m breeze_invariant still passes
7. Update docs/CHANGELOG.md and relevant docs/
8. Submit PR using the template
```

Concrete example: see how `load_glossary` was added.
[`tests/test_glossary.py`](tests/test_glossary.py) was written first (RED),
then `_asr_common.py::load_glossary` (GREEN).

## Setup for development

```bash
git clone https://github.com/thc1006/taiwan-asr-toolkit.git
cd taiwan-asr-toolkit

# uv recommended (10-100× faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install --system --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
uv pip install --system -e ".[dev,all]"

# Verify
pytest -m fast
```

## Code style

- **Formatter / linter**: `ruff` (config in `pyproject.toml`)
- **Line length**: 100 chars (long lines OK in tests)
- **Type hints**: encouraged, especially on public APIs
- **Docstrings**: 1-3 lines, in the language matching the surrounding code (English for code,
  mixed Chinese/English in user-facing prints is fine)

## Commits

This repo uses **clean Conventional Commits**. Format:

```
type(scope): subject

optional body
```

Types: `feat`, `fix`, `perf`, `test`, `docs`, `refactor`, `chore`, `ci`.

**Do NOT add `Co-Authored-By: <AI>` or any AI signature** to commit messages.
Commits are signed by the human committer.

Examples:
```
feat(breeze): inject glossary into hotwords + initial_prompt
fix(qwen3): map zh-TW to "Chinese" for qwen-asr 0.0.6 API
perf(qwen3): cross-file chunk pool batching saves 4.4% on 11-file workload
test: lock 圓三 → 研三 hot-word fix as regression
docs(arch): explain why GPU VAD is slower than CPU ONNX for Silero
```

## Pull request

Use the PR template (auto-loads from `.github/PULL_REQUEST_TEMPLATE.md`). Required:

- [ ] New behavior has a failing test that now passes
- [ ] All 56 existing tests pass
- [ ] `pytest -m breeze_invariant` green
- [ ] Performance numbers if applicable (RTF, CER)
- [ ] No AI signatures in commits

## Reporting bugs

Use the bug report template. Critical info:
- `pytest -m fast` output
- `pytest -m breeze_invariant` output
- GPU + driver via `nvidia-smi`
- PyTorch version + CUDA version

## Performance contributions

If you're contributing speed optimizations, include before/after numbers:

```
$ python benchmark.py --timing /tmp/timing.tsv
```

Performance claims must include the test set used. The reference test set is
the 11 files in `music/` (excluded from the repo for size; numbers in
[`docs/BENCHMARK.md`](docs/BENCHMARK.md)).

## License

By contributing, you agree your contributions are licensed under MIT (see [LICENSE](LICENSE)).
