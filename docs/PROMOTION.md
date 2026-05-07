# Promotion playbook (manual steps)

> **Internal launch checklist — public for transparency, not part of the user-facing docs.**
> This is the maintainer's own to-do list for getting the toolkit in front of relevant audiences.
> If you are evaluating the toolkit, you do not need to read this — start with [`README.md`](../README.md)
> or [`examples/quickstart.ipynb`](../examples/quickstart.ipynb) instead.

Everything below is draft copy that the human maintainer reviews and posts. Nothing here is
automated, nothing pretends to be organic engagement, and the linked communities
(HN / Reddit / HF) all have rules against low-effort spam — so the drafts focus on technical
substance and concrete numbers rather than marketing fluff.

## 1. PyPI publish (one-time, ~10 min)

```bash
# Install build tooling
pip install build twine

# Build sdist + wheel
rm -rf dist/
python -m build

# Upload to TestPyPI first (verify everything looks right)
twine upload --repository testpypi dist/*
# Inspect: https://test.pypi.org/project/taiwan-asr-toolkit/

# Upload to real PyPI
twine upload dist/*
# Verify: https://pypi.org/project/taiwan-asr-toolkit/
```

You'll need a PyPI account + API token (https://pypi.org/manage/account/token/).
Save the token in `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcCJ...your-token...
```

After PyPI is live, update README install section to `pip install taiwan-asr-toolkit`.

## 2. HN post (Show HN)

**Best time**: Tuesday-Thursday, 8-10 AM US Eastern.

**Title** (must be ≤80 chars, factual, no exclamations):

```
Show HN: Taiwan Mandarin ASR with hot-word injection (RTF 1554x on RTX 5090)
```

**Body**:

```
Hi HN — I built taiwan-asr-toolkit because every general Mandarin ASR I tried
(Whisper-Large-v3, faster-whisper, whisperX) defaulted to Simplified Chinese
output and consistently mangled NTU dorm names like 研三舍 / 延三舍 to
圓三 / 圓山 — proper nouns nobody outside Taiwan would recognize.

The toolkit ships two production-grade Mandarin ASR engines (Qwen3-ASR-1.7B
and MediaTek's Breeze-ASR-25) on an identical pipeline so the benchmark
numbers actually compare the model rather than my plumbing. Glossary terms
are injected at decode time via Whisper's initial_prompt + faster-whisper's
hotwords parameter, fixing 研三 at the source. An optional Qwen3-8B context
polish stage uses the same glossary as a do-not-touch list to prevent the
LLM from "fixing" proper nouns it doesn't recognize.

Hardware target is RTX 5090 (Blackwell sm_120). Real numbers on 712 minutes
of mixed Taiwan-Mandarin lecture/interview audio:
- Breeze-ASR-25: RTF 382x average, 1554x on the longest sparse-audio file
- Qwen3-ASR-1.7B: RTF 354x average

The repo is fully MIT, every claim has a regression test (56 of them; 5 are
contract tests that lock the Breeze model id so optimizations can't
accidentally swap to a faster but generic Whisper variant). Open in Colab
runs the full pipeline on a bundled 30-second sample on the free T4.

Repo: https://github.com/thc1006/taiwan-asr-toolkit
Colab: https://colab.research.google.com/github/thc1006/taiwan-asr-toolkit/blob/main/examples/quickstart.ipynb
Benchmark deep dive: https://github.com/thc1006/taiwan-asr-toolkit/blob/main/docs/BENCHMARK.md

Happy to discuss design choices in comments — particularly the
condition_on_previous_text=False default for Mandarin and why CTranslate2
ends up faster than HF transformers on Whisper architecture in v3+.
```

## 3. Twitter / X thread

**Tweet 1** (hook):

```
Built taiwan-asr-toolkit: Traditional Chinese (Taiwan Mandarin) ASR that
finally gets 研三舍, 軟體, and 雷射 right.

Two SOTA models compared on identical pipeline. RTF 1554x on RTX 5090.
56 TDD tests including 5 that lock the Breeze model id.

(thread, 5 tweets)
github.com/thc1006/taiwan-asr-toolkit
```

**Tweet 2** (problem):

```
Stock Whisper / faster-whisper / whisperX pain points on Taiwan Mandarin:
- Output 簡體 by default (软件 instead of 軟體)
- Proper nouns die: 研三舍 → 圓三
- VAD silently fails on long sparse audio → 48-min hallucinated segment
- Variable VAD chunks waste 5-10x compute through padding

Toolkit fixes all four.
```

**Tweet 3** (mechanism):

```
Hot-word injection at the source:
glossary.txt → Whisper's initial_prompt + faster-whisper hotwords
fixes 研三 before the LLM polish ever sees it.

LLM polish uses the same glossary as do-not-touch list, so it won't "correct"
proper nouns it doesn't recognize.
```

**Tweet 4** (numbers):

```
Real numbers on 712 min of Taiwan Mandarin (lectures + interviews):

Breeze-ASR-25 (CTranslate2 bf16):
  RTF 382x average · 1554x on the easy file · 0 hallucinated segments

Qwen3-ASR-1.7B (HF + length-sorted batching):
  RTF 354x · 0 hallucinated segments

CER on hand-corrected 55s fixture: 2.34%
```

**Tweet 5** (try it):

```
Open in Colab badge runs the full pipeline on a bundled 30-second sample,
free T4 GPU, zero install on your machine:

colab.research.google.com/github/thc1006/taiwan-asr-toolkit/blob/main/examples/quickstart.ipynb

If you've ever debugged condition_on_previous_text=True repetition cascades on
Mandarin, drop a star.
```

## 4. Reddit (/r/MachineLearning, /r/LocalLLaMA, /r/taiwan)

For /r/MachineLearning — use the **[P] Project** flair, focus on technical
contribution:

```
[P] Taiwan ASR Toolkit: production-grade Traditional Chinese pipeline
    with hot-word injection (Qwen3-ASR + Breeze-ASR-25, RTF 1554x on RTX 5090)
```

Body: similar to HN body but more technical. Mention TDD + invariant tests.

For /r/LocalLLaMA — focus on the local-only angle and the Qwen3-8B polish:

```
Local Mandarin ASR + Qwen3-8B context polish — full pipeline, no API calls
```

For /r/taiwan / r/taiwanese — focus on the Taiwan-Mandarin angle:

```
我做了一個專門給台灣繁中語音轉文字的開源工具,自動避免「軟件 / 激光」這種大陸用詞
```

## 5. HuggingFace model page comments (polite cross-promotion)

For https://hf.co/Qwen/Qwen3-ASR-1.7B — open a Discussion (not an issue):

```
Title: Open-source toolkit using this model for Taiwan-Mandarin

Built taiwan-asr-toolkit (MIT) which integrates Qwen3-ASR-1.7B with hot-word
injection, OpenCC s2twp Traditional-Chinese normalization, and a Qwen3-8B
LLM polish stage. RTF 354x average on RTX 5090 across 712 min of Taiwan
Mandarin lectures.

Repo: https://github.com/thc1006/taiwan-asr-toolkit

Happy to share benchmark numbers or contribute docs/examples upstream if
useful.
```

For https://hf.co/MediaTek-Research/Breeze-ASR-25 — same template, swap
Qwen3 for Breeze and emphasize the CTranslate2 bf16 path (RTF 382x).

## 6. Awesome-list submissions (one-time)

Submit PRs adding the toolkit to:

- https://github.com/sindresorhus/awesome (under Speech)
- https://github.com/jdorfman/awesome-json-datasets (no, not relevant)
- https://github.com/keonl/awesome-nlp (Speech section)
- https://github.com/crownpku/Awesome-Chinese-NLP (very relevant, has Speech section)
- https://github.com/HqWei/Sound-Recognition-Tutorial (if exists for Chinese)

Format the entry as:

```
- [taiwan-asr-toolkit](https://github.com/thc1006/taiwan-asr-toolkit) -
  Production-grade Traditional Chinese / Taiwan Mandarin ASR with Qwen3-ASR
  and Breeze-ASR-25, hot-word injection, LLM context polish, speaker
  diarization. RTF up to 1554x on RTX 5090. MIT.
```

## 7. After the first 50 stars

- Add CONTRIBUTORS list
- Open 5-10 "good first issue" labeled tickets to spark community engagement
- Consider a v0.5.1 patch release with first round of community fixes
- Cross-link from related repos with PR (e.g., a "see also" entry in awesome-asr)

## 8. Cadence

- **Day 0**: PyPI publish + HN post + Twitter thread (within 1 hour of each
  other for compounding momentum)
- **Day 1-3**: Reddit cross-posts (one per day to avoid spam flags)
- **Day 7**: HF Discussion comments
- **Day 14**: Awesome-list PRs (lower priority but high long-tail value)
- **Day 30**: First v0.5.x patch release with feedback
