# Examples

Tutorials and side-by-side evidence for the toolkit. New users start here.

## Files

| File | What it is | When to read |
|---|---|---|
| [`quickstart.ipynb`](quickstart.ipynb) | Colab-ready Jupyter notebook. Detects your GPU, downloads a 30-second Taiwan-Mandarin sample, runs Breeze-ASR-25 with hot-word injection, prints the transcript. | First time using the toolkit, or want a no-install Colab demo. |
| [`compare_alternatives.md`](compare_alternatives.md) | Real character-level comparison vs. stock Whisper / faster-whisper / whisperX on a known-hard NTU proper noun (`研三舍`). Shows what hot-word injection actually fixes. | After running the notebook, when you want to know "is this real or marketing?". |

## Open quickstart in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thc1006/taiwan-asr-toolkit/blob/main/examples/quickstart.ipynb)

The notebook auto-detects your GPU runtime (T4 / L4 / A100 / RTX Pro 6000 / B100 / B200 / CPU)
and the toolkit picks the optimal dtype + batch size. No manual tuning required.

## Want to add an example?

PRs welcome. Please follow the existing format:

- A self-contained notebook or markdown file
- No external dependencies beyond what `pip install -e ".[all]"` provides (or document the extras)
- If your example produces transcripts, output should be Traditional Chinese (the toolkit defaults guarantee this)
- See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for the TDD + commit conventions
