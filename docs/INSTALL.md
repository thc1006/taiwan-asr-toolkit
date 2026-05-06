# Installation Guide

This toolkit is tuned for **NVIDIA Blackwell (sm_120, RTX 5090)** but runs on
Hopper (H100), Ada (4090/L4), Ampere (A100/3090), and CPU (slow). Most CUDA
features auto-detect.

## Prerequisites

- **Python 3.10+** (3.11–3.13 tested)
- **NVIDIA GPU** with CUDA 12.6+ (CUDA 12.8 recommended for Blackwell)
- **FFmpeg 4.0+** (audio decoding)
- **HuggingFace account** (to download Qwen3-ASR + Breeze-ASR-25 models, free)
- ~30 GB disk for model cache + ~10 GB for first-time Breeze→CTranslate2 conversion

Verify your GPU:
```bash
nvidia-smi # should show driver and CUDA version
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Quick install (uv recommended)

[uv](https://github.com/astral-sh/uv) installs Python deps 10–100x faster than pip.

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repo
git clone https://github.com/thc1006/taiwan-asr-toolkit.git
cd taiwan-asr-toolkit

# Install PyTorch first (cu128 wheels for Blackwell; use cu124 for older GPUs)
uv pip install --system --index-url https://download.pytorch.org/whl/cu128 \
    torch torchaudio

# Install everything (core + eval + diarize)
uv pip install --system -e ".[all]"
```

## Plain pip install

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
pip install -e ".[all]"
```

## What gets installed

| Group | Packages | Disk |
|---|---|---|
| core | torch, transformers, qwen-asr, faster-whisper, ctranslate2, silero-vad, opencc-python-reimplemented | ~3 GB |
| eval | jiwer | ~5 MB |
| polish (auto) | (uses transformers, triggers Qwen3-8B download on first run, ~14 GB) | +14 GB |
| diarize | pyannote.audio | +500 MB |
| dev | pytest, ruff, mypy | ~50 MB |

## First run downloads

| Trigger | Model | Size | License |
|---|---|---|---|
| `asr-qwen3 …` | Qwen/Qwen3-ASR-1.7B + ForcedAligner-0.6B | ~5 GB | Apache-2.0 |
| `asr-breeze …` | MediaTek-Research/Breeze-ASR-25 + CT2 conversion | ~3 GB → ~2 GB | Apache-2.0 |
| `asr-polish …` | Qwen/Qwen3-8B (or Qwen2.5-7B fallback) | ~14 GB | Apache-2.0 |
| `asr-diarize …` | tensorlake/speaker-diarization-3.1 (community mirror) | ~600 MB | requires HF license accept |

## HuggingFace setup

```bash
# Configure HF token (one-time, for gated models like pyannote)
huggingface-cli login

# OR set env var
export HF_TOKEN=hf_...your_token_here...
```

For pyannote (diarization), accept license on each of:
- https://hf.co/pyannote/speaker-diarization-3.1
- https://hf.co/pyannote/speaker-diarization-community-1
- https://hf.co/pyannote/segmentation-3.0

## Known compatibility issues

### `torchaudio` ABI mismatch
If you see `OSError: undefined symbol: torch_library_impl`, your torchaudio
doesn't match torch. Force-reinstall the matching version:
```bash
pip install --force-reinstall --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 \
  torchaudio==<your_torch_version>
```

### conda ffmpeg lacks libsoxr
Symptom: `Requested resampling engine is unavailable`. The toolkit auto-falls
back to ffmpeg's built-in `swresample`, which is fine for 16 kHz speech ASR.
If you want libsoxr, install ffmpeg from your distro instead of conda:
```bash
sudo apt install ffmpeg     # Debian/Ubuntu
brew install ffmpeg         # macOS
```

## Verifying install

```bash
pytest -m fast              # 47 tests, ~1 second
asr-qwen3 --help
asr-breeze --help
```

If `pytest -m fast` is all green, the toolkit is correctly installed.
