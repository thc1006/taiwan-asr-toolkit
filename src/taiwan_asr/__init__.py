"""Taiwan ASR Toolkit — production-grade Traditional Chinese / Taiwan Mandarin
speech-to-text built on Qwen3-ASR-1.7B and MediaTek Breeze-ASR-25.

Public API is exposed lazily — heavy modules (torch / transformers /
faster_whisper) are imported only when their classes are accessed.
"""
__version__ = "0.5.0"
__all__ = ["__version__"]
