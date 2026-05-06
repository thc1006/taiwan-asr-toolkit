# -*- coding: utf-8 -*-
"""
Breeze-ASR-25 (台灣華語) 極致優化版 v2 — RTX 5090 (Blackwell sm_120)
模型: MediaTek-Research/Breeze-ASR-25 (Whisper-Large-v2 fine-tune,專攻台灣繁中)

策略 = 「準度為先,免費的速度全榨」:
  默認 bfloat16 (準度等同 FP32,Blackwell TC 原生)
  默認 beam_size=5 (Pareto 最佳)
  默認 condition_on_previous_text=False (中文重複幻覺剋星)
  默認 OpenCC s2twp 後處理 (簡→繁台灣化, 軟件→軟體)
  默認 BatchedInferencePipeline batch=24 (5090 32GB 餵得飽)

「免費」加速 (準度 0 損失):
   ONNX Silero VAD (faster-whisper 內建即用,3-5x VAD 速度)
   ffmpeg pipe 直讀 f32le (skip 暫存 wav)
   24 核 CPU 全開 (OMP/MKL/cpu_threads)

額外旗標 (使用者自選 trade-off):
  --fast              compute_type=int8_bfloat16 (1.5-1.7x 速度,中文 CER +0.3-0.5%)
  --beam N            預設 5,可調 1 (greedy,最快) ~ 10 (極致準度)
  --backend xxx       auto / faster-whisper / transformers
  --no-s2tw           關 OpenCC 後處理
"""
from __future__ import annotations

# (一) 必須最早!
from taiwan_asr.common import init_env
_NCPU = init_env()

import os, sys, gc, time, argparse, warnings, subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

import numpy as np
import soundfile as sf
import torch
from taiwan_asr.common import (
    init_torch, Segment, save_outputs, S2TW, AudioIO, SileroVAD, Stopwatch,
    load_glossary,
)
init_torch(_NCPU)


# ============================================================
# 5090 配置
# ============================================================
@dataclass
class HwConfig:
    device: str = "cuda:0"
    cuda_idx: int = 0
    dtype: torch.dtype = torch.bfloat16
    ct2_compute: str = "bfloat16"     # 準度模式
    batch_size: int = 32              # 5090 32GB,Whisper-L-v2 ~3GB,batch=32 安全
    cpu_threads: int = 0
    num_workers: int = 4              # CT2 多 worker 加速 IO/解碼
    chunk_sec: float = 28.0           # 與 Qwen3 對等
    pad_sec: float = 0.4
    desc: str = ""


def detect_hw(fast: bool) -> HwConfig:
    cfg = HwConfig()
    cfg.cpu_threads = _NCPU
    if not torch.cuda.is_available():
        cfg.device = "cpu"
        cfg.dtype = torch.float32
        cfg.ct2_compute = "int8" if fast else "float32"
        cfg.batch_size = 1
        cfg.desc = "CPU"
        return cfg
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cfg.desc = f"{name} ({vram:.0f}GB, cc={cap[0]}.{cap[1]})"

    # Blackwell sm_120 / sm_100 — covers RTX 5090 (32GB), RTX Pro 6000 (96GB), B100/B200 (192GB)
    if cap[0] >= 12:
        cfg.dtype = torch.bfloat16
        cfg.ct2_compute = "int8_bfloat16" if fast else "bfloat16"
        if vram >= 140:                # B100/B200 (192GB)
            cfg.batch_size = 96
        elif vram >= 70:               # RTX Pro 6000 (96GB)
            cfg.batch_size = 64
        elif vram >= 30:               # RTX 5090 (32GB)
            cfg.batch_size = 32
        else:                           # smaller Blackwell (if any)
            cfg.batch_size = 16
    elif cap[0] == 9:                  # Hopper
        cfg.dtype = torch.bfloat16
        cfg.ct2_compute = "int8_bfloat16" if fast else "bfloat16"
        cfg.batch_size = 48 if vram >= 70 else 32
    elif cap[0] == 8 and cap[1] == 9:  # Ada
        cfg.dtype = torch.bfloat16
        cfg.ct2_compute = "int8_bfloat16" if fast else "bfloat16"
        cfg.batch_size = 12 if vram >= 22 else 6
    elif cap[0] == 8:                  # Ampere
        bf = torch.cuda.is_bf16_supported()
        cfg.dtype = torch.bfloat16 if bf else torch.float16
        cfg.ct2_compute = ("int8_bfloat16" if (fast and bf) else
                           "int8_float16" if fast else
                           "bfloat16" if bf else "float16")
        cfg.batch_size = 8 if vram >= 38 else 4
    else:                              # Turing/Volta
        cfg.dtype = torch.float16
        cfg.ct2_compute = "int8_float16" if fast else "float16"
        cfg.batch_size = 4
    return cfg


# ============================================================
# 後端 1: faster-whisper (CTranslate2)
# ============================================================
class FasterWhisperBackend:
    MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
    NAME = "faster-whisper (CTranslate2)"

    def __init__(self, cfg: HwConfig, beam: int, s2tw: S2TW,
                 use_manual_vad: bool = True,
                 glossary: Optional[List[str]] = None):
        self.cfg = cfg
        self.beam = beam
        self.s2tw = s2tw
        self.use_manual_vad = use_manual_vad
        self.glossary: List[str] = list(glossary) if glossary else []
        self.model = None
        self.batched = None
        # 與 Qwen3 共用同一支 ONNX Silero — 管線對等
        self.vad = SileroVAD(
            sr=16000, max_sec=cfg.chunk_sec,
            pad_sec=cfg.pad_sec, min_sec=0.4,
            threshold=0.4,
        )
        # 模型存到家目錄,bfloat16 為基底,load 時可降級到 int8_bfloat16
        self.ct2_dir = os.path.expanduser("~/.cache/breeze-asr-ct2-bf16")

    def _build_prompt_and_hotwords(self) -> tuple:
        """從 glossary 組出 (initial_prompt, hotwords)。
        initial_prompt: Whisper 看過的 context,讓模型先認識這些詞。
        hotwords: faster-whisper >=1.0 的 boost 機制。"""
        base = "以下是普通話的句子,請用繁體中文輸出。"
        if not self.glossary:
            return base, None
        # 限制長度避免 prompt 吃掉 token budget;前 30 個是預設的 NTU 宿舍詞
        terms = self.glossary[:30]
        prompt = base + " 常見專有名詞:" + "、".join(terms) + "。"
        hot = " ".join(terms)
        return prompt, hot

    def _ensure_ct2(self):
        marker = os.path.join(self.ct2_dir, "model.bin")
        if os.path.isfile(marker):
            return
        print(f" 首次使用,將 Breeze-ASR-25 轉 CTranslate2 (基底 bfloat16,可 load 時降級)…")
        os.makedirs(self.ct2_dir, exist_ok=True)
        cmd = [
            "ct2-transformers-converter",
            "--model", self.MODEL_ID,
            "--output_dir", self.ct2_dir,
            "--quantization", "bfloat16",   # 基底用 bf16,load 時可動態降到 int8_bf16
            "--copy_files",
            "tokenizer.json", "preprocessor_config.json",
            "generation_config.json", "tokenizer_config.json",
            "vocab.json", "merges.txt", "normalizer.json",
            "special_tokens_map.json", "added_tokens.json",
            "--force",
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"CT2 轉換失敗,請改用 --backend transformers: {e}")

    def load(self) -> "FasterWhisperBackend":
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        self._ensure_ct2()
        print(f"⏳ 載入 faster-whisper (compute={self.cfg.ct2_compute},beam={self.beam})…")
        t0 = time.time()
        self.model = WhisperModel(
            self.ct2_dir,
            device="cuda" if self.cfg.device.startswith("cuda") else "cpu",
            device_index=self.cfg.cuda_idx,
            compute_type=self.cfg.ct2_compute,
            cpu_threads=self.cfg.cpu_threads,
            num_workers=self.cfg.num_workers,
        )
        try:
            self.batched = BatchedInferencePipeline(model=self.model)
            print(f" faster-whisper + BatchedInferencePipeline 就緒 ({time.time()-t0:.1f}s)")
        except Exception:
            self.batched = None
            print(f" faster-whisper 就緒 (no batched pipeline) ({time.time()-t0:.1f}s)")
        print(f" OpenCC: {self.s2tw}  |  manual_vad={self.use_manual_vad}")

        # 暖機:跑 1 秒 dummy 讓 cuDNN/CT2 把 kernel JIT 完
        try:
            t1 = time.time()
            dummy = np.zeros(16000, dtype=np.float32)  # 1s silence
            iter_w, _ = (self.batched or self.model).transcribe(
                dummy, language="zh", beam_size=1,
                vad_filter=False, word_timestamps=False,
            )
            for _ in iter_w: pass
            print(f" warmup 完成 ({time.time()-t1:.2f}s)")
        except Exception as e:
            print(f" (warmup 略過: {type(e).__name__})")
        return self

    def transcribe(self, src_file: str, language: str = "zh") -> tuple:
        sw = Stopwatch()
        # 用我們自己的 ffmpeg pipe 解碼到 numpy,避免 faster-whisper 內建可能用不同解碼
        audio, dur = AudioIO.decode_to_array(src_file)
        sw.lap("audio decode (ffmpeg→f32 pipe)")

        # Whisper 全域解碼參數 — 中文 ASR 業界共識
        prompt, hotwords = self._build_prompt_and_hotwords()
        common = dict(
            language=language,
            task="transcribe",
            beam_size=self.beam,
            best_of=self.beam,
            patience=1.0,
            length_penalty=1.0,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,        # 中文必須 False
            word_timestamps=True,
            initial_prompt=prompt,
        )
        if hotwords:
            common["hotwords"] = hotwords

        #  與 Qwen3 對等 pipeline:用我們自己的 ONNX Silero VAD,
        #   把每段語音時間戳餵給 faster-whisper,
        #   完全跳過內建 VAD (那是 11 個 >60s 失控段的元兇)。
        # API 差異:BatchedInferencePipeline 用 list-of-dict;非 batched 用 flat list。
        if self.use_manual_vad:
            chunks = self.vad.speech_chunks(audio)
            sw.lap(f"VAD ({self.vad._using})")
            n_chunks = len(chunks)
            print(f" 音訊: {dur:.1f}s | 我方 VAD: {n_chunks} 段 (每段 ≤{self.cfg.chunk_sec}s) | "
                  f"batch={self.cfg.batch_size} | beam={self.beam}")
            common["vad_filter"] = False
            if n_chunks == 0:
                common["clip_timestamps"] = "0"
            elif self.batched is not None:
                # BatchedInferencePipeline 吃 list-of-dict
                common["clip_timestamps"] = [
                    {"start": float(c["start"]), "end": float(c["end"])}
                    for c in chunks
                ]
            else:
                # 非 batched 吃 flat list of floats
                flat: List[float] = []
                for c in chunks:
                    flat.extend([float(c["start"]), float(c["end"])])
                common["clip_timestamps"] = flat
        else:
            common["vad_filter"] = True
            common["vad_parameters"] = dict(
                threshold=0.4,
                min_silence_duration_ms=250,
                speech_pad_ms=400,
                max_speech_duration_s=self.cfg.chunk_sec,  # 強制 ≤28s
            )
            print(f" 音訊: {dur:.1f}s | fw 內建 VAD (max={self.cfg.chunk_sec}s) | "
                  f"batch={self.cfg.batch_size} | beam={self.beam}")

        t0 = time.time()
        if self.batched is not None:
            it, info = self.batched.transcribe(
                audio, batch_size=self.cfg.batch_size, **common
            )
        else:
            it, info = self.model.transcribe(audio, **common)

        out: List[Segment] = []
        last_print = t0
        for s in it:
            txt = self.s2tw(s.text.strip())
            if not txt:
                continue
            words = None
            if getattr(s, "words", None):
                words = [
                    {"start": w.start, "end": w.end,
                     "word": self.s2tw(w.word), "prob": float(getattr(w, "probability", 0.0))}
                    for w in s.words if w.word and w.word.strip()
                ]
            out.append(Segment(
                start=s.start, end=s.end, text=txt,
                language=info.language,
                avg_logprob=float(getattr(s, "avg_logprob", 0.0)),
                no_speech_prob=float(getattr(s, "no_speech_prob", 0.0)),
                words=words,
            ))
            now = time.time()
            if now - last_print >= 1.0 and dur > 0:
                pct = min(100.0, 100 * s.end / dur)
                el = now - t0
                rtf = s.end / max(el, 1e-3)
                print(f"\r   ▶ {pct:5.1f}% | {el:6.1f}s | RTF≈{rtf:5.1f}x",
                      end="", flush=True)
                last_print = now
        print()
        sw.lap("ASR (faster-whisper + s2twp)")
        print(f" Breeze 完成 | 段 {len(out)} | 耗 {time.time()-t0:.1f}s | "
              f"音訊 {dur:.1f}s | RTF={dur/max(time.time()-t0,1e-3):.1f}x | "
              f"lang {info.language} (信心 {info.language_probability:.2f})")
        return out, sw


# ============================================================
# 後端 2: HF transformers (fallback)
# ============================================================
class TransformersBackend:
    MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
    NAME = "transformers (HF + torch.compile)"

    def __init__(self, cfg: HwConfig, beam: int, s2tw: S2TW,
                 compile_mode: Optional[str] = "reduce-overhead"):
        self.cfg = cfg
        self.beam = beam
        self.s2tw = s2tw
        self.compile_mode = compile_mode
        self.pipe = None

    def load(self) -> "TransformersBackend":
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        print("⏳ 載入 Breeze-ASR-25 (transformers, bf16, sdpa)…")
        t0 = time.time()
        proc = AutoProcessor.from_pretrained(self.MODEL_ID)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.MODEL_ID, dtype=self.cfg.dtype,
            low_cpu_mem_usage=True, attn_implementation="sdpa",
        )
        model.to(self.cfg.device)

        if self.compile_mode and self.cfg.device.startswith("cuda"):
            try:
                model.generation_config.cache_implementation = "static"
                model.generation_config.max_new_tokens = 440
                model.forward = torch.compile(
                    model.forward, mode=self.compile_mode,
                    fullgraph=True, dynamic=False,
                )
                print(f" torch.compile 已啟用 (mode={self.compile_mode})")
            except Exception as e:
                print(f" ℹ  torch.compile 略過: {type(e).__name__}: {str(e)[:100]}")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model, tokenizer=proc.tokenizer,
            feature_extractor=proc.feature_extractor,
            torch_dtype=self.cfg.dtype, device=self.cfg.device,
            chunk_length_s=30, stride_length_s=(6, 2),
            return_timestamps=True,
        )
        print(f" transformers 就緒 ({time.time()-t0:.1f}s)")
        print(f" OpenCC: {self.s2tw}")
        return self

    @torch.inference_mode()
    def transcribe(self, src_file: str, language: str = "zh"):
        sw = Stopwatch()
        # 直接讓 pipeline 吃檔案 (內部會用 librosa/soundfile 重採樣)
        gen_kwargs = {
            "task": "transcribe",
            "language": language,
            "num_beams": self.beam,
            "max_new_tokens": 440,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_prev_tokens": False,
        }
        try:
            dur = sf.info(src_file).duration
        except Exception:
            dur = 0.0
        print(f" 推論中 (batch={self.cfg.batch_size}, beam={self.beam})…")
        t0 = time.time()
        result = self.pipe(src_file, batch_size=self.cfg.batch_size, generate_kwargs=gen_kwargs)
        el = time.time() - t0
        sw.lap("ASR (HF pipeline)")

        out: List[Segment] = []
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        for c in chunks:
            txt = self.s2tw((c.get("text") or "").strip())
            if not txt:
                continue
            ts = c.get("timestamp")
            if isinstance(ts, (tuple, list)) and len(ts) >= 2:
                s_t = ts[0] if ts[0] is not None else 0.0
                e_t = ts[1] if ts[1] is not None else (s_t + 1.0)
            else:
                s_t = e_t = 0.0
            out.append(Segment(start=s_t, end=e_t, text=txt))
        if not out and isinstance(result, dict) and result.get("text"):
            out.append(Segment(0.0, dur, self.s2tw(result["text"].strip())))
        sw.lap("post (s2twp)")
        print(f" Breeze 完成 | 段 {len(out)} | 耗 {el:.1f}s | "
              f"音訊 {dur:.1f}s | RTF={dur/max(el,1e-3):.1f}x")
        return out, sw


# ============================================================
# 後端選擇
# ============================================================
def select_backend(cfg, beam, s2tw, requested: str,
                   use_manual_vad: bool = True,
                   glossary: Optional[List[str]] = None):
    if requested in ("faster-whisper", "fw", "auto"):
        try:
            import faster_whisper, ctranslate2  # noqa
            return FasterWhisperBackend(cfg, beam, s2tw,
                                        use_manual_vad=use_manual_vad,
                                        glossary=glossary)
        except ImportError:
            if requested != "auto":
                raise SystemExit(" 未安裝 faster-whisper / ctranslate2")
            print("ℹ  faster-whisper 不可用,改用 transformers backend")
    return TransformersBackend(cfg, beam, s2tw)


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Breeze-ASR-25 5090 v2 (準度+速度兼顧)")
    ap.add_argument("inputs", nargs="+", help="音訊檔 (mp3/wav/m4a/mp4/flac…)")
    ap.add_argument("--backend", choices=["auto", "faster-whisper", "transformers"], default="auto")
    ap.add_argument("--lang", default="zh", help="預設 zh")
    ap.add_argument("--out", default="./transcripts/breeze", help="輸出目錄")
    ap.add_argument("--batch", type=int, default=0, help="覆寫 batch size (0=自動)")
    ap.add_argument("--beam", type=int, default=5, help="beam size (預設 5;1=最快 greedy,10=極致準度)")
    ap.add_argument("--fast", action="store_true",
                    help="開啟 int8_bfloat16 量化 (1.5-1.7x 速度;中文 CER +0.3-0.5%%)")
    ap.add_argument("--no-compile", action="store_true", help="關 torch.compile (僅 transformers)")
    ap.add_argument("--no-s2tw", action="store_true", help="關 OpenCC 簡→繁台灣化")
    ap.add_argument("--internal-vad", action="store_true",
                    help="改用 faster-whisper 內建 VAD (預設用我方 ONNX Silero,管線與 Qwen3 對等)")
    ap.add_argument("--glossary-file", default="",
                    help="保護詞彙檔 (一行一詞);會餵給 Whisper initial_prompt + hotwords")
    args = ap.parse_args()

    print("=" * 72)
    print(" Breeze-ASR-25 v2 (5090 Blackwell + bf16 + beam=5 + s2twp + 24 核 CPU)")
    print("=" * 72)
    cfg = detect_hw(fast=args.fast)
    if args.batch > 0: cfg.batch_size = args.batch
    print(f" {cfg.desc}  CPU threads={_NCPU}")
    print(f" dtype={cfg.dtype} | ct2_compute={cfg.ct2_compute} | "
          f"batch={cfg.batch_size} | beam={args.beam}")

    s2tw = S2TW(not args.no_s2tw)
    glossary = load_glossary(args.glossary_file) if args.glossary_file else []
    if glossary:
        print(f" Glossary: {len(glossary)} 詞 → 注入 initial_prompt + hotwords")
    backend = select_backend(cfg, args.beam, s2tw, args.backend,
                             use_manual_vad=not args.internal_vad,
                             glossary=glossary)
    if isinstance(backend, TransformersBackend) and args.no_compile:
        backend.compile_mode = None
    print(f" backend = {backend.NAME}")
    backend.load()

    for src in args.inputs:
        if not os.path.exists(src):
            print(f" 找不到 {src}"); continue
        print("\n" + "─" * 72)
        print(f" {src}")
        segs, sw = backend.transcribe(src, language=args.lang)
        sw.report()
        save_outputs(segs, src, args.out, suffix="breeze")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n 完成。")


if __name__ == "__main__":
    main()
