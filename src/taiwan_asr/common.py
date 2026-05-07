# -*- coding: utf-8 -*-
"""
ASR 共用工具 — Blackwell sm_120 / RTX 5090 / i9-14900 (24T) 優化共用模組
* 環境變數 (必須在 import torch / numpy 之前設定)
* 音檔處理 (ffmpeg pipe 直讀,跳過暫存 wav)
* OpenCC 簡→繁台灣化 (多重 fallback)
* Silero VAD ONNX 後端封裝
* Length-sorted batching helper
"""
from __future__ import annotations
import os, sys, gc, time, math, json, subprocess, tempfile, threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any, Iterator

# ------------------------------------------------------------------
# 1) 全核 CPU + 大頁式 GPU 記憶體 (必須在 import torch 之前!)
# ------------------------------------------------------------------
def init_env(cpu_threads: Optional[int] = None) -> int:
    n = cpu_threads or (os.cpu_count() or 8)
    # OpenMP / MKL: NumPy / SciPy / sklearn / pyworld 共用
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # PyTorch 2.9+ 用 PYTORCH_ALLOC_CONF;舊版用 PYTORCH_CUDA_ALLOC_CONF
    _alloc = "expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.9"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", _alloc)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _alloc)
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # 抑制 TF/oneDNN 雜訊
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return n


def init_torch(cpu_threads: int):
    import torch
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(max(2, cpu_threads // 8))
    except Exception:
        pass
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.95)
        except Exception:
            pass


# ------------------------------------------------------------------
# 2) Segment + 輸出
# ------------------------------------------------------------------
@dataclass
class Segment:
    start: float
    end: float
    text: str
    language: str = ""
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    words: Optional[List[Dict[str, Any]]] = None
    speaker_id: Optional[str] = None    # diarize 後填入 (e.g., "SPEAKER_00")

    @staticmethod
    def fmt_time(t: float, srt: bool = False) -> str:
        if t is None or t < 0:
            t = 0.0
        h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
        if srt:
            return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
        return f"{h:02d}:{m:02d}:{int(s):02d}"

    def to_line(self) -> str:
        lang = f" [{self.language}]" if self.language else ""
        spk = f" {self.speaker_id}:" if self.speaker_id else ""
        return f"[{self.fmt_time(self.start)} - {self.fmt_time(self.end)}]{lang}{spk} {self.text}"

    def to_srt(self, idx: int) -> str:
        spk = f"{self.speaker_id}: " if self.speaker_id else ""
        return (
            f"{idx}\n"
            f"{self.fmt_time(self.start, True)} --> {self.fmt_time(self.end, True)}\n"
            f"{spk}{self.text}\n"
        )


def save_outputs(segs: List[Segment], src: str, out_dir: str, suffix: str) -> Dict[str, str]:
    if not segs:
        print(" 無有效轉錄結果,不寫檔")
        return {}
    base = Path(src).stem
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = {
        "txt": str(out / f"{base}_{suffix}.txt"),
        "srt": str(out / f"{base}_{suffix}.srt"),
        "json": str(out / f"{base}_{suffix}.json"),
    }
    with open(paths["txt"], "w", encoding="utf-8") as f:
        for s in segs:
            f.write(s.to_line() + "\n")
    with open(paths["srt"], "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            f.write(s.to_srt(i) + "\n")
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in segs], f, ensure_ascii=False, indent=1)
    for k, p in paths.items():
        print(f" {k.upper():4s} → {p}")
    return paths


# ------------------------------------------------------------------
# 3) OpenCC 簡→繁台灣化 (3 重 fallback;對已是繁體的字串近似冪等)
# ------------------------------------------------------------------
class S2TW:
    """簡體 → 繁體(台灣慣用詞) 後處理。

    嘗試順序 (越前面品質越好):
      1. opencc-python-reimplemented   OpenCC('s2twp')   ←  簡→繁(台灣慣用詞,例:軟件→軟體)
      2. opencc-python-reimplemented   OpenCC('s2tw')    ←  簡→繁(基礎)
      3. opencc                        OpenCC('s2twp.json')
      若全部失敗則 passthrough。
    """
    def __init__(self, enabled: bool = True):
        self.cc = None
        self.scheme = ""
        if not enabled:
            return
        for scheme, fn in [
            ("s2twp",      lambda: __import__("opencc").OpenCC("s2twp")),
            ("s2twp.json", lambda: __import__("opencc").OpenCC("s2twp.json")),
            ("s2tw",       lambda: __import__("opencc").OpenCC("s2tw")),
            ("s2tw.json",  lambda: __import__("opencc").OpenCC("s2tw.json")),
        ]:
            try:
                self.cc = fn()
                self.scheme = scheme
                return
            except Exception:
                continue

    def __call__(self, txt: str) -> str:
        if not txt or not self.cc:
            return txt
        try:
            return self.cc.convert(txt)
        except Exception:
            return txt

    def __repr__(self):
        return f"S2TW(scheme={self.scheme or 'disabled'})"


# ------------------------------------------------------------------
# 4) ffmpeg 直讀 PCM_F32LE 到 numpy (跳過暫存 wav,省 IO + 跳過 int16→f32)
# ------------------------------------------------------------------
class AudioIO:
    SR = 16000

    @staticmethod
    def decode_to_array(src: str, sr: int = SR, threads: Optional[int] = None) -> Tuple["np.ndarray", float]:
        """以 ffmpeg 直接 pipe 出 float32 mono 16kHz,全程不寫暫存檔。
        重採樣優先用 soxr (高品質),失敗自動退回 swr (ffmpeg 內建,對 16kHz 語音也足夠)。"""
        import numpy as np
        n = threads or (os.cpu_count() or 8)
        base = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-threads", str(n),
            "-i", src, "-vn", "-sn",
            "-ar", str(sr), "-ac", "1",
        ]
        out = ["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
        # 嘗試 1: soxr 高品質 (precision=28)
        for af in [["-af", "aresample=resampler=soxr:precision=28"], []]:
            try:
                proc = subprocess.run(base + af + out, capture_output=True, check=True)
                audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
                return audio, len(audio) / sr
            except subprocess.CalledProcessError:
                continue
        raise RuntimeError(f"ffmpeg 解碼失敗: {src}")

    @staticmethod
    def write_wav(audio: "np.ndarray", sr: int, dst: Optional[str] = None) -> str:
        """如後端只接受檔案路徑,將已解碼的 numpy 寫成最小臨時 wav。"""
        import soundfile as sf
        if dst is None:
            fd, dst = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        sf.write(dst, audio, sr, subtype="PCM_16")
        return dst


# ------------------------------------------------------------------
# 5) Silero VAD ONNX 後端 (CPU SIMD,~3-5x 快過 PyTorch 後端)
# ------------------------------------------------------------------
class SileroVAD:
    """Silero VAD wrapper supporting:
       device='cpu'    → ONNX runtime CPU (低開銷小檔最快)
       device='cuda:0' → PyTorch CUDA (~5-10x 快於 CPU 在長音檔)
       device='auto'   → cuda 可用時用 cuda,否則 cpu
    """
    def __init__(self, sr: int = 16000, max_sec: float = 28.0,
                 pad_sec: float = 0.4, min_sec: float = 0.4,
                 threshold: float = 0.4,
                 device: str = "auto"):
        self.sr = sr
        self.max_sec = max_sec
        self.pad_sec = pad_sec
        self.min_sec = min_sec
        self.threshold = threshold
        self.device = device
        self._resolved_device = ""   # 實際解析後 (cpu / cuda:0)
        self._model = None
        self._get_ts = None
        self._using = ""             # onnx / torch / cuda / hub

    def _resolve_device(self) -> str:
        # 'auto' 永遠選 cpu — 實測 (5090) Silero VAD 太小,
        # CPU ONNX SIMD 快過 GPU PyTorch (kernel launch 開銷 > 計算)。
        # 想用 GPU 請顯式設 device='cuda:0' (大量並行音訊時或許有用)。
        if self.device == "auto":
            return "cpu"
        return self.device

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        dev = self._resolve_device()
        self._resolved_device = dev
        is_cuda = dev.startswith("cuda")

        if is_cuda:
            # GPU 模式: PyTorch 後端 + .to(cuda)
            try:
                from silero_vad import load_silero_vad, get_speech_timestamps
                self._model = load_silero_vad(onnx=False).to(dev)
                self._get_ts = get_speech_timestamps
                self._using = "cuda"
                return
            except Exception:
                pass
            # fallback: torch.hub
            try:
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad", trust_repo=True,
                )
                self._model = model.to(dev)
                self._get_ts = utils[0]
                self._using = "cuda"
                return
            except Exception:
                # GPU 失敗,退回 CPU
                pass

        # CPU 模式: 優先 ONNX (SIMD 加速)
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
            self._model = load_silero_vad(onnx=True)
            self._get_ts = get_speech_timestamps
            self._using = "onnx"
            self._resolved_device = "cpu"
            return
        except Exception:
            pass
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
            self._model = load_silero_vad(onnx=False)
            self._get_ts = get_speech_timestamps
            self._using = "torch"
            self._resolved_device = "cpu"
            return
        except Exception:
            pass
        # 最後 fallback torch.hub
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad", trust_repo=True,
        )
        self._model = model
        self._get_ts = utils[0]
        self._using = "hub"
        self._resolved_device = "cpu"

    def speech_chunks(self, audio) -> List[Dict[str, Any]]:
        """回傳 list[{'audio':(np,sr),'start':sec,'end':sec}]"""
        import numpy as np, torch
        self._ensure()
        # ONNX → numpy;PyTorch CPU → torch tensor;PyTorch CUDA → torch tensor on cuda
        if self._using == "onnx":
            x_in = audio
        elif self._using == "cuda":
            x_in = torch.from_numpy(audio).float().to(self._resolved_device)
        else:
            x_in = torch.from_numpy(audio).float()

        ts_list = self._get_ts(
            x_in, self._model,
            sampling_rate=self.sr,
            threshold=self.threshold,
            min_speech_duration_ms=120,
            min_silence_duration_ms=200,
            speech_pad_ms=int(self.pad_sec * 1000),
        )
        out: List[Dict[str, Any]] = []
        max_n = int(self.max_sec * self.sr)
        cur_s = cur_e = -1
        for ts in ts_list:
            s, e = int(ts["start"]), int(ts["end"])
            n = e - s
            if n > max_n:
                if cur_s >= 0:
                    out.append(self._mk(audio, cur_s, cur_e))
                    cur_s = cur_e = -1
                for i in range(s, e, max_n):
                    j = min(i + max_n, e)
                    out.append(self._mk(audio, i, j))
                continue
            if cur_s < 0:
                cur_s, cur_e = s, e
            elif (e - cur_s) <= max_n:
                cur_e = e
            else:
                out.append(self._mk(audio, cur_s, cur_e))
                cur_s, cur_e = s, e
        if cur_s >= 0:
            out.append(self._mk(audio, cur_s, cur_e))
        out = [c for c in out if (c["end"] - c["start"]) >= self.min_sec]
        return out

    def _mk(self, audio, s_n: int, e_n: int) -> Dict[str, Any]:
        s_n = max(0, s_n); e_n = min(len(audio), e_n)
        return {"audio": (audio[s_n:e_n], self.sr),
                "start": s_n / self.sr, "end": e_n / self.sr}


# ------------------------------------------------------------------
# 5b) Glossary 載入 — 給 ASR initial_prompt / hotwords 用
# ------------------------------------------------------------------
def builtin_glossary_path() -> Path:
    """Return the path to the packaged default NTU glossary that ships with the wheel.

    Resolution order:
      1. importlib.resources (works for installed wheels and editable installs)
      2. fallback: relative to this file (works in clone without install)
    """
    try:
        # importlib.resources >=3.9 returns a Traversable; .as_posix() works for filesystem
        from importlib.resources import files
        ref = files("taiwan_asr.data") / "ntu_glossary.txt"
        # Materialize to a real path (importlib.resources may return zip-internal traversable)
        if hasattr(ref, "is_file") and ref.is_file():
            return Path(str(ref))
    except Exception:
        pass
    return Path(__file__).parent / "data" / "ntu_glossary.txt"


def load_glossary(path: str) -> List[str]:
    """讀 glossary 文字檔。每行一個詞,跳過 # 註解與空行,維持出現順序去重。

    Magic value:
      path == "builtin" (case-insensitive) -> use the packaged default NTU glossary
      that ships with the wheel (research dorms / departments / school names).

    檔案不存在則返回 []。"""
    if path and str(path).lower() == "builtin":
        p = builtin_glossary_path()
    else:
        p = Path(path)
    if not p.is_file():
        return []
    seen = set()
    out: List[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# ------------------------------------------------------------------
# 6) Length-sorted batching — 同一 batch 內長度相近,padding 浪費降到最低
# ------------------------------------------------------------------
def length_sorted_batches(items: List[Dict[str, Any]], batch_size: int) -> List[List[int]]:
    """回傳 list of 索引 list (回到原 items 用)。
    策略:依 audio 長度遞減排序,然後切 batch。"""
    if not items:
        return []
    order = sorted(range(len(items)),
                   key=lambda i: -len(items[i]["audio"][0]))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


# ------------------------------------------------------------------
# 7) 計時 helper
# ------------------------------------------------------------------
class Stopwatch:
    def __init__(self):
        self.events: List[Tuple[str, float]] = []
        self.t0 = time.perf_counter()
        self.last = self.t0

    def lap(self, label: str):
        now = time.perf_counter()
        self.events.append((label, now - self.last))
        self.last = now

    def get(self, prefix: str) -> float:
        """以 prefix 比對,回傳第一個吻合事件的耗時。"""
        for label, dt in self.events:
            if label.startswith(prefix):
                return dt
        return 0.0

    def total(self) -> float:
        return time.perf_counter() - self.t0

    def report(self):
        total = self.total()
        print("⏱  時間分析:")
        for label, dt in self.events:
            pct = 100 * dt / max(total, 1e-9)
            print(f" {label:<28s} {dt:7.2f}s  ({pct:5.1f}%)")
        print(f" {'TOTAL':<28s} {total:7.2f}s")
