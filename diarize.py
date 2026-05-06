# -*- coding: utf-8 -*-
"""
講者分離 (Speaker Diarization) — pyannote.audio 3.x

用法 (CLI):
    python3 diarize.py transcripts/breeze/標準錄音 886_breeze.json music/標準錄音 886.mp3
    → 寫入 transcripts/breeze-diarized/標準錄音 886_breeze-diarized.{txt,srt,json}

設計原則:
* 不替代 ASR — 只在 ASR 完成後加 speaker_id 標籤
* 缺 HF token / 缺模型時 graceful fail (回傳原 segments,不 crash)
* 與 _asr_common.Segment 相容 (使用 speaker_id 欄位)
* GPU 自動 (有 cuda 用 cuda)
"""
from __future__ import annotations

from _asr_common import init_env
_NCPU = init_env()

import os, sys, gc, time, json, argparse, warnings
from pathlib import Path
from dataclasses import asdict
from typing import List, Tuple, Optional

warnings.filterwarnings("ignore")
import logging
for _logger_name in ("pyannote", "pytorch_lightning", "speechbrain"):
    try:
        logging.getLogger(_logger_name).setLevel(logging.ERROR)
    except Exception:
        pass

from _asr_common import Segment, save_outputs, AudioIO


# ============================================================
# 1) Diarization 主管線 — 跑 pyannote pipeline 取得 (start, end, speaker)
# ============================================================
def run_pyannote(audio_array, sr: int = 16000,
                 device: str = "auto",
                 model_id: str = "tensorlake/speaker-diarization-3.1",
                 hf_token: Optional[str] = None) -> List[Tuple[float, float, str]]:
    """跑 pyannote.audio 講者分離。
    回傳 list of (start_sec, end_sec, speaker_id)。
    缺套件 / 缺 token / 缺模型 → 回傳 [] (上游決定退路)。
    """
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        print(" pyannote.audio 未安裝 (uv pip install --system pyannote.audio)", file=sys.stderr)
        return []

    dev = device
    if device == "auto":
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    token = hf_token or os.environ.get("HF_TOKEN")
    pipeline = None
    last_err = None
    # pyannote v4 用 token=,v3 用 use_auth_token=,v2 不需要
    for kwargs in (
        {"token": token} if token else {},
        {"use_auth_token": token} if token else {},
        {"token": True},        # v4 自動讀 ~/.cache/huggingface/token
        {"use_auth_token": True},
        {},
    ):
        try:
            pipeline = Pipeline.from_pretrained(model_id, **kwargs)
            break
        except TypeError as e:
            last_err = e; continue
        except Exception as e:
            last_err = e; break
    if pipeline is None:
        msg = str(last_err)
        is_gated = "gated" in msg.lower() or "Gated" in msg or "403" in msg
        print(f" pyannote 模型載入失敗 ({type(last_err).__name__})", file=sys.stderr)
        if is_gated:
            print(" ↪ 此模型受 license 保護,請執行下列步驟 (一次性):", file=sys.stderr)
            print(" 1. 開啟瀏覽器登入 HuggingFace 帳號", file=sys.stderr)
            print(f" 2. 造訪 https://hf.co/{model_id} 點 'Agree and access repository'", file=sys.stderr)
            print(f" 3. 同樣造訪 https://hf.co/pyannote/speaker-diarization-community-1", file=sys.stderr)
            print(f" 4. 同樣造訪 https://hf.co/pyannote/segmentation-3.0", file=sys.stderr)
            print(" 5. 確認 https://hf.co/settings/tokens 的 token 有 'Read' 權限", file=sys.stderr)
            print(f" 6. 重跑此命令 (HF_TOKEN 已自動讀 ~/.cache/huggingface/token)", file=sys.stderr)
        else:
            print(f" 錯誤:{msg[:200]}", file=sys.stderr)
        return []

    if dev.startswith("cuda"):
        try:
            pipeline.to(torch.device(dev))
        except Exception:
            pass

    # pyannote v3 接受 in-memory 字典格式
    import torch
    waveform = torch.from_numpy(audio_array).float().unsqueeze(0)  # (1, T)
    audio_in = {"waveform": waveform, "sample_rate": sr}

    diar = pipeline(audio_in)
    out: List[Tuple[float, float, str]] = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        out.append((float(turn.start), float(turn.end), str(speaker)))
    return out


# ============================================================
# 2) 對齊器 — 把 ASR segments 與 diar 結果合併,給每段加 speaker_id
# ============================================================
def assign_speakers(
    segments: List[Segment],
    diar_segments: List[Tuple[float, float, str]],
) -> List[Segment]:
    """以「最大時間重疊」原則:每個 ASR 段找與 diar 段重疊最久的 speaker。
    diar_segments 為空時,直接返回 segments (不修改 speaker_id)。
    """
    if not diar_segments:
        return segments
    out: List[Segment] = []
    for seg in segments:
        # 計算 seg 與每個 diar 段的重疊長度
        best_spk = None
        best_overlap = 0.0
        for d_start, d_end, d_spk in diar_segments:
            overlap = max(0.0, min(seg.end, d_end) - max(seg.start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = d_spk
        # 若沒重疊,fallback 用中點落在哪個 diar 段
        if best_spk is None:
            mid = (seg.start + seg.end) / 2
            for d_start, d_end, d_spk in diar_segments:
                if d_start <= mid <= d_end:
                    best_spk = d_spk
                    break
        # 複製並加 speaker_id (避免 mutate 原 segment)
        new_seg = Segment(
            start=seg.start, end=seg.end, text=seg.text,
            language=seg.language,
            avg_logprob=seg.avg_logprob,
            no_speech_prob=seg.no_speech_prob,
            words=seg.words,
            speaker_id=best_spk,
        )
        out.append(new_seg)
    return out


# ============================================================
# 3) CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="ASR 結果 + pyannote 講者分離")
    ap.add_argument("asr_json", help="ASR 輸出 JSON (含 segments)")
    ap.add_argument("audio", help="原始音訊檔")
    ap.add_argument("--out", default="", help="輸出目錄 (預設:同層 + -diarized)")
    ap.add_argument("--model", default="tensorlake/speaker-diarization-3.1",
                    help="預設用開放鏡像 tensorlake/...,不需 license accept;"
                         "或改 pyannote/speaker-diarization-3.1 (需先在 HF 接受授權)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hf-token", default="", help="HF token (或設環境變數 HF_TOKEN)")
    args = ap.parse_args()

    print("=" * 72)
    print(" Speaker Diarization (pyannote.audio)")
    print("=" * 72)

    # 載入 ASR segments
    asr_path = Path(args.asr_json)
    if not asr_path.is_file():
        print(f" 找不到 ASR JSON: {asr_path}"); return
    asr_data = json.loads(asr_path.read_text(encoding="utf-8"))
    segments = [Segment(
        start=float(s.get("start", 0.0) or 0.0),
        end=float(s.get("end", 0.0) or 0.0),
        text=s.get("text", "") or "",
        language=s.get("language", "") or "",
        avg_logprob=float(s.get("avg_logprob", 0.0) or 0.0),
        no_speech_prob=float(s.get("no_speech_prob", 0.0) or 0.0),
        words=s.get("words"),
        speaker_id=s.get("speaker_id"),
    ) for s in asr_data]
    print(f" ASR: {len(segments)} 段 from {asr_path}")

    # 載入音訊
    print(f" 解碼: {args.audio}")
    audio, dur = AudioIO.decode_to_array(args.audio)
    print(f" {dur:.1f}s ({dur/60:.1f} 分)")

    # 跑 pyannote
    print(f"⏳ 載入 + 跑 pyannote {args.model}…")
    t0 = time.time()
    diar = run_pyannote(audio, sr=16000, device=args.device,
                        model_id=args.model,
                        hf_token=args.hf_token or None)
    el = time.time() - t0
    if not diar:
        print(" diarization 為空,輸出將不含 speaker 標籤")
    else:
        speakers = sorted(set(d[2] for d in diar))
        print(f" {len(diar)} 個 turn 偵測到,{len(speakers)} 位 speaker ({speakers}) | "
              f"耗時 {el:.1f}s")

    # 對齊
    out_segs = assign_speakers(segments, diar)

    # 輸出
    if args.out:
        out_dir = args.out
    else:
        # transcripts/breeze/foo.json → transcripts/breeze-diarized/
        parent = asr_path.parent
        out_dir = str(parent.parent / (parent.name + "-diarized"))

    # 推導 base_name (去 _qwen3 / _breeze suffix)
    import re
    stem = asr_path.stem
    m = re.search(r"_(qwen3|breeze)$", stem)
    base = stem[:m.start()] if m else stem
    suffix = (m.group(1) if m else "asr") + "-diarized"

    save_outputs(out_segs, str(asr_path.with_name(base)), out_dir, suffix=suffix)
    print("\n 完成。")


if __name__ == "__main__":
    main()
