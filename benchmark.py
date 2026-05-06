# -*- coding: utf-8 -*-
"""
ASR 完整 Benchmark — 速度 + 準度多維度比較 (Qwen3 vs Breeze)

無人工 ground truth 時,「最準」用以下證據綜合判斷:
  1. **覆蓋率**  (sum 段時長 / 音訊時長): 漏音越少越好
  2. **幻覺信號** (Whisper 經典失控指標):
       - 段長 > 60s (一個 chunk 應 ≤30s,超出代表 boundary 失控)
       - 段內 chars/sec ∉ [1, 18] (中文正常範圍,過密=亂碼,過疏=漏字)
       - 重複字元比 (連續 3+ 重複)
  3. **段密度** (chars/sec 整體): 中文正常約 4-7 chars/sec
  4. **詞彙多樣性** (unique chars / total chars)
  5. **繁體率** (s2twp 後成效)
  6. **跨模型一致性** Jaccard / SeqSim (兩模型同意處)

最後給出加權排名與單檔判決。
"""
from __future__ import annotations
import os, sys, json, re, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

# 跑分檔案會用到 ffprobe
import subprocess


# ─────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────
def load_segments(p: Path) -> List[Dict[str, Any]]:
    if not p.is_file():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def normalize_text(t: str) -> str:
    return re.sub(
        r"[\s\,\.\!\?\;\:\、\。\,\!\?\;\:\「\」\『\』\"\'\(\)（）\-\_\d]+",
        "", t or "")


def char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def edit_distance_ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    if not a and not b: return 1.0
    return SequenceMatcher(None, a, b).ratio()


_SIMPLIFIED_HINTS = set("国发现实际经济认识让说话语门间问题点头计较实际讲话数据软件视频质量")
_TRADITIONAL_HINTS = set("國發現實際經濟認識讓說話語門間問題點頭計較實際講話數據軟體影片品質")


def trad_ratio(text: str) -> float:
    if not text: return 0.0
    s = sum(1 for c in text if c in _SIMPLIFIED_HINTS)
    t = sum(1 for c in text if c in _TRADITIONAL_HINTS)
    return 1.0 if (s + t) == 0 else t / (s + t)


def file_audio_duration(path: Path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def char_repeat_ratio(text: str, run_len: int = 3) -> float:
    """連續 ≥ run_len 個重複字元的比例 (越低越正常)"""
    if not text: return 0.0
    n = len(text)
    rep = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and text[j + 1] == text[i]:
            j += 1
        run = j - i + 1
        if run >= run_len:
            rep += run
        i = j + 1
    return rep / n


# ─────────────────────────────────────────────────────────────
# 每模型每檔指標
# ─────────────────────────────────────────────────────────────
@dataclass
class ModelStats:
    n_segs: int = 0
    n_chars: int = 0
    coverage_sec: float = 0.0
    coverage_ratio: float = 0.0
    chars_per_sec: float = 0.0
    vocab_unique: int = 0
    vocab_diversity: float = 0.0
    trad_ratio: float = 0.0
    repeat_ratio: float = 0.0
    n_seg_too_long: int = 0          # > 60s
    n_seg_too_dense: int = 0         # chars/sec > 18
    n_seg_too_sparse: int = 0        # chars/sec < 1 (and > 5s 段)
    halluc_score: float = 0.0        # 綜合幻覺分 (0=無, 1=嚴重)
    quality_score: float = 0.0       # 綜合品質分 (越高越好)


def compute_model_stats(segs: List[Dict[str, Any]], audio_dur: float) -> ModelStats:
    s = ModelStats()
    if not segs or audio_dur <= 0:
        return s
    full_text_norm = ""
    full_text_raw = ""
    for sg in segs:
        text = (sg.get("text") or "").strip()
        st = float(sg.get("start", 0.0) or 0.0)
        en = float(sg.get("end", 0.0) or 0.0)
        seg_dur = max(0.0, en - st)
        full_text_norm += normalize_text(text)
        full_text_raw += text
        s.n_segs += 1
        s.n_chars += len(normalize_text(text))
        s.coverage_sec += seg_dur
        # 幻覺信號
        if seg_dur > 60.0:
            s.n_seg_too_long += 1
        cps = len(text) / max(seg_dur, 0.5)
        if cps > 18.0:
            s.n_seg_too_dense += 1
        if seg_dur > 5.0 and cps < 1.0:
            s.n_seg_too_sparse += 1

    s.coverage_ratio = min(1.0, s.coverage_sec / audio_dur)
    s.chars_per_sec = s.n_chars / max(audio_dur, 1e-9)
    s.vocab_unique = len(set(full_text_norm))
    s.vocab_diversity = s.vocab_unique / max(s.n_chars, 1)
    s.trad_ratio = trad_ratio(full_text_raw)
    s.repeat_ratio = char_repeat_ratio(full_text_norm)

    # 幻覺分: 嚴重失控 (>60s 段) + 太密 + 太稀 + 重複,加權
    halluc_raw = (
        s.n_seg_too_long * 0.5 +
        s.n_seg_too_dense * 0.05 +
        s.n_seg_too_sparse * 0.05 +
        s.repeat_ratio * 5.0
    )
    s.halluc_score = min(1.0, halluc_raw / max(s.n_segs, 1))

    # 品質分: 覆蓋 + 中文密度合理 + 繁體 + 多樣性 - 幻覺
    cps_score = 1.0 if 3.0 <= s.chars_per_sec <= 9.0 else max(0.0, 1.0 - abs(s.chars_per_sec - 6.0) / 6.0)
    s.quality_score = (
        0.35 * s.coverage_ratio +
        0.20 * cps_score +
        0.15 * s.trad_ratio +
        0.15 * min(1.0, s.vocab_diversity * 5.0) +
        0.15 * (1.0 - s.halluc_score)
    )
    return s


# ─────────────────────────────────────────────────────────────
# 跨模型一致性
# ─────────────────────────────────────────────────────────────
@dataclass
class FilePair:
    file: str
    duration: float
    qwen3: ModelStats
    breeze: ModelStats
    char_jaccard: float = 0.0
    char_sim: float = 0.0
    qwen3_cer: Optional[float] = None
    breeze_cer: Optional[float] = None
    gt_clip_end: Optional[float] = None
    gt_chars: int = 0


# ─────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────
def fmt_pct(x: float) -> str:
    return f"{100*x:5.1f}%"


def main():
    ap = argparse.ArgumentParser(description="ASR Benchmark (Qwen3 vs Breeze)")
    ap.add_argument("--qwen3-dir", default="transcripts/qwen3")
    ap.add_argument("--breeze-dir", default="transcripts/breeze")
    ap.add_argument("--audio-dir", default="music")
    ap.add_argument("--timing", default="",
                    help="可選:TSV 計時檔 (file\\tdur\\tbreeze_asr\\tqwen3_asr) 提供速度欄")
    ap.add_argument("--gt-dir", default="tests/fixtures",
                    help="ground-truth 目錄;檔名 {audio_basename}_first_60s_gt.txt 表示前 60 秒 GT")
    ap.add_argument("--out", default="transcripts/BENCHMARK.md")
    args = ap.parse_args()

    qdir = Path(args.qwen3_dir)
    bdir = Path(args.breeze_dir)
    adir = Path(args.audio_dir)

    # 計時 (若有)
    timing: Dict[str, Dict[str, float]] = {}
    if args.timing and Path(args.timing).is_file():
        for line in Path(args.timing).read_text(encoding="utf-8").splitlines():
            p = line.strip().split("\t")
            if len(p) == 4 and not p[0].startswith("#") and p[0] != "file":
                try:
                    timing[p[0]] = {
                        "dur": float(p[1]),
                        "breeze_asr": float(p[2]),
                        "qwen3_asr": float(p[3]),
                    }
                except ValueError:
                    continue

    # 找 GT 檔 (檔名規則:{stem}_first_{N}s_gt.txt)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None

    def _find_gt(stem: str):
        """回傳 (gt_path, clip_end_seconds) 或 (None, None)。"""
        if not gt_dir or not gt_dir.is_dir():
            return None, None
        # 容忍 NFC/NFD 與底線變體的檔名匹配
        for p in gt_dir.iterdir():
            name = p.name
            if not name.startswith(stem):
                continue
            if not name.endswith("_gt.txt"):
                continue
            # 解析 _first_{N}s_gt.txt
            mid = name[len(stem):-len("_gt.txt")]
            m = re.search(r"_first_(\d+(?:\.\d+)?)s", mid)
            clip_end = float(m.group(1)) if m else None
            return p, clip_end
        return None, None

    pairs: List[FilePair] = []
    for af in sorted(adir.iterdir(), key=lambda p: p.name):
        if not af.is_file():
            continue
        stem = af.stem
        q = load_segments(qdir / f"{stem}_qwen3.json")
        b = load_segments(bdir / f"{stem}_breeze.json")
        dur = file_audio_duration(af)
        q_norm = "".join(normalize_text(s.get("text", "")) for s in q)
        b_norm = "".join(normalize_text(s.get("text", "")) for s in b)

        # CER (若有 ground-truth)
        q_cer = b_cer = None
        gt_clip_end = None
        gt_chars = 0
        gt_path, gt_clip_end = _find_gt(stem)
        if gt_path and gt_path.is_file():
            try:
                from cer_eval import (
                    compute_cer as _ccer, hyp_text_from_path, read_text_file, normalize as _norm,
                )
                ref_text = read_text_file(str(gt_path))
                gt_chars = len(_norm(ref_text))
                # hyp 拼接已剪裁的 segments
                q_hyp = hyp_text_from_path(str(qdir / f"{stem}_qwen3.json"),
                                           clip_end=gt_clip_end)
                b_hyp = hyp_text_from_path(str(bdir / f"{stem}_breeze.json"),
                                           clip_end=gt_clip_end)
                q_cer = _ccer(ref_text, q_hyp)
                b_cer = _ccer(ref_text, b_hyp)
            except Exception:
                pass

        pairs.append(FilePair(
            file=af.name,
            duration=dur,
            qwen3=compute_model_stats(q, dur),
            breeze=compute_model_stats(b, dur),
            char_jaccard=char_jaccard(q_norm, b_norm) if q_norm and b_norm else 0.0,
            char_sim=edit_distance_ratio(q_norm, b_norm) if q_norm and b_norm else 0.0,
            qwen3_cer=q_cer, breeze_cer=b_cer,
            gt_clip_end=gt_clip_end, gt_chars=gt_chars,
        ))

    # ─── 輸出 ───
    print("\n" + "=" * 92)
    print(" ASR 完整 Benchmark (Qwen3-ASR-1.7B vs Breeze-ASR-25)")
    print("=" * 92)

    # (1) 速度排行 (若有計時)
    if timing:
        print("\n##  速度 (越右邊越快)")
        print(f"{'檔名':<30} {'時長':>8} {'Br_ASR':>8} {'Br_RTF':>8} {'Q3_ASR':>8} {'Q3_RTF':>8} {'勝者':>6}")
        print("-" * 92)
        tot_dur = tot_b = tot_q = 0.0
        b_wins = q_wins = 0
        for p in pairs:
            t = timing.get(p.file)
            if not t: continue
            br_rtf = t["dur"] / max(t["breeze_asr"], 1e-9)
            q3_rtf = t["dur"] / max(t["qwen3_asr"], 1e-9)
            winner = "Q3 " if q3_rtf > br_rtf else "Br "
            if q3_rtf > br_rtf: q_wins += 1
            else: b_wins += 1
            tot_dur += t["dur"]; tot_b += t["breeze_asr"]; tot_q += t["qwen3_asr"]
            print(f"{p.file[:28]:<30} {t['dur']/60:>6.1f}min "
                  f"{t['breeze_asr']:>7.2f}s {br_rtf:>7.0f}x "
                  f"{t['qwen3_asr']:>7.2f}s {q3_rtf:>7.0f}x {winner:>6}")
        print("-" * 92)
        print(f"{'總計':<30} {tot_dur/60:>6.1f}min "
              f"{tot_b:>7.1f}s {tot_dur/max(tot_b,1e-9):>7.0f}x "
              f"{tot_q:>7.1f}s {tot_dur/max(tot_q,1e-9):>7.0f}x  Q3:{q_wins} Br:{b_wins}")

    # (2) 覆蓋率 + 幻覺
    print("\n##  覆蓋率 + 幻覺信號 (越高/越低見表頭箭頭)")
    print(f"{'檔名':<30} {'cov% ↑':>10} {'cov% ↑':>10} {'>60s 段 ↓':>10} {'>60s 段 ↓':>10} {'halluc ↓':>10} {'halluc ↓':>10}")
    print(f"{'':<30} {'  Q3':>10} {'  Br':>10} {'  Q3':>10} {'  Br':>10} {'  Q3':>10} {'  Br':>10}")
    print("-" * 92)
    for p in pairs:
        q, b = p.qwen3, p.breeze
        print(f"{p.file[:28]:<30} {fmt_pct(q.coverage_ratio):>10} {fmt_pct(b.coverage_ratio):>10} "
              f"{q.n_seg_too_long:>10d} {b.n_seg_too_long:>10d} "
              f"{q.halluc_score:>10.3f} {b.halluc_score:>10.3f}")

    # (3) 內容指標
    print("\n##  內容指標 (chars/s 中文常態 4-7,vocab 多樣性 0.05-0.2 為健康)")
    print(f"{'檔名':<30} {'Q3 字':>8} {'Br 字':>8} {'Q3 c/s':>8} {'Br c/s':>8} {'Q3 vocab':>10} {'Br vocab':>10}")
    print("-" * 92)
    for p in pairs:
        q, b = p.qwen3, p.breeze
        print(f"{p.file[:28]:<30} {q.n_chars:>8d} {b.n_chars:>8d} "
              f"{q.chars_per_sec:>8.2f} {b.chars_per_sec:>8.2f} "
              f"{q.vocab_diversity:>10.4f} {b.vocab_diversity:>10.4f}")

    # (4) 跨模型同意度
    print("\n##  跨模型一致性 (越高代表兩模型同意)")
    print(f"{'檔名':<30} {'Jaccard':>10} {'SeqSim':>10} {'Q3 繁':>8} {'Br 繁':>8}")
    print("-" * 92)
    for p in pairs:
        print(f"{p.file[:28]:<30} {p.char_jaccard:>10.3f} {p.char_sim:>10.3f} "
              f"{p.qwen3.trad_ratio:>8.2f} {p.breeze.trad_ratio:>8.2f}")

    # (5) 綜合品質分
    print("\n##  綜合品質分 (覆蓋 35% + 中文密度 20% + 繁體 15% + 多樣性 15% + 反幻覺 15%)")
    print(f"{'檔名':<30} {'Q3 score':>12} {'Br score':>12} {'勝者':>8}")
    print("-" * 92)
    q_score_sum = b_score_sum = 0.0
    q_score_wins = b_score_wins = 0
    for p in pairs:
        qs, bs = p.qwen3.quality_score, p.breeze.quality_score
        q_score_sum += qs; b_score_sum += bs
        if qs > bs:
            winner = "Q3 "; q_score_wins += 1
        elif bs > qs:
            winner = "Br "; b_score_wins += 1
        else:
            winner = "tie"
        print(f"{p.file[:28]:<30} {qs:>12.3f} {bs:>12.3f} {winner:>8}")
    print("-" * 92)
    avg_q = q_score_sum / max(len(pairs), 1)
    avg_b = b_score_sum / max(len(pairs), 1)
    print(f"{'平均':<30} {avg_q:>12.3f} {avg_b:>12.3f}  Q3勝{q_score_wins} Br勝{b_score_wins}")

    # ─── CER (若有 GT) ───
    pairs_with_cer = [p for p in pairs if p.breeze_cer is not None]
    if pairs_with_cer:
        print("\n##  真 CER (vs ground truth,越低越好)\n")
        print(f"{'檔名':<30} {'GT 字數':>8} {'GT 範圍':>10} {'Q3 CER':>9} {'Br CER':>9}")
        print("-" * 75)
        for p in pairs_with_cer:
            rng = f"0-{p.gt_clip_end:.0f}s" if p.gt_clip_end else "全段"
            print(f"{p.file[:28]:<30} {p.gt_chars:>8d} {rng:>10} "
                  f"{p.qwen3_cer*100:>8.2f}% {p.breeze_cer*100:>8.2f}%")

    # ─── 最終結論 ───
    print("\n" + "=" * 92)
    print(" 結論 (基於上述客觀指標 + 真 CER 若可用)")
    print("=" * 92)
    if avg_q > avg_b + 0.02:
        print(f" **準度排名**: Qwen3-ASR-1.7B 勝出 (品質分 {avg_q:.3f} vs Breeze {avg_b:.3f})")
    elif avg_b > avg_q + 0.02:
        print(f" **準度排名**: Breeze-ASR-25 勝出 (品質分 {avg_b:.3f} vs Qwen3 {avg_q:.3f})")
    else:
        print(f" **準度排名**: 兩模型大致相當 (品質分 Q3={avg_q:.3f}, Br={avg_b:.3f})")
    if timing:
        rtf_q = tot_dur / max(tot_q, 1e-9)
        rtf_b = tot_dur / max(tot_b, 1e-9)
        if rtf_q > rtf_b:
            print(f" **速度排名**: Qwen3-ASR 較快 (RTF {rtf_q:.0f}x vs Breeze {rtf_b:.0f}x)")
        else:
            print(f" **速度排名**: Breeze-ASR 較快 (RTF {rtf_b:.0f}x vs Qwen3 {rtf_q:.0f}x)")

    print("\n 證據摘要:")
    q_long = sum(p.qwen3.n_seg_too_long for p in pairs)
    b_long = sum(p.breeze.n_seg_too_long for p in pairs)
    print(f" • >60s 失控段: Qwen3={q_long}, Breeze={b_long}  → {'Qwen3 較穩' if q_long < b_long else ('Breeze 較穩' if b_long < q_long else '相當')}")
    q_cov = sum(p.qwen3.coverage_ratio for p in pairs) / len(pairs)
    b_cov = sum(p.breeze.coverage_ratio for p in pairs) / len(pairs)
    print(f" • 平均覆蓋率: Qwen3={fmt_pct(q_cov)}, Breeze={fmt_pct(b_cov)}")
    avg_jacc = sum(p.char_jaccard for p in pairs) / len(pairs)
    avg_sim = sum(p.char_sim for p in pairs) / len(pairs)
    print(f" • 跨模型同意度: Jaccard={avg_jacc:.3f}, SeqSim={avg_sim:.3f}")

    # ─── Markdown 輸出 ───
    out_p = Path(args.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# ASR Benchmark Report\n")
    md.append(f"**Hardware**: RTX 5090 (Blackwell sm_120) + i9-14900 + 125GB RAM\n")
    md.append(f"**檔案數**: {len(pairs)} | **總時長**: {sum(p.duration for p in pairs)/60:.1f} 分鐘\n")
    md.append(f"**模型**: Qwen/Qwen3-ASR-1.7B + Aligner-0.6B  vs  MediaTek-Research/Breeze-ASR-25\n")
    md.append(f"**配置**: 兩者均 bf16 + cuDNN-SDPA + OpenCC s2twp 後處理\n\n")

    if timing:
        md.append("##  速度\n")
        md.append("| 檔名 | 時長 | Breeze ASR | Breeze RTF | Qwen3 ASR | Qwen3 RTF | 勝者 |")
        md.append("|---|---:|---:|---:|---:|---:|:--:|")
        for p in pairs:
            t = timing.get(p.file);
            if not t: continue
            br_rtf = t["dur"] / max(t["breeze_asr"], 1e-9)
            q3_rtf = t["dur"] / max(t["qwen3_asr"], 1e-9)
            md.append(f"| {p.file} | {t['dur']/60:.2f}min | {t['breeze_asr']:.2f}s | {br_rtf:.0f}x | "
                      f"{t['qwen3_asr']:.2f}s | {q3_rtf:.0f}x | {'Q3 ' if q3_rtf>br_rtf else 'Br '} |")
        md.append(f"| **總計** | **{tot_dur/60:.1f}min** | **{tot_b:.1f}s** | "
                  f"**{tot_dur/max(tot_b,1e-9):.0f}x** | **{tot_q:.1f}s** | "
                  f"**{tot_dur/max(tot_q,1e-9):.0f}x** | — |")
        md.append("")

    md.append("##  覆蓋率 / 幻覺信號\n")
    md.append("| 檔名 | Q3 cov% | Br cov% | Q3 >60s | Br >60s | Q3 halluc | Br halluc |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for p in pairs:
        q, b = p.qwen3, p.breeze
        md.append(f"| {p.file} | {fmt_pct(q.coverage_ratio)} | {fmt_pct(b.coverage_ratio)} | "
                  f"{q.n_seg_too_long} | {b.n_seg_too_long} | "
                  f"{q.halluc_score:.3f} | {b.halluc_score:.3f} |")
    md.append("")

    md.append("##  綜合品質分\n")
    md.append("計算式: 0.35 × 覆蓋率 + 0.20 × 中文密度合理性 + 0.15 × 繁體率 + 0.15 × 詞彙多樣性 + 0.15 × (1 − 幻覺分)\n")
    md.append("| 檔名 | Q3 score | Br score | 勝者 |")
    md.append("|---|---:|---:|:--:|")
    for p in pairs:
        qs, bs = p.qwen3.quality_score, p.breeze.quality_score
        winner = "Q3 " if qs > bs else ("Br " if bs > qs else "tie")
        md.append(f"| {p.file} | {qs:.3f} | {bs:.3f} | {winner} |")
    md.append(f"| **平均** | **{avg_q:.3f}** | **{avg_b:.3f}** | "
              f"**{'Q3' if avg_q>avg_b else ('Br' if avg_b>avg_q else 'tie')}** |")
    md.append("")

    md.append("## 結論\n")
    if avg_q > avg_b + 0.02:
        md.append(f"- **準度**: Qwen3-ASR-1.7B 勝 (品質 {avg_q:.3f} vs {avg_b:.3f})")
    elif avg_b > avg_q + 0.02:
        md.append(f"- **準度**: Breeze-ASR-25 勝 (品質 {avg_b:.3f} vs {avg_q:.3f})")
    else:
        md.append(f"- **準度**: 兩者相當 (Q3={avg_q:.3f}, Br={avg_b:.3f})")
    if timing:
        md.append(f"- **速度**: Qwen3 RTF={tot_dur/max(tot_q,1e-9):.0f}x, Breeze RTF={tot_dur/max(tot_b,1e-9):.0f}x")
    md.append(f"- **幻覺失控**: Qwen3={q_long}, Breeze={b_long}")

    out_p.write_text("\n".join(md), encoding="utf-8")
    print(f"\n Markdown → {out_p}")


if __name__ == "__main__":
    main()
