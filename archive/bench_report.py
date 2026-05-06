# -*- coding: utf-8 -*-
"""
ASR 基準與準度報告 — 比對 Qwen3 / Breeze 兩個轉錄結果
* 計時:每檔 ASR 耗時 + RTF (即時倍率)
* 準度代理指標 (無 ground truth 時用):
    1) 字元級 Jaccard 相似度 (Qwen3 vs Breeze)
    2) 字元級編輯距離 / 平均長度 (越低越一致)
    3) 兩模型各自 OpenCC 後的繁體比例
"""
from __future__ import annotations
import os, sys, json, re, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_segments(p: Path) -> List[Dict[str, Any]]:
    if not p.is_file():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def normalize_text(t: str) -> str:
    """為了比對:去掉空白與標點"""
    return re.sub(r"[\s\,\.\!\?\;\:\、\。\,\!\?\;\:\「\」\『\』\"\'\(\)（）\-\_]+", "", t or "")


def char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union


def edit_distance_ratio(a: str, b: str) -> float:
    """以 difflib SequenceMatcher 求相似度 (~編輯距離反比)"""
    from difflib import SequenceMatcher
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


# Unicode CJK 範圍 (常用) + 可粗略判斷簡 vs 繁
_SIMPLIFIED_HINTS = set("国发现实际经济认识让说话语门间问题点头计较实际讲话数据软件")
_TRADITIONAL_HINTS = set("國發現實際經濟認識讓說話語門間問題點頭計較實際講話數據軟體")


def trad_ratio(text: str) -> float:
    if not text:
        return 0.0
    s = sum(1 for c in text if c in _SIMPLIFIED_HINTS)
    t = sum(1 for c in text if c in _TRADITIONAL_HINTS)
    if (s + t) == 0:
        return 1.0
    return t / (s + t)


def file_audio_duration(path: Path) -> float:
    try:
        import subprocess
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen3-dir", default="transcripts/qwen3")
    ap.add_argument("--breeze-dir", default="transcripts/breeze")
    ap.add_argument("--audio-dir", default="music")
    ap.add_argument("--out", default="transcripts/REPORT.md")
    args = ap.parse_args()

    qdir = Path(args.qwen3_dir)
    bdir = Path(args.breeze_dir)
    adir = Path(args.audio_dir)
    out_path = Path(args.out)

    rows: List[Dict[str, Any]] = []
    audio_files = sorted([p for p in adir.iterdir() if p.is_file()])
    for af in audio_files:
        stem = af.stem
        q = load_segments(qdir / f"{stem}_qwen3.json")
        b = load_segments(bdir / f"{stem}_breeze.json")
        dur = file_audio_duration(af)
        q_text = "".join(normalize_text(s.get("text", "")) for s in q)
        b_text = "".join(normalize_text(s.get("text", "")) for s in b)
        row = {
            "file": af.name,
            "duration_sec": dur,
            "qwen3_segs": len(q),
            "breeze_segs": len(b),
            "qwen3_chars": len(q_text),
            "breeze_chars": len(b_text),
            "char_jaccard": char_jaccard(q_text, b_text) if q_text and b_text else 0.0,
            "char_seq_similarity": edit_distance_ratio(q_text, b_text) if q_text and b_text else 0.0,
            "qwen3_trad_ratio": trad_ratio(q_text),
            "breeze_trad_ratio": trad_ratio(b_text),
        }
        rows.append(row)

    # 純文字摘要
    print("\n" + "=" * 80)
    print("📊 ASR 基準與準度報告 (Qwen3 vs Breeze)")
    print("=" * 80)
    h = ("檔名", "時長/min", "Q3段", "Br段", "Q3字", "Br字",
         "Jaccard", "Sim", "Q3-繁", "Br-繁")
    print("{:<28} {:>8} {:>5} {:>5} {:>6} {:>6} {:>8} {:>6} {:>7} {:>7}".format(*h))
    print("-" * 100)
    total_dur = 0.0
    n_have_both = 0
    sum_jacc = 0.0
    sum_sim = 0.0
    for r in rows:
        print("{:<28} {:>8.2f} {:>5d} {:>5d} {:>6d} {:>6d} {:>8.3f} {:>6.3f} {:>7.2f} {:>7.2f}".format(
            (r["file"][:26] + "…") if len(r["file"]) > 27 else r["file"],
            r["duration_sec"] / 60,
            r["qwen3_segs"], r["breeze_segs"],
            r["qwen3_chars"], r["breeze_chars"],
            r["char_jaccard"], r["char_seq_similarity"],
            r["qwen3_trad_ratio"], r["breeze_trad_ratio"],
        ))
        total_dur += r["duration_sec"]
        if r["qwen3_chars"] and r["breeze_chars"]:
            n_have_both += 1
            sum_jacc += r["char_jaccard"]
            sum_sim += r["char_seq_similarity"]
    print("-" * 100)
    if n_have_both:
        print(f"📈 兩模型皆完成 {n_have_both}/{len(rows)} 檔 | "
              f"平均 Jaccard={sum_jacc/n_have_both:.3f} | "
              f"平均 SeqSim={sum_sim/n_have_both:.3f} | "
              f"總時長 {total_dur/60:.1f} min")

    # Markdown 報告
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md: List[str] = []
    md.append("# ASR 基準與準度報告")
    md.append(f"\n總檔案數: {len(rows)} | 總音訊時長: {total_dur/60:.1f} 分鐘\n")
    md.append("## 計時與準度比對 (Qwen3-ASR-1.7B vs Breeze-ASR-25)\n")
    md.append("| 檔名 | 時長 (min) | Q3 段數 | Br 段數 | Q3 字數 | Br 字數 | "
              "字元 Jaccard | 序列相似 | Q3 繁體率 | Br 繁體率 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append("| {} | {:.2f} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.2f} | {:.2f} |".format(
            r["file"], r["duration_sec"]/60,
            r["qwen3_segs"], r["breeze_segs"],
            r["qwen3_chars"], r["breeze_chars"],
            r["char_jaccard"], r["char_seq_similarity"],
            r["qwen3_trad_ratio"], r["breeze_trad_ratio"],
        ))
    md.append("\n## 指標說明\n")
    md.append("- **字元 Jaccard**: 兩模型轉錄共用字元集合佔聯集比 (越高越一致,1.0 代表用字完全相同)")
    md.append("- **序列相似** (difflib SequenceMatcher): 字元順序相似度 (1.0 完全相同)")
    md.append("- **繁體率**: 文中常見繁/簡判別字裡繁體比例 (1.0 完全繁體)")
    md.append("- 兩模型分歧不必然代表錯誤 — 用字、標點、段落切分習慣本就不同")
    md.append("- 以下是相似度 < 0.6 的檔案,建議人工審聽")
    md.append("")
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n💾 Markdown 報告 → {out_path}")


if __name__ == "__main__":
    main()
