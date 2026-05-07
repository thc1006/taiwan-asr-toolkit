# -*- coding: utf-8 -*-
"""
中文 CER / WER 計算 — 用 jiwer 做底層 edit-distance,前面套中文正規化:
  1. 全形/半形標點移除
  2. 空白移除
  3. 數字保留 (語音 ASR 數字也算 char)
  4. OpenCC s2twp 簡→繁 (避免簡繁差異被計為錯誤)
  5. 大小寫不敏感 (英文)

CER = (Substitutions + Deletions + Insertions) / 參考文字字元數
"""
from __future__ import annotations
import re
import unicodedata
from typing import Dict, Any, Optional


_PUNCT_RE = re.compile(
    r"[\s,\.!?;:、。,!?;:「」『』\"'()（）\-—_…⋯‧\*#@/\\\[\]\{\}<>《》〈〉【】~`]+"
)
# `⋯` U+22EF MIDLINE HORIZONTAL ELLIPSIS, `‧` U+2027 HYPHENATION POINT —
# common in Apple Voice Memos transcripts and Chinese typography. Without
# them in the strip set, CER between two ASR outputs that only differ in
# ellipsis style is non-zero. Treat all ellipsis variants the same.


_S2TW = None  # 延後載入避免測試開銷


def _get_s2tw():
    global _S2TW
    if _S2TW is not None:
        return _S2TW
    try:
        from opencc import OpenCC
        for scheme in ("s2twp", "s2twp.json", "s2tw", "s2tw.json"):
            try:
                _S2TW = OpenCC(scheme)
                return _S2TW
            except Exception:
                continue
    except Exception:
        pass
    _S2TW = False  # mark as unavailable
    return _S2TW


def normalize(text: str) -> str:
    """中文 ASR 比對前的正規化:
      * Unicode NFKC (全形數字/英文 → 半形)
      * 移除標點與空白
      * 簡→繁 (s2twp,若 OpenCC 可用)
      * 英文小寫
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = _PUNCT_RE.sub("", s)
    s = s.lower()
    cc = _get_s2tw()
    if cc:
        try:
            s = cc.convert(s)
        except Exception:
            pass
    return s


def compute_cer(reference: str, hypothesis: str) -> float:
    """字元錯誤率 (Character Error Rate)。"""
    import jiwer
    ref_n = normalize(reference)
    hyp_n = normalize(hypothesis)
    if not ref_n and not hyp_n:
        return 0.0
    if not ref_n:
        # 空 ref 不能算 CER (除零),回傳 hyp 長度作 proxy
        return float(len(hyp_n))
    # jiwer.cer 接受字串,內部 split 成 char
    return float(jiwer.cer(ref_n, hyp_n))


def compute_metrics(reference: str, hypothesis: str) -> Dict[str, Any]:
    """回傳細節 dict: cer, wer, hits, substitutions, deletions, insertions, ref_chars。"""
    import jiwer
    ref_n = normalize(reference)
    hyp_n = normalize(hypothesis)
    if not ref_n:
        return {
            "cer": float(len(hyp_n)),
            "wer": float(len(hyp_n.split())),
            "hits": 0, "substitutions": 0,
            "deletions": 0, "insertions": len(hyp_n),
            "ref_chars": 0, "hyp_chars": len(hyp_n),
        }
    # process_words 給 word-level (中文一字當一詞時可以加空格)
    # 這裡我們用 char-level: 把每個字元當一個「token」
    ref_chars = list(ref_n)
    hyp_chars = list(hyp_n)
    ref_str = " ".join(ref_chars)
    hyp_str = " ".join(hyp_chars)

    out = jiwer.process_words(ref_str, hyp_str)
    # NOTE (L3 v0.5.5): "wer" key is intentionally identical to "cer" — we
    # treat each Chinese character as one word, so jiwer's word-level WER
    # is char-level CER. Returned in both keys for backward compatibility,
    # but downstream code SHOULD prefer "cer" because Mandarin has no word
    # segmentation; "wer" alone would be meaningless.
    return {
        "cer": float(out.wer),
        "wer": float(out.wer),
        "hits": int(out.hits),
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "ref_chars": len(ref_chars),
        "hyp_chars": len(hyp_chars),
    }


def read_text_file(path) -> str:
    """讀文字檔,跳過 # 開頭的註解行 (適用於 ref / GT 檔)。"""
    from pathlib import Path
    raw = Path(path).read_text(encoding="utf-8")
    lines = []
    for ln in raw.splitlines():
        if ln.lstrip().startswith("#"):
            continue
        lines.append(ln)
    return "\n".join(lines)


def hyp_text_from_path(path, clip_start: float = 0.0,
                       clip_end: Optional[float] = None) -> str:
    """從 .txt 或 .json 取出 hypothesis 文字。.json 可依時戳剪裁。"""
    import json
    from pathlib import Path
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        parts = []
        for s in data:
            st = float(s.get("start", 0.0) or 0.0)
            en = float(s.get("end", 0.0) or 0.0)
            # 只要段有任何部分落在 [clip_start, clip_end] 內就計入
            if clip_end is not None and st >= clip_end:
                continue
            if en <= clip_start:
                continue
            parts.append(s.get("text", ""))
        return "".join(parts)
    # plain text 不支援剪裁
    text = p.read_text(encoding="utf-8")
    return text


def main():
    import argparse, json
    from pathlib import Path
    ap = argparse.ArgumentParser(description="中文 ASR CER/WER 評估")
    ap.add_argument("--ref", required=True, help="ground-truth 文字檔 (.txt)")
    ap.add_argument("--hyp", required=True,
                    help="ASR 輸出 (.txt 或 .json — JSON 自動拼接 segments[].text)")
    ap.add_argument("--clip-start", type=float, default=0.0,
                    help="只比對 hyp 中 start>=此秒數的段")
    ap.add_argument("--clip-end", type=float, default=None,
                    help="只比對 hyp 中 start<此秒數的段")
    ap.add_argument("--json", action="store_true", help="輸出為 JSON")
    args = ap.parse_args()

    ref_text = read_text_file(args.ref)
    hyp_text = hyp_text_from_path(args.hyp, args.clip_start, args.clip_end)

    m = compute_metrics(ref_text, hyp_text)
    if args.json:
        print(json.dumps(m, ensure_ascii=False, indent=2))
    else:
        print(f" CER: {m['cer']*100:.2f}%  ({m['ref_chars']} ref chars, {m['hyp_chars']} hyp chars)")
        print(f" Sub: {m['substitutions']}  Del: {m['deletions']}  Ins: {m['insertions']}  Hits: {m['hits']}")


if __name__ == "__main__":
    main()
