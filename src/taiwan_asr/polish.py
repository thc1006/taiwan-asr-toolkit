# -*- coding: utf-8 -*-
"""
LLM 上下文修正 — 用 Qwen3 LLM 把 ASR 結果做語境級錯字校正 + 標點補完。

核心想法:
  ASR 是「聲學模型」,出錯多半是「同音錯字」(登機/登記、軟件/軟體、思維/順位…)。
  把 ASR 結果丟給 LLM,給它周圍 N 段做上下文,LLM 看得出來「這句話應該是…」。

運作流程:
  輸入: transcripts/{model}/{name}_{model}.json   (一份 ASR 結果)
  輸出: transcripts/{model}-polished/{name}_{model}.{txt,srt,json}

特性:
  * 預設使用 Qwen/Qwen2.5-7B-Instruct (穩定可靠;~14GB bf16)
  * 也可改用 Qwen/Qwen2.5-3B-Instruct (--model 旗標) 換速度
  * 滑窗策略:每次給 LLM N 段 (預設 5),含前後 1 段做 context
  * 強制 JSON 輸出 + schema 驗證,LLM 偏離則自動 fallback 原文
  * 永遠保持段數一致 (不合併/拆分)
  * 永遠保持時間戳記不變
  * OpenCC s2twp 兜底
"""
from __future__ import annotations

from taiwan_asr.common import init_env
_NCPU = init_env()

import os, sys, gc, time, json, re, argparse, warnings
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple

warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from taiwan_asr.common import init_torch, Segment, save_outputs, S2TW, Stopwatch
init_torch(_NCPU)


SYSTEM_PROMPT = """你是專業的台灣中文 ASR 後處理助手。你的任務是把語音轉錄(可能含同音錯字、漏標點)依據前後文做**最小幅度**修正。

**嚴格規則**:
1. 只修「明顯」的同音錯字。例:「登機嗎」(語境是登記櫃台)→「登記嗎」;「軟件」→「軟體」。
2. 補上自然的中文標點符號 (,。?!:「」)。
3. **保留所有專有名詞原樣** — 包含但不限於:
   - 人名、暱稱、職稱
   - 地名、建築物名、教室名
   - **宿舍/書院/系所名** (如「延三」「延平」「圓山宿舍」「住輔組」「祝福二組」「男一」「BOT 宿舍」等)
   - 學校、公司、組織、社團、課程、活動名稱
   - 產品、軟體、書籍、節目、品牌名
   即使你不認得這個名詞、即使聽起來像錯字,也**絕不可改**。
4. **遇到不確定的詞,一律保留原樣**,寧可漏修也不要誤改。
5. 嚴禁改變原意、改寫語氣、刪除任何內容。
6. 嚴禁合併或拆分段落 — 段數必須一致。
7. 嚴禁加入解釋或注釋。
8. 必須輸出繁體中文 (台灣慣用詞: 軟體、雷射、檔案、影片、滑鼠…)。
9. 輸出必須是合法 JSON,格式: {"fixed": ["段1修正","段2修正",...]}, 順序對應輸入。

如果原句已通順,直接照抄。寧可保守也不可激進。"""


USER_TEMPLATE = """以下是 ASR 轉錄的 {n} 個連續段落(原始輸入,可能含錯字):

{numbered_segments}{glossary_block}

請依規則輸出 JSON,鍵為 "fixed",值為長度 {n} 的陣列,順序對應上面 1~{n} 段。"""


def _format_glossary(terms: List[str]) -> str:
    if not terms:
        return ""
    bullet = "\n".join(f"  - {t}" for t in terms)
    return f"\n\n**必須完整保留的專有名詞** (出現時不可改寫,即使聽起來像錯字):\n{bullet}"


def _try_extract_json(text: str) -> Optional[Dict[str, Any]]:
    """從 LLM 輸出抓出第一個合法 JSON object。"""
    if not text:
        return None
    # 嘗試直接 parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # 抓第一個 {...} 區塊 (非貪婪、跨行)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    raw = m.group(0)
    # 處理 trailing commas
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
    try:
        return json.loads(raw)
    except Exception:
        return None


class Qwen3Polisher:
    DEFAULT_MODELS = [
        "Qwen/Qwen3-8B",                  # 本機已快取,優先
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
    ]

    def __init__(self, model_id: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: str = "cuda:0",
                 glossary: Optional[List[str]] = None):
        self.model_id = model_id or self.DEFAULT_MODELS[0]
        self.dtype = dtype or torch.bfloat16
        self.device = device
        self.tok = None
        self.model = None
        self.s2tw = S2TW(True)
        # 預設保護詞彙 (台大常見) + 使用者自訂
        self.glossary: List[str] = list(dict.fromkeys(
            (glossary or []) + [
                # 台大宿舍 / 學生事務常見專有名詞
                "研三", "研三舍", "研一", "研一舍", "研二", "研二舍",
                "延三", "延三舍", "延平", "延平學舍",
                "男一", "男二", "男三", "男四", "男五", "男六", "男七", "男八",
                "女一", "女二", "女三", "女四", "女五", "女六", "女七", "女八", "女九",
                "圓山", "圓山宿舍", "BOT 宿舍", "BOT宿舍",
                "住輔組", "祝福二組", "課指組", "課務組",
                "台大", "臺大", "NTU",
            ]
        ))

    def load(self) -> "Qwen3Polisher":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        last_err = None
        for mid in [self.model_id] + [m for m in self.DEFAULT_MODELS if m != self.model_id]:
            try:
                print(f"⏳ 載入 LLM: {mid} (dtype={self.dtype})…")
                t0 = time.time()
                self.tok = AutoTokenizer.from_pretrained(mid)
                self.model = AutoModelForCausalLM.from_pretrained(
                    mid,
                    dtype=self.dtype,
                    device_map=self.device,
                    attn_implementation="sdpa",
                    low_cpu_mem_usage=True,
                )
                self.model.eval()
                self.model_id = mid
                print(f" LLM 就緒 ({time.time()-t0:.1f}s)")
                return self
            except Exception as e:
                print(f" {mid} 載入失敗: {type(e).__name__}: {str(e)[:120]}")
                last_err = e
                continue
        raise RuntimeError(f"所有候選 LLM 都載入失敗。最後一次錯誤: {last_err}")

    @torch.inference_mode()
    def polish_window(self, texts: List[str]) -> List[str]:
        """處理 N 段,回傳修正後 N 段。失敗則 passthrough。"""
        if not texts:
            return []
        n = len(texts)
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        # 只把該視窗中真的有提到的 glossary 字詞列入 prompt,避免 token 浪費
        joined = "".join(texts)
        active_glossary = [t for t in self.glossary if t in joined]
        user = USER_TEMPLATE.format(
            n=n, numbered_segments=numbered,
            glossary_block=_format_glossary(active_glossary),
        )
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ]
        prompt = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=min(2048, max(256, sum(len(t) for t in texts) * 3)),
            do_sample=False,
            num_beams=1,                 # greedy 夠用,LLM 校對任務確定性高
            repetition_penalty=1.05,
            pad_token_id=self.tok.eos_token_id,
        )
        gen = out[0, inputs["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        data = _try_extract_json(text)
        if not data or "fixed" not in data or not isinstance(data["fixed"], list):
            return texts  # passthrough
        fixed = data["fixed"]
        # schema 驗證:必須長度一致,內容是 str
        if len(fixed) != n or not all(isinstance(x, str) for x in fixed):
            return texts
        # OpenCC 兜底繁體
        return [self.s2tw(x.strip() or texts[i]) for i, x in enumerate(fixed)]

    def polish_segments(
        self,
        segments: List[Dict[str, Any]],
        window: int = 5,
        verbose: bool = True,
    ) -> Tuple[List[Segment], Stopwatch]:
        sw = Stopwatch()
        out: List[Segment] = []
        n = len(segments)
        if n == 0:
            sw.lap("polish")
            return [], sw
        i = 0
        done = 0
        t0 = time.perf_counter()
        while i < n:
            j = min(i + window, n)
            texts_in = [(s.get("text") or "").strip() for s in segments[i:j]]
            try:
                texts_out = self.polish_window(texts_in)
            except Exception as e:
                print(f"\n     視窗 {i}..{j} polish 失敗: {type(e).__name__}: {str(e)[:80]} → passthrough")
                texts_out = texts_in
            for k, txt in enumerate(texts_out):
                seg = segments[i + k]
                out.append(Segment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=txt,
                    language=seg.get("language", ""),
                    avg_logprob=float(seg.get("avg_logprob", 0.0) or 0.0),
                    no_speech_prob=float(seg.get("no_speech_prob", 0.0) or 0.0),
                    words=seg.get("words", None),
                ))
            done = j
            i = j
            if verbose:
                el = time.perf_counter() - t0
                pct = 100 * done / n
                print(f"\r     {done}/{n} ({pct:5.1f}%) | 耗 {el:6.1f}s", end="", flush=True)
        if verbose:
            print()
        sw.lap("polish")
        return out, sw


def main():
    ap = argparse.ArgumentParser(description="LLM 上下文修正 (Qwen2.5)")
    ap.add_argument("inputs", nargs="+", help="ASR 輸出 JSON (transcripts/qwen3/*.json 或 transcripts/breeze/*.json)")
    ap.add_argument("--out", default="", help="輸出目錄 (預設: 同層 + -polished)")
    ap.add_argument("--model", default=None, help="LLM 模型 (預設 Qwen2.5-7B-Instruct)")
    ap.add_argument("--window", type=int, default=5, help="每次 LLM 看幾段 (預設 5)")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--glossary", default="",
                    help="保護詞彙 (逗號分隔, e.g. --glossary '延三舍,祝福二組,延平學舍')")
    ap.add_argument("--glossary-file", default="",
                    help="保護詞彙檔 (每行一個詞)")
    args = ap.parse_args()

    extra_glossary: List[str] = []
    if args.glossary:
        extra_glossary += [t.strip() for t in args.glossary.split(",") if t.strip()]
    if args.glossary_file:
        gp = Path(args.glossary_file)
        if gp.is_file():
            extra_glossary += [
                ln.strip() for ln in gp.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")
            ]

    print("=" * 72)
    print(" LLM 上下文修正 (Qwen3 後處理)")
    print("=" * 72)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    polisher = Qwen3Polisher(model_id=args.model, dtype=dtype,
                             glossary=extra_glossary).load()
    print(f" 保護詞彙: {len(polisher.glossary)} 個 ({'自訂+內建' if extra_glossary else '內建'})")

    for src_json in args.inputs:
        p = Path(src_json)
        if not p.is_file():
            print(f" 找不到 {src_json}"); continue
        try:
            segs_in = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f" JSON 解析失敗 {src_json}: {e}"); continue

        # 推導輸出目錄: transcripts/qwen3/x.json → transcripts/qwen3-polished/
        if args.out:
            out_dir = Path(args.out)
        else:
            out_dir = p.parent.parent / (p.parent.name + "-polished")

        # 推導 base name (去掉 _qwen3 / _breeze 尾)
        stem = p.stem
        suffix_match = re.search(r"_(qwen3|breeze)$", stem)
        suffix = suffix_match.group(1) if suffix_match else "polished"
        base_for_out = stem[:suffix_match.start()] if suffix_match else stem

        # 重組 src 給 save_outputs (用 stem 作虛擬檔名)
        virtual_src = str(p.with_name(base_for_out))
        print("\n" + "─" * 72)
        print(f" {p}  →  {out_dir}/")
        out_segs, sw = polisher.polish_segments(segs_in, window=args.window)
        sw.report()
        # 保存
        save_outputs(out_segs, virtual_src, str(out_dir),
                     suffix=suffix + "-polished")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n 完成。")


if __name__ == "__main__":
    main()
