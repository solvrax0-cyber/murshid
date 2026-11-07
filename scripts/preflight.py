"""Pre-demo check (run ~2 minutes before presenting). Prints ✅/❌ per item and warms the answer cache.

Usage:  python -m scripts.preflight            # checks + warm the demo questions
        python -m scripts.preflight --no-llm   # skip live LLM calls (index / offline checks only)
Exit code 0 = everything green. Never prints keys or the admin token."""
from __future__ import annotations

import argparse
import sys
import time
import uuid

from app import config, llm, store
from app.index import HybridIndex
from app.rag import RAGPipeline

# Demo script order (docs/demo_script.md). (question, expect_refusal, follow_up_of_previous)
DEMO = [
    ("نجحت بالبرمجة الكينونية وأنا بسنة ثانية، شو المواد اللي بقدر أسجلها الفصل الجاي؟", False, False),  # semester planning
    ("معدلي 1.8، كم ساعة أقدر أسجل؟", False, False),                                                   # GPA
    ("وإذا كان معدلي أعلى من 3؟", False, True),                                                         # follow-up on GPA
    ("ما المتطلب السابق لمادة الخوارزميات؟", False, False),                                              # table
    ("حتى أي أسبوع يمكنني الانسحاب من مساق؟", False, False),                                             # 2024 guide vs 2025 regulation
    ("متى امتحان الخوارزميات؟", True, False),                                                           # must refuse
]

ok_all = True


def check(name: str, ok: bool, detail: str = "") -> bool:
    global ok_all
    ok_all &= ok
    print(f"{'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    store.init()

    check("LLM key configured", llm.available(), f"{len(config.LLM_API_KEYS)} key(s)")
    check("Admin token configured", bool(config.ADMIN_TOKEN))
    idx = HybridIndex.load()
    if not idx.chunks:
        print("   (no index found -> building)")
        idx = HybridIndex.build(config.source_folders())
    sources = sorted({c.source for c in idx.chunks})
    check("Index has chunks", len(idx.chunks) > 0, f"{len(idx.chunks)} chunks from {len(sources)} document(s)")
    check("Dense search (dense=true)", idx.has_dense, config.EMBEDDING_MODEL or "disabled")

    if a.no_llm or not llm.available():
        print("   (skipping live LLM checks)")
        return 0 if ok_all else 1

    t0 = time.time()
    text, usage = llm.chat([{"role": "user", "content": "أجب بكلمة واحدة: نعم"}], max_tokens=60)
    check("LLM answers", bool(text), f"active model: {usage.get('model', '-')}, {time.time() - t0:.1f}s" if text
          else f"all models failed ({usage.get('error')}) -> UI will run in fallback mode")

    pipe = RAGPipeline(idx)
    sid = None
    for q, must_refuse, follow in DEMO:
        sid = sid if follow else "preflight-" + uuid.uuid4().hex[:8]
        t0 = time.time()
        ans = pipe.answer(q, sid)
        src = ans.citations[0]["label"] if ans.citations else "-"
        if must_refuse:
            good = ans.refused
            detail = "refused" if good else "NOT refused: " + ans.answer[:80]
        else:
            good = (not ans.refused) and bool(ans.citations) and ans.mode == "llm"
            detail = f"[{ans.mode}{', cached' if ans.usage.get('cached') else ''}] {src}"
        check(f"Demo: {q[:45]}", good, f"{detail} · {time.time() - t0:.1f}s")
        if not must_refuse:
            print("     ↳", ans.answer.replace("\n", " ")[:160])
    print("\nREADY ✅" if ok_all else "\nNOT READY ❌ — fix the red items (or present in fallback mode)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
