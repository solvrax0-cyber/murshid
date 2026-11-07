"""Evaluation harness -> eval/results/report.md (paste the summary table into your slides).

Metrics
  * Hit@k        : at least one expected source (doc + page/article) in the top-k chunks
  * Precision@k  : relevant chunks among top-k / k
  * MRR          : 1 / rank of first relevant chunk
  * Refusal acc. : out-of-docs questions refused  &  answerable questions NOT refused
  * Citation rate: answered questions with >=1 valid [n] citation
  * Faithfulness : LLM-judge share of answer claims supported by retrieved chunks
                   (offline proxy: share of answer tokens found in cited chunks)
  * Correctness  : LLM-judge vs reference answer (1 / 0.5 / 0)   (offline proxy: token F1)

Question file (JSONL), one object per line:
  {"id": "q1", "type": "direct|dialect|table|multi_doc|follow_up|not_in_docs|out_of_scope",
   "question": "...", "reference": "...", "sources": [{"doc": "لائحة", "page": 4, "article": "14"}],
   "should_refuse": false, "follow_up_of": "q5"}
Usage: python -m eval.run_eval --questions eval/questions.sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from app import config, llm, prompts, store
from app.arabic import normalize, tokenize
from app.index import HybridIndex
from app.rag import RAGPipeline

OUT = Path(__file__).parent / "results"
JUDGE_USED = {"llm": 0, "proxy": 0}  # how many scores really came from the LLM judge


def is_relevant(hit: dict, sources: list[dict]) -> bool:
    for s in sources:
        if s.get("doc") and normalize(s["doc"]) not in normalize(hit["source"]):
            continue
        if s.get("article") and str(s["article"]) != str(hit.get("article")):
            continue
        if s.get("page") and hit.get("page") and int(s["page"]) != int(hit["page"]):
            continue
        return True
    return False


def token_f1(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    common = sum(min(ta.count(t), tb.count(t)) for t in set(ta))
    if not common:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def faithfulness(answer: str, context: str) -> float:
    if llm.available():
        d = llm.chat_json([{"role": "user", "content": prompts.JUDGE_FAITHFULNESS.format(context=context, answer=answer)}], model=config.JUDGE_MODEL or None)
        if d and d.get("claims"):
            JUDGE_USED["llm"] += 1
            return round(min(1.0, d.get("supported", 0) / max(1, d["claims"])), 2)
    JUDGE_USED["proxy"] += 1
    at, ct = tokenize(answer), set(tokenize(context))
    return round(sum(t in ct for t in at) / len(at), 2) if at else 0.0


def correctness(q: str, ref: str, answer: str) -> float:
    if llm.available():
        d = llm.chat_json([{"role": "user", "content": prompts.JUDGE_CORRECTNESS.format(question=q, reference=ref, answer=answer)}], model=config.JUDGE_MODEL or None)
        if d and "score" in d:
            JUDGE_USED["llm"] += 1
            return float(d["score"])
    JUDGE_USED["proxy"] += 1
    return round(token_f1(answer, ref), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(Path(__file__).parent / "questions.sample.jsonl"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tag", default="", help="label for this run, e.g. 'chunk800-e5small'")
    ap.add_argument("--sleep", type=float, default=0, help="seconds between questions (free-tier rate limits)")
    ap.add_argument("--patient", action="store_true", help="long fallback deadline (240s) so free-tier 429/503 are waited out instead of degrading to offline")
    a = ap.parse_args()
    if a.patient:
        config.LLM_DEADLINE, config.LLM_RETRY_WAIT, config.LLM_TIMEOUT = 240, 6, 30  # wait out quota windows

    store.init()
    pipe = RAGPipeline(HybridIndex.load())
    full_text = {c.id: c.text for c in pipe.index.chunks}  # judge sees the FULL cited chunk (UI snippet is cut at 500 chars)
    qs = [json.loads(l) for l in Path(a.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    sessions: dict[str, str] = {}
    rows = []
    for q in qs:
        sid = sessions.get(q.get("follow_up_of"), uuid.uuid4().hex)
        sessions[q["id"]] = sid
        if a.sleep and rows:
            time.sleep(a.sleep)
        t0 = time.time()
        llm.usage_reset()
        ans = pipe.answer(q["question"], sid)
        tok = llm.usage_total()  # rewrite + answer calls of this question (judge calls come later)
        retrieved = ans.retrieved[: a.k]
        should_refuse = bool(q.get("should_refuse"))
        row = {"id": q["id"], "type": q.get("type", ""), "question": q["question"], "answer": ans.answer,
               "refused": ans.refused, "should_refuse": should_refuse, "mode": ans.mode,
               "latency_ms": int((time.time() - t0) * 1000), "top": retrieved[0]["label"] if retrieved else "-",
               "top_dense": retrieved[0]["dense"] if retrieved else 0,
               "in_tokens": tok["prompt_tokens"], "out_tokens": tok["completion_tokens"], "llm_calls": tok["calls"],
               "model": ans.usage.get("model"), "top_cov": retrieved[0]["coverage"] if retrieved else 0}
        row["refusal_ok"] = ans.refused == should_refuse
        if not should_refuse and q.get("sources"):
            rel = [is_relevant(h, q["sources"]) for h in retrieved]
            row["hit"] = any(rel)
            row["precision"] = round(sum(rel) / a.k, 2)
            row["rr"] = round(1 / (rel.index(True) + 1), 2) if any(rel) else 0.0
        if not should_refuse and not ans.refused:
            ctx = "\n\n".join(full_text.get(c.get("chunk_id"), c["snippet"]) for c in ans.citations) or "\n".join(h["label"] for h in retrieved)
            row["cited"] = bool(ans.citations)
            row["faithfulness"] = faithfulness(ans.answer, ctx)
            if q.get("reference"):
                row["correctness"] = correctness(q["question"], q["reference"], ans.answer)
        rows.append(row)
        flag = "✔" if row["refusal_ok"] and row.get("hit", True) else "✘"
        print(f"{flag} {q['id']:>4} [{row['type']}] hit={row.get('hit','-')} refused={ans.refused} | {q['question'][:50]}")

    def avg(key, subset=None):
        vals = [r[key] for r in (rows if subset is None else subset) if key in r]
        return round(sum(vals) / len(vals), 3) if vals else None

    answerable = [r for r in rows if not r["should_refuse"]]
    refusable = [r for r in rows if r["should_refuse"]]
    summary = {
        "run": a.tag or time.strftime("%Y-%m-%d %H:%M"), "n_questions": len(rows),
        f"hit@{a.k}": avg("hit"), f"precision@{a.k}": avg("precision"), "mrr": avg("rr"),
        "refusal_recall (should refuse -> refused)": avg("refusal_ok", refusable),
        "false_refusal_rate": round(sum(r["refused"] for r in answerable) / len(answerable), 3) if answerable else None,
        "citation_rate": avg("cited"), "faithfulness": avg("faithfulness"), "correctness": avg("correctness"),
        "passed": f"{sum(r['refusal_ok'] and r.get('hit', True) and r.get('correctness', 1) >= 0.5 for r in rows)}/{len(rows)}",
        "llm": (f"answer={config.LLM_MODEL}, judge={config.JUDGE_MODEL or config.LLM_MODEL}" if llm.available() else "judge=offline proxy"),
        # honesty check: answers produced by the LLM vs offline fallback, and judge scores by source
        "answers_by_mode": {m: sum(r["mode"] == m for r in rows) for m in sorted({r["mode"] for r in rows})},
        # valid LLM run = no answer fell back to the offline quote and every judge score came from the LLM
        "all_llm": llm.available() and not any(r["mode"] == "offline" for r in rows) and JUDGE_USED["proxy"] == 0,
        "avg_in_tokens_per_llm_answer": avg("in_tokens", [r for r in rows if r["mode"] == "llm"]),
        "avg_out_tokens_per_llm_answer": avg("out_tokens", [r for r in rows if r["mode"] == "llm"]),
        "avg_latency_ms": avg("latency_ms"),
        "judge_scores_llm/proxy": f"{JUDGE_USED['llm']}/{JUDGE_USED['proxy']}",
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [f"# تقرير تقييم مُرشِد — {summary['run']}\n", "| المقياس | القيمة |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in summary.items()]
    md += ["\n## النتائج حسب نوع السؤال\n", "| النوع | العدد | نجح |", "|---|---|---|"]
    for t in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == t]
        ok = sum(r["refusal_ok"] and r.get("hit", True) for r in sub)
        md.append(f"| {t} | {len(sub)} | {ok} |")
    md += ["\n## الأسئلة التي فشلت (حلّلوا السبب وحسّنوا)\n", "| # | السؤال | المشكلة | أعلى مقطع | dense | coverage |", "|---|---|---|---|---|---|"]
    for r in rows:
        problems = []
        if not r["refusal_ok"]:
            problems.append("رفض خاطئ" if r["refused"] else "لم يرفض (هلوسة محتملة)")
        if r.get("hit") is False:
            problems.append("الاسترجاع لم يجد المصدر")
        if r.get("correctness", 1) < 0.5:
            problems.append("إجابة غير صحيحة")
        if problems:
            md.append(f"| {r['id']} | {r['question'][:60]} | {'، '.join(problems)} | {r['top'][:50]} | {r['top_dense']} | {r['top_cov']} |")
    md += ["\n## معايرة عتبة الرفض\n",
           "قارنوا قيم dense/coverage لأعلى مقطع في الأسئلة التي يجب رفضها مقابل الأسئلة القابلة للإجابة، "
           "واضبطوا MIN_DENSE_SCORE و MIN_TERM_COVERAGE في ملف .env بحيث تفصل بينهما.\n",
           "| # | يجب الرفض؟ | dense | coverage |", "|---|---|---|---|"]
    md += [f"| {r['id']} | {'نعم' if r['should_refuse'] else 'لا'} | {r['top_dense']} | {r['top_cov']} |" for r in rows]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nreport -> {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
