"""The RAG pipeline: understand -> retrieve -> gate -> generate -> verify citations -> log."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import config, llm, prompts, store
from .arabic import detect_lang, dialect_to_msa, expand_query, normalize, tokenize
from .index import HybridIndex, Hit

GREETING_RX = re.compile(r"^\W*(مرحبا|مرحباً|اهلا|أهلا|أهلاً|السلام عليكم|سلام|هاي|hi|hello|صباح الخير|مساء الخير|شكرا|شكراً|يعطيك العافيه|يعطيك العافية|تسلم)\W*$", re.I)
OOS_RX = re.compile(r"(قصيد|شعر|نكت|نكتة|اغني|أغني|طبخ|وصفة|مباراة|الطقس|اكتب لي (قصة|مقال|كود|برنامج)|ترجم|سياس)")
FOLLOWUP_RX = re.compile(r"^\s*(و\s*(إذا|اذا|لو|ماذا|شو|كيف|هل|متى|كم)|طيب|طب|وبالنسبة|وفي حال|وإذا|واذا|ولو|وماذا عن|وكم|what about|and if|and what|how about)", re.I)
CITE_RX = re.compile(r"\[(\d{1,2})\]")


def is_refusal(text: str) -> bool:
    return "لا أجد هذه المعلومة" in text or "can't find this information" in text.replace("’", "'")


@dataclass
class Answer:
    answer: str
    refused: bool
    intent: str
    rewritten: str
    citations: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    grounded: bool = True
    mode: str = "llm"
    latency_ms: int = 0
    log_id: int | None = None
    usage: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, index: HybridIndex):
        self.index = index
        self.document_openings = {}
        for chunk in index.chunks:
            if chunk.source.lower().endswith(".pdf") and chunk.page == 1 and chunk.chunk_type == "text":
                self.document_openings.setdefault(chunk.source, chunk)

    # ------------------------------------------------------------ 1. understand
    def understand(self, question: str, hist: list[dict]) -> tuple[str, str, list[str]]:
        if GREETING_RX.match(question):
            return "greeting", question, []
        if OOS_RX.search(question):
            return "out_of_scope", question, []
        if llm.available():
            h = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in hist[-config.HISTORY_TURNS * 2:]) or "(لا يوجد)"
            data = llm.chat_json([
                {"role": "system", "content": prompts.SYSTEM_REWRITE},
                {"role": "user", "content": prompts.USER_REWRITE.format(history=h, question=question)},
            ])
            if data and data.get("standalone"):
                kws = data.get("keywords") or []
                return data.get("intent", "academic"), str(data["standalone"]), [str(k) for k in kws][:8]
        # offline heuristic: dialect map + follow-up stitching
        standalone = dialect_to_msa(question)
        last_user = next((m["content"] for m in reversed(hist) if m["role"] == "user"), None)
        if last_user and (FOLLOWUP_RX.match(question) or len(question.split()) < 5):  # short or "and if…" -> follow-up
            standalone = f"{dialect_to_msa(last_user)} {standalone}"
        return "academic", standalone, []

    # ------------------------------------------------------------ 2-3. retrieve + gate
    def retrieve(self, question: str, standalone: str, keywords: list[str]) -> list[Hit]:
        queries = [standalone, expand_query(question), expand_query(standalone)]
        if keywords:
            queries.append(" ".join(keywords))
        hits = self.index.search(queries, k=config.TOP_K)
        # Include a matched PDF's opening alongside its detailed passages. Plans
        # often state their total hours only on page 1, while later course tables
        # dominate the rankings. Keep separate citations with the correct page.
        seen = {hit.chunk.id for hit in hits}
        openings = []
        for hit in hits:
            opening = self.document_openings.get(hit.chunk.source)
            if opening and opening.id not in seen:
                openings.append(Hit(opening, rrf=0.0))
                seen.add(opening.id)
        return hits + openings

    def passes_gate(self, hits: list[Hit]) -> bool:
        if not hits:
            return False
        cov = max(h.coverage for h in hits)
        if self.index.has_dense:
            dense = max(h.dense for h in hits)
            return not (dense < config.MIN_DENSE_SCORE and cov < config.MIN_TERM_COVERAGE)
        return cov >= config.MIN_TERM_COVERAGE

    # ------------------------------------------------------------ 4. generate
    @staticmethod
    def build_context(hits: list[Hit]) -> str:
        blocks = []
        for n, h in enumerate(hits, 1):
            c = h.chunk
            meta = f"المصدر: {c.title}"
            if c.page:
                meta += f" | صفحة {c.page}"
            if c.article:
                meta += f" | المادة {c.article}"
            if c.version_year:
                meta += f" | سنة الإصدار {c.version_year}"
            blocks.append(f"[{n}] ({meta})\n{c.text}")
        return "\n\n".join(blocks)

    def generate(self, question: str, standalone: str, hits: list[Hit], hist: list[dict], lang: str = "ar") -> tuple[str, str, dict]:
        if llm.available():
            msgs = [{"role": "system", "content": prompts.SYSTEM_ANSWER}]
            for m in hist[-config.HISTORY_TURNS * 2:]:
                msgs.append({"role": m["role"], "content": m["content"][:600]})
            msgs.append({"role": "user", "content": prompts.USER_ANSWER.format(
                context=self.build_context(hits), question=question, rewritten=standalone, lang=lang)})
            # cache key = standalone question + retrieved chunk ids + model (repeats / eval re-runs cost no quota)
            key = llm.hash_key("answer", normalize(standalone), [h.chunk.id for h in hits], config.LLM_MODEL, lang,
                               llm.hash_key(prompts.SYSTEM_ANSWER, prompts.USER_ANSWER))  # prompt edits invalidate
            text, usage = llm.chat(msgs, cache_key=key)
            if text:
                return text, "llm", usage
        return self.extractive(standalone, hits), "offline", {}

    @staticmethod
    def extractive(query: str, hits: list[Hit]) -> str:
        """Offline fallback (no internet / no key): quote the best-matching sentences verbatim."""
        q = set(tokenize(query))
        # without an LLM to read all passages, quote the top-3 hit that covers most query terms
        # (RRF rank alone may put a related article above the table row that actually answers)
        top = hits[:3]
        j = max(range(len(top)), key=lambda i: (round(top[i].coverage, 2), -i))
        if top[j].coverage - top[0].coverage < 0.2:  # only override the fused ranking on a clear margin
            j = 0
        best = top[j]
        is_table = best.chunk.chunk_type == "table"
        sents = [s.strip() for s in re.split(r"(?<=[\.\!\؟\?\n؛])\s+|\n", best.chunk.text) if s.strip() and s.strip() != "جدول:" and not s.strip().startswith("#")]
        if not is_table and len(best.chunk.text) <= 450:  # short self-contained rule: quote it whole
            body = "\n".join(sents)
            return f"حسب المستند: {body} [{j + 1}]\n\n_(وضع دون اتصال: مقتطف مباشر من المستند دون صياغة)_"
        toks = [set(tokenize(s)) for s in sents]
        df = {t: sum(t in ts for ts in toks) for t in q}
        # rare query terms (e.g. the course name) matter more than terms shared by every row
        score = [sum(1 / df[t] for t in q & ts if df[t]) for ts in toks]
        n_pick = 1 if is_table else 3
        ranked = sorted(range(len(sents)), key=lambda i: -score[i])[:n_pick]
        picked = sorted(i for i in ranked if score[i] > 0) or [0]
        body = "\n".join(sents[i] for i in picked)[:700]
        return f"حسب المستند: {body} [{j + 1}]\n\n_(وضع دون اتصال: مقتطف مباشر من المستند دون صياغة)_"

    # ------------------------------------------------------------ 5. main entry
    def answer(self, question: str, session_id: str = "cli") -> Answer:
        t0 = time.time()
        question = question.strip()[:1000]
        hist = store.history(session_id)
        lang = detect_lang(question)
        intent, standalone, keywords = self.understand(question, hist)

        if intent in ("greeting", "out_of_scope"):
            en = lang == "en"
            text = (prompts.GREETING_REPLY_EN if en else prompts.GREETING_REPLY) if intent == "greeting" else                 (prompts.OUT_OF_SCOPE_REPLY_EN if en else prompts.OUT_OF_SCOPE_REPLY)
            ans = Answer(text, refused=intent == "out_of_scope", intent=intent, rewritten=standalone, mode="rule")
        else:
            hits = self.retrieve(question, standalone, keywords)
            retrieved = [{
                "rank": n, "label": h.chunk.label(), "source": h.chunk.source, "page": h.chunk.page,
                "article": h.chunk.article, "bm25": round(h.bm25, 2), "dense": round(h.dense, 3),
                "rerank": None if h.rerank is None else round(h.rerank, 3), "coverage": round(h.coverage, 2),
            } for n, h in enumerate(hits, 1)]
            if not self.passes_gate(hits):
                text = config.REFUSAL_TEXT_EN if lang == "en" else f"{config.REFUSAL_TEXT} يرجى مراجعة {config.REFERRAL_OFFICE}."
                ans = Answer(text, True, "academic", standalone, retrieved=retrieved, mode="gate")
            else:
                text, mode, usage = self.generate(question, standalone, hits, hist, lang)
                cited = sorted({int(n) for n in CITE_RX.findall(text) if 1 <= int(n) <= len(hits)})
                # a cited partial answer that also says "the rest is not in the documents" is still an answer
                refused = is_refusal(text) and not cited
                citations = [] if refused else [{
                    "n": n, "label": hits[n - 1].chunk.label(), "title": hits[n - 1].chunk.title,
                    "source": hits[n - 1].chunk.source, "page": hits[n - 1].chunk.page,
                    "article": hits[n - 1].chunk.article, "year": hits[n - 1].chunk.version_year,
                    "snippet": hits[n - 1].chunk.text[:500], "chunk_id": hits[n - 1].chunk.id,
                } for n in cited]
                ans = Answer(text, refused, "academic", standalone, citations, retrieved,
                             grounded=refused or bool(citations), mode=mode, usage=usage)

        ans.latency_ms = int((time.time() - t0) * 1000)
        store.add_message(session_id, "user", question)
        store.add_message(session_id, "assistant", ans.answer)
        top = ans.citations[0]["source"] if ans.citations else None
        ans.log_id = store.log_question(session_id, question, ans.rewritten, ans.intent, ans.refused, top, ans.latency_ms)
        return ans
