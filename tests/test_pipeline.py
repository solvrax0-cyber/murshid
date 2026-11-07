"""Run: python -m pytest -q   (uses the synthetic sample docs; no API key needed)"""
import os
import tempfile
from pathlib import Path

os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.sqlite3")
os.environ["EMBEDDING_MODEL"] = ""  # keyword-only for fast, offline tests
os.environ["LLM_API_KEYS"] = ""
os.environ["LLM_API_KEY"] = ""  # never call the real LLM from tests (dotenv does not override set vars)

from app import config, llm, store  # noqa: E402
from app.arabic import dialect_to_msa, fix_visual_order, normalize, tokenize  # noqa: E402
from app.index import HybridIndex  # noqa: E402
from app.ingest import Chunk, chunk_document  # noqa: E402
from app.index import Hit  # noqa: E402
from app.rag import RAGPipeline  # noqa: E402

store.init()
INDEX = HybridIndex.build([config.SAMPLE_DIR], out_dir=Path(tempfile.mkdtemp()))
PIPE = RAGPipeline(INDEX)


def test_normalize_and_stem():
    assert normalize("الإنذارُ الأكاديميّ") == "الانذار الاكاديمي"
    assert tokenize("الساعات")[0] == tokenize("ساعة")[0]


def test_dialect():
    assert "ما" in dialect_to_msa("شو بصير إذا غبت")
    assert "تغيبت" in dialect_to_msa("شو بصير إذا غبت")


def test_visual_order_fix():
    assert fix_visual_order("ةعماجلا يف بلاطلا ليجستلا") == "التسجيل الطالب في الجامعة"


def test_articles_are_chunks_with_metadata():
    chunks = chunk_document(config.SAMPLE_DIR / "لائحة_الدراسة_والامتحانات_تجريبي_2025.md")
    arts = {c.article for c in chunks}
    assert {"5", "6", "7", "8"} <= arts
    c8 = next(c for c in chunks if c.article == "8")
    assert c8.page == 3 and c8.version_year == 2025 and c8.doc_type == "regulation"


def test_table_rows_are_self_describing():
    chunks = chunk_document(config.SAMPLE_DIR / "الخطة_الدراسية_علم_الحاسوب_تجريبي_2024.md")
    t = next(c for c in chunks if c.chunk_type == "table")
    assert "اسم المادة: البرمجة المتقدمة | الساعات المعتمدة: 3" in t.text


def test_retrieval_hits_table_for_course_question():
    hits = INDEX.search(["كم ساعة معتمدة لمادة البرمجة المتقدمة"], k=3)
    assert any(h.chunk.chunk_type == "table" for h in hits)


def test_out_of_scope_and_greeting():
    assert PIPE.answer("اكتب لي قصيدة عن الجامعة", "t1").intent == "out_of_scope"
    assert PIPE.answer("مرحبا", "t2").intent == "greeting"


def test_llm_path_with_fake_model(monkeypatch):
    """Simulate the LLM to check rewrite JSON, citation parsing and refusal handling."""
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {"intent": "academic", "standalone": "ما الحد الأقصى للعبء الدراسي للطالب الذي معدله أقل من 2.0؟", "keywords": ["العبء الدراسي", "المعدل التراكمي"]})
    monkeypatch.setattr(llm, "chat", lambda *a, **k: ("لا يجوز لك تسجيل أكثر من 12 ساعة معتمدة [1]. وتُحسب [9] غير موجودة.", {}))
    a = PIPE.answer("معدلي 1.8 كم ساعة بقدر أسجل", "t3")
    assert not a.refused and a.citations and a.citations[0]["n"] == 1  # [9] ignored (out of range)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (f"{config.REFUSAL_TEXT} يرجى مراجعة دائرة القبول والتسجيل.", {}))
    b = PIPE.answer("متى امتحان الخوارزميات؟", "t4")
    assert b.refused and not b.citations


def test_follow_up_uses_history():
    sid = "t5"
    PIPE.answer("معدلي 1.8 كم ساعة أقدر أسجل؟", sid)
    a = PIPE.answer("وإذا كان معدلي أعلى من 3؟", sid)
    assert "أسجل" in a.rewritten or "ساعة" in a.rewritten


def test_pii_redaction():
    store.add_message("t6", "user", "رقمي الجامعي 2021123456 وايميلي a.b@uni.edu")
    h = store.history("t6")[-1]["content"]
    assert "2021123456" not in h and "@" not in h


def test_api():
    from fastapi.testclient import TestClient
    from app.server import app

    c = TestClient(app)
    r = c.post("/api/chat", json={"question": "كم ساعة معتمدة لمادة البرمجة المتقدمة؟"})
    assert r.status_code == 200 and r.json()["session_id"]
    assert c.get("/api/admin/stats").status_code in (401, 503)
    assert c.get("/").status_code == 200


def test_upload_keeps_existing_documents(tmp_path, monkeypatch):
    """Regression: an admin upload must be indexed on top of the corpus, not replace it."""
    from app import server

    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / "new.md").write_text("# تعليمات المكتبة\nتفتح المكتبة الساعة الثامنة صباحاً.", encoding="utf-8")
    folders = server._source_folders()
    assert config.SAMPLE_DIR in folders and config.UPLOAD_DIR in folders


def test_imports_and_raw_keep_existing_corpus_on_rebuild(tmp_path, monkeypatch):
    """CLI and server must use the same additive corpus, including later uploads."""
    from app import server

    for setting, name in (("SAMPLE_DIR", "existing"), ("RAW_DIR", "raw"),
                          ("INCOMING_DIR", "incoming"), ("UPLOAD_DIR", "uploads")):
        folder = tmp_path / name
        folder.mkdir()
        (folder / f"{name}.md").write_text(
            f"# {name}\nتعليمات التسجيل الأكاديمي الخاصة بالمستند {name} محفوظة ضمن مصادر الجامعة.",
            encoding="utf-8",
        )
        monkeypatch.setattr(config, setting, folder)
    folders = config.source_folders()
    assert server._source_folders() == folders
    idx = HybridIndex.build(folders, out_dir=tmp_path / "index")
    assert {c.source for c in idx.chunks} == {"existing.md", "raw.md", "incoming.md", "uploads.md"}


def test_pdf_retrieval_includes_opening_with_its_own_page(monkeypatch):
    opening = Chunk("plan-0", "الخطة الدراسية 2026 مجموع الساعات 138 ساعة معتمدة.",
                    "plan.pdf", "الخطة الدراسية", "study_plan", page=1)
    detail = Chunk("plan-1", "متطلبات هندسة البرمجيات الإجبارية والاختيارية.",
                   "plan.pdf", "الخطة الدراسية", "study_plan", page=5)
    idx = HybridIndex([opening, detail], None)
    monkeypatch.setattr(idx, "search", lambda *a, **k: [Hit(detail, 0.1), Hit(detail, 0.09)])
    hits = RAGPipeline(idx).retrieve("كم ساعة في الخطة؟", "كم ساعة في الخطة؟", [])
    assert [h.chunk.id for h in hits].count(opening.id) == 1
    assert hits[-1].chunk.page == 1
    assert "138" in RAGPipeline.build_context(hits)
    monkeypatch.setattr(idx, "search", lambda *a, **k: [Hit(opening, 0.1)])
    assert len(RAGPipeline(idx).retrieve("الخطة", "الخطة", [])) == 1


# ---------------------------------------------------------------- resilience (fake model, no network)
class _FakeErr(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.status_code = code


class _FakeClient:
    """Mimics openai.OpenAI().chat.completions.create; behaviour[(model, key)] = status code or answer text."""

    def __init__(self, key, behaviour, calls):
        self.key, self.behaviour, self.calls = key, behaviour, calls
        self.chat = self
        self.completions = self

    def create(self, model, **kw):
        self.calls.append((model, self.key))
        out = self.behaviour.get((model, self.key), 503)
        if isinstance(out, int):
            raise _FakeErr(out)
        from types import SimpleNamespace as NS

        return NS(choices=[NS(message=NS(content=out))], usage=NS(model_dump=lambda: {"prompt_tokens": 100, "completion_tokens": 20}))


def _fake_llm(monkeypatch, behaviour, keys=("k1", "k2"), models=("main", "alt")):
    calls = []
    monkeypatch.setattr(config, "LLM_API_KEYS", list(keys))
    monkeypatch.setattr(config, "LLM_MODEL", models[0])
    monkeypatch.setattr(config, "LLM_MODEL_FALLBACKS", list(models[1:]))
    monkeypatch.setattr(config, "LLM_RETRY_WAIT", 0)
    monkeypatch.setattr(config, "LLM_DEADLINE", 5)
    monkeypatch.setattr(llm, "_clients", {})
    monkeypatch.setattr(llm, "_make_client", lambda key: _FakeClient(key, behaviour, calls))
    return calls


def test_fallback_next_key_then_next_model(monkeypatch):
    calls = _fake_llm(monkeypatch, {("main", "k1"): 429, ("main", "k2"): 503, ("alt", "k1"): 404, ("alt", "k2"): "جواب [1]"})
    text, usage = llm.chat([{"role": "user", "content": "س"}])
    assert text == "جواب [1]" and usage["model"] == "alt"
    assert calls == [("main", "k1"), ("main", "k2"), ("alt", "k1"), ("alt", "k2")]
    st = llm.last_status()
    assert st["ok"] and st["model"] == "alt" and st["fallback_used"] and "k2" not in str(st)


def test_all_fail_returns_none_and_pipeline_falls_back(monkeypatch):
    _fake_llm(monkeypatch, {})  # everything 503
    text, usage = llm.chat([{"role": "user", "content": "س"}])
    assert text is None and llm.last_status()["ok"] is False
    a = PIPE.answer("كم ساعة معتمدة لمادة البرمجة المتقدمة؟", "t-fb")
    assert a.mode == "offline" and a.citations  # cited offline quote, not an error


def test_non_retryable_error_stops_early(monkeypatch):
    calls = _fake_llm(monkeypatch, {("main", "k1"): 401})
    assert llm.chat([{"role": "user", "content": "س"}])[0] is None and len(calls) == 1


def test_cache_avoids_second_call(monkeypatch):
    calls = _fake_llm(monkeypatch, {("main", "k1"): "مخزن [1]"}, keys=("k1",))
    monkeypatch.setattr(config, "LLM_CACHE", True)
    key = llm.hash_key("test-cache", "q", ["c1"], "main")
    first = llm.chat([{"role": "user", "content": "س"}], cache_key=key)
    second = llm.chat([{"role": "user", "content": "س"}], cache_key=key)
    assert first[0] == second[0] == "مخزن [1]" and len(calls) == 1
    assert second[1]["cached"] and second[1]["model"] == "main"


def test_refused_question_appears_in_knowledge_gaps_immediately():
    q = "متى يفتح مكتب السكن الجامعي أبوابه يوم الجمعة؟"
    a = PIPE.answer(q, "t-gap")
    assert a.refused
    gaps = [g["question"] for g in store.stats()["knowledge_gaps"]]
    assert any("السكن" in g for g in gaps)


def test_cited_partial_answer_is_not_a_refusal(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: None)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (f"يمكنك تسجيل البرمجة المتقدمة COMP231 [1]. {config.REFUSAL_TEXT} بخصوص باقي المواد.", {}))
    a = PIPE.answer("كم ساعة معتمدة لمادة البرمجة المتقدمة؟", "t-partial")
    assert not a.refused and a.citations


def test_detect_lang():
    from app.arabic import detect_lang

    assert detect_lang("How many absences are allowed in a course?") == "en"
    assert detect_lang("ما الحد الأقصى للغياب؟") == "ar"
    assert detect_lang("كم ساعة لمادة COMP231؟") == "ar"


def test_english_refusal_text_offline():
    a = PIPE.answer("When is the algorithms exam next week?", "t-en")
    assert a.refused and a.answer == config.REFUSAL_TEXT_EN
    from app.rag import is_refusal

    assert is_refusal(config.REFUSAL_TEXT_EN) and is_refusal(config.REFUSAL_TEXT)


def test_offline_follow_up_merges_previous_question():
    sid = "t-follow"
    PIPE.answer("كم ساعة يسجل الطالب الذي معدله أقل من 2؟", sid)
    a = PIPE.answer("وإذا كان معدلي أعلى من 3؟", sid)
    assert "معدله" in a.rewritten or "يسجل" in a.rewritten
    b = PIPE.answer("what about summer?", sid)
    assert "يسجل" in b.rewritten or "معدل" in b.rewritten


def test_login_and_conversations_are_private():
    from fastapi.testclient import TestClient
    from app.server import app

    c = TestClient(app)
    assert c.post("/api/login", json={"student_id": "123", "name": "س"}).status_code == 422
    ta = c.post("/api/login", json={"student_id": "22210001", "name": "طالب أ"}).json()["token"]
    tb = c.post("/api/login", json={"student_id": "22210002", "name": "طالب ب"}).json()["token"]
    A, B = {"Authorization": f"Bearer {ta}"}, {"Authorization": f"Bearer {tb}"}
    sid = c.post("/api/chat", headers=A, json={"question": "كم ساعة معتمدة لمادة البرمجة المتقدمة؟"}).json()["session_id"]
    convs = c.get("/api/conversations", headers=A).json()
    assert convs[0]["id"] == sid and convs[0]["title"].startswith("كم ساعة")
    assert len(c.get(f"/api/conversations/{sid}", headers=A).json()) == 2
    assert c.get(f"/api/conversations/{sid}", headers=B).status_code == 404  # not B's
    assert c.delete(f"/api/conversations/{sid}", headers=B).status_code == 404
    assert c.get("/api/conversations").status_code == 401  # no token
    bad = "Bearer " + ta[:-2] + ("00" if not ta.endswith("00") else "11")
    assert c.get("/api/conversations", headers={"Authorization": bad}).status_code == 401  # tampered
    sid_b = c.post("/api/chat", headers=B, json={"question": "مرحبا", "session_id": sid}).json()["session_id"]
    assert sid_b != sid  # B cannot append to A's conversation
    assert c.delete(f"/api/conversations/{sid}", headers=A).json()["ok"]
    assert store.stats()["registered_students"] >= 2
