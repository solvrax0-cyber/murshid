"""FastAPI app: chat API + admin (upload / re-index / analytics) + static RTL web UI."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, llm, store
from .index import HybridIndex
from .rag import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("murshid")
WEB = config.ROOT / "web"

app = FastAPI(title="Murshid – Arabic Academic Advisor (RAG)", version="1.0")
store.init()
if (WEB / "static").exists():
    app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
_state = {"index": None, "pipeline": None, "indexing": False}
_lock = threading.Lock()


def _source_folders() -> list[Path]:
    return config.source_folders()


def _load_or_build() -> None:
    idx = HybridIndex.load()
    if not idx.chunks:
        log.info("no index found -> building from %s", _source_folders())
        idx = HybridIndex.build(_source_folders())
    _state["index"], _state["pipeline"] = idx, RAGPipeline(idx)


def _rebuild() -> None:
    with _lock:
        _state["indexing"] = True
        try:
            idx = HybridIndex.build(_source_folders())
            _state["index"], _state["pipeline"] = idx, RAGPipeline(idx)
        finally:
            _state["indexing"] = False


_load_or_build()

# ------------------------------------------------------------------ security helpers
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "?"
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= config.RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "عدد كبير من الطلبات، حاول بعد دقيقة.")
    q.append(now)


def require_admin(token: str | None) -> None:
    if not config.ADMIN_TOKEN:
        raise HTTPException(503, "ADMIN_TOKEN is not configured on the server")
    if not token or not hmac.compare_digest(token, config.ADMIN_TOKEN):
        raise HTTPException(401, "رمز الإدارة غير صحيح")


# ------------------------------------------------------------------ API
class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=64)


class LoginIn(BaseModel):
    student_id: str = Field(pattern=r"^\d{8}$")
    name: str = Field(min_length=2, max_length=60)


# ------------------------------------------------------------------ demo student login (HMAC-signed token)
def _sign(payload: str) -> str:
    return hmac.new(config.SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()


def make_token(student_id: str, name: str) -> str:
    raw = json.dumps({"sid": student_id, "name": name, "exp": int(time.time()) + config.LOGIN_TTL_HOURS * 3600},
                     ensure_ascii=False, separators=(",", ":"))
    payload = base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")
    return f"{payload}.{_sign(payload)}"


def read_token(token: str | None) -> dict | None:
    try:
        payload, sig = (token or "").split(".", 1)
        if not hmac.compare_digest(sig, _sign(payload)):
            return None
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return data if data.get("exp", 0) > time.time() else None
    except Exception:
        return None


def require_user(authorization: str | None) -> dict:
    user = read_token((authorization or "").removeprefix("Bearer ").strip())
    if not user:
        raise HTTPException(401, "يرجى تسجيل الدخول")
    return user


class FeedbackIn(BaseModel):
    log_id: int
    value: int


@app.post("/api/login")
def login(body: LoginIn, request: Request):
    rate_limit(request)
    store.upsert_user(body.student_id, body.name.strip())
    return {"token": make_token(body.student_id, body.name.strip()), "name": body.name.strip(), "student_id": body.student_id}


@app.get("/api/conversations")
def conversations(authorization: str | None = Header(default=None)):
    return store.list_conversations(require_user(authorization)["sid"])


@app.get("/api/conversations/{conv_id}")
def conversation(conv_id: str, authorization: str | None = Header(default=None)):
    user = require_user(authorization)
    if store.conversation_owner(conv_id) != user["sid"]:
        raise HTTPException(404, "المحادثة غير موجودة")
    return store.history(conv_id, turns=200)


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str, authorization: str | None = Header(default=None)):
    user = require_user(authorization)
    if not store.delete_conversation(conv_id, user["sid"]):
        raise HTTPException(404, "المحادثة غير موجودة")
    return {"ok": True}


@app.post("/api/chat")
def chat(body: ChatIn, request: Request, authorization: str | None = Header(default=None)):
    rate_limit(request)
    sid = body.session_id if body.session_id and re.fullmatch(r"[\w-]{6,64}", body.session_id) else uuid.uuid4().hex
    user = read_token((authorization or "").removeprefix("Bearer ").strip())
    if user:
        owner = store.conversation_owner(sid)
        if owner and owner != user["sid"]:
            sid = uuid.uuid4().hex  # never append to someone else's conversation
        store.touch_conversation(sid, user["sid"], body.question)
    ans = _state["pipeline"].answer(body.question, sid)
    out = asdict(ans)
    out["session_id"] = sid
    return out


@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    return store.history(session_id, turns=50)


@app.post("/api/feedback")
def feedback(body: FeedbackIn):
    store.set_feedback(body.log_id, body.value)
    return {"ok": True}


@app.get("/api/health")
def health():
    idx = _state["index"]
    return {
        "chunks": len(idx.chunks), "sources": sorted({c.source for c in idx.chunks}),
        "dense": idx.has_dense, "embedding_model": config.EMBEDDING_MODEL or None,
        "reranker": config.RERANK_MODEL or None, "llm": config.LLM_MODEL if llm.available() else "offline",
        # resilience: configured chain (never the keys themselves) + what the last call actually did
        "llm_fallbacks": config.LLM_MODEL_FALLBACKS if llm.available() else [], "llm_keys": len(config.LLM_API_KEYS),
        "llm_last_call": llm.last_status(),
        "indexing": _state["indexing"],
    }


@app.get("/api/admin/stats")
def admin_stats(x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    return store.stats()


@app.post("/api/admin/upload")
async def admin_upload(bg: BackgroundTasks, file: UploadFile = File(...), x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    name = Path(file.filename or "").name
    if not re.search(r"\.(pdf|md|txt)$", name, re.I):
        raise HTTPException(400, "يُسمح فقط بملفات PDF أو MD أو TXT")
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, "الملف كبير جداً")
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w؀-ۿ .\-]", "_", name)
    (config.UPLOAD_DIR / safe).write_bytes(data)
    bg.add_task(_rebuild)
    return {"ok": True, "saved": safe, "message": "تم الرفع، جارٍ إعادة الفهرسة"}


@app.post("/api/admin/reindex")
def admin_reindex(bg: BackgroundTasks, x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    bg.add_task(_rebuild)
    return {"ok": True}


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")
