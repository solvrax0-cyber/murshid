"""Central configuration. Every secret comes from environment variables (.env), never from code."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


# --- paths ---
RAW_DIR = Path(_env("DATA_DIR", str(ROOT / "data" / "raw")))
SAMPLE_DIR = ROOT / "data" / "sample"
INCOMING_DIR = Path(_env("INCOMING_DIR", str(ROOT / "data" / "incoming")))
UPLOAD_DIR = Path(_env("UPLOAD_DIR", str(ROOT / "data" / "uploads")))  # admin uploads, added on top of raw/sample
INDEX_DIR = Path(_env("INDEX_DIR", str(ROOT / "data" / "index")))
DB_PATH = Path(_env("DB_PATH", str(ROOT / "data" / "murshid.sqlite3")))


def source_folders() -> list[Path]:
    """Keep the existing corpus and add imported documents and admin uploads."""
    folders = [SAMPLE_DIR, RAW_DIR, INCOMING_DIR, UPLOAD_DIR]
    return list(dict.fromkeys(folder.resolve() for folder in folders if folder.is_dir()))

# --- LLM (any OpenAI-compatible endpoint: Gemini, OpenAI, Groq, Ollama...) ---
LLM_API_KEY = _env("LLM_API_KEY")
# several keys (comma-separated) -> tried in turn when one hits its quota; LLM_API_KEY alone still works
LLM_API_KEYS = [k.strip() for k in (_env("LLM_API_KEYS") or LLM_API_KEY).split(",") if k.strip()]
LLM_BASE_URL = _env("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL = _env("LLM_MODEL", "gemini-3.5-flash-lite")
# tried in order after LLM_MODEL fails with 429/503/404/timeout (each on every key)
LLM_MODEL_FALLBACKS = [m.strip() for m in _env("LLM_MODEL_FALLBACKS", "gemini-3.1-flash-lite").split(",") if m.strip()]
LLM_REASONING_EFFORT = _env("LLM_REASONING_EFFORT")  # e.g. "none" / "low"; applied to LLM_MODEL only
LLM_TIMEOUT = _float("LLM_TIMEOUT", 10)       # per attempt
LLM_DEADLINE = _float("LLM_DEADLINE", 15)     # whole fallback chain; then the offline quote is returned
LLM_RETRY_WAIT = _float("LLM_RETRY_WAIT", 0.5)  # pause before attempt n = n * this (capped at 8s)
LLM_CACHE = _env("LLM_CACHE", "1") not in ("0", "false", "no")
JUDGE_MODEL = _env("JUDGE_MODEL")  # eval judge; a different model avoids self-grading (empty = LLM_MODEL)

# --- retrieval ---
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")  # "" disables dense search
RERANK_MODEL = _env("RERANK_MODEL")  # e.g. BAAI/bge-reranker-v2-m3 (optional, slower)
TOP_K = _int("TOP_K", 5)
CANDIDATES = _int("CANDIDATES", 30)
CHUNK_MAX_CHARS = _int("CHUNK_MAX_CHARS", 1100)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
TABLE_ROWS_PER_CHUNK = _int("TABLE_ROWS_PER_CHUNK", 12)
# Refusal gate: below these, we refuse without calling the LLM (calibrate with eval/run_eval.py)
MIN_DENSE_SCORE = _float("MIN_DENSE_SCORE", 0.82)
MIN_TERM_COVERAGE = _float("MIN_TERM_COVERAGE", 0.45)

# --- app ---
ADMIN_TOKEN = _env("ADMIN_TOKEN")
# signs the demo student-login tokens; random per start if unset (logins then expire on restart)
import secrets as _secrets  # noqa: E402
SECRET_KEY = _env("SECRET_KEY") or _secrets.token_hex(32)
LOGIN_TTL_HOURS = _int("LOGIN_TTL_HOURS", 12)
HISTORY_TURNS = _int("HISTORY_TURNS", 4)
RATE_LIMIT_PER_MIN = _int("RATE_LIMIT_PER_MIN", 30)
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 25)
UNIVERSITY_NAME = _env("UNIVERSITY_NAME", "جامعة الخليل")
REFERRAL_OFFICE = _env("REFERRAL_OFFICE", "دائرة القبول والتسجيل أو مرشدك الأكاديمي")

REFUSAL_TEXT = "لا أجد هذه المعلومة في المستندات المتاحة لي."
REFUSAL_TEXT_EN = "I can't find this information in the documents available to me. Please contact the Admission and Registration Office."
