"""Ingestion: PDF / Markdown / text -> clean, structure-aware chunks with metadata.

Chunking strategy (why):
  * Regulations are numbered articles ("المادة (14)") that are self-contained rules and
    are what students must cite -> we split on article boundaries first and keep the article
    number as metadata.
  * Long free text (student guide) has no articles -> recursive split ~1100 chars with
    150 overlap, on paragraph/sentence boundaries so a rule is not cut mid-sentence.
  * Tables (study plans) are the #1 source of wrong answers -> every table row is rewritten
    as "header: value | header: value" and rows are grouped with the header repeated,
    so each chunk is self-describing ("مادة البرمجة المتقدمة | الساعات: 3 | المتطلب: ...").
  * Every chunk keeps source file, title, page range, article, doc type and version year
    (used for citations and to prefer the newest version when documents conflict).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config
from .arabic import clean_text, fix_visual_order, reversed_score

log = logging.getLogger(__name__)

ARTICLE_RX = re.compile(
    r"(?:^|\n)[ \t]*(?:ال)?(?:مادة|المادة|بند)[ \t]*[\(\[\-]?[ \t]*(\d{1,3})[ \t]*[\)\]]?[ \t]*[:\-–]?",
)
HEADING_RX = re.compile(r"^(?:#{1,4}\s+.+|(?:الفصل|الباب|القسم|الجزء)\s+.+)$", re.M)
PAGE_MARK_RX = re.compile(r"^\s*-{3}\s*صفحة\s*(\d+)\s*-{3}\s*$", re.M)
YEAR_RX = re.compile(r"(20\d{2})")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    title: str
    doc_type: str
    page: int | None = None
    page_end: int | None = None
    article: str | None = None
    section: str | None = None
    chunk_type: str = "text"  # text | table
    version_year: int | None = None
    extra: dict = field(default_factory=dict)

    def label(self) -> str:
        parts = [self.title]
        if self.page:
            parts.append(f"صفحة {self.page}" + (f"-{self.page_end}" if self.page_end and self.page_end != self.page else ""))
        if self.article:
            parts.append(f"المادة {self.article}")
        elif self.section:
            parts.append(self.section[:60])
        return " · ".join(parts)


def guess_doc_type(name: str) -> str:
    if re.search(r"لائح|نظام|تعليمات|قانون|regulation|bylaw", name, re.I):
        return "regulation"
    if re.search(r"خط[ةه]|plan|curriculum|مساق", name, re.I):
        return "study_plan"
    return "guide"


# ---------------------------------------------------------------- loaders
def _table_to_rows(table: list[list], reverse: bool | None = False) -> list[str]:
    rows = [[clean_text(fix_visual_order(str(c or ""), reverse)).replace("\n", " ") for c in r] for r in table if r]
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return []
    header = rows[0]
    out = []
    for r in rows[1:]:
        cells = []
        for h, c in zip(header, r):
            if c:
                cells.append(f"{h}: {c}" if h else c)
        if cells:
            out.append(" | ".join(cells))
    return out


def load_pdf(path: Path) -> tuple[list[tuple[int, str]], list[tuple[int, list[str]]]]:
    """Return ([(page_no, text)], [(page_no, table_rows)]).

    Arabic PDFs are tricky: we tested on the hackathon's own PDFs and found
      * PyMuPDF: logical letters but words in reversed order, and some ligature letters dropped;
      * pdfplumber: whole lines in visual (reversed) order BUT ligature glyphs intact.
    So we use pdfplumber, detect visual order per page, reverse lines *before* NFKC
    (ligatures then expand correctly), and fall back to PyMuPDF if a page yields no text."""
    import logging as _l

    import pdfplumber

    _l.getLogger("pdfminer").setLevel(_l.ERROR)
    pages, tables = [], []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            reverse = reversed_score(raw) >= 2
            txt = clean_text(fix_visual_order(raw, reverse))
            if len(txt) < 30:
                txt = _pymupdf_page(path, i - 1) or txt
            if len(txt) < 30:
                log.warning("%s page %s has almost no text (scanned? needs OCR)", path.name, i)
            pages.append((i, txt))
            try:
                for t in page.extract_tables() or []:
                    rows = _table_to_rows(t, reverse)
                    if rows:
                        tables.append((i, rows))
            except Exception as e:  # tables are a bonus; never break ingestion
                log.warning("table extraction failed for %s p%s: %s", path.name, i, e)
    return pages, tables


def _pymupdf_page(path: Path, idx: int) -> str:
    try:
        import pymupdf

        with pymupdf.open(path) as doc:
            return clean_text(doc[idx].get_text("text"))
    except Exception:
        return ""


MD_TABLE_RX = re.compile(r"((?:^\|.*\|[ \t]*\n?){2,})", re.M)


def load_text(path: Path) -> tuple[list[tuple[int, str]], list[tuple[int, list[str]]]]:
    raw = clean_text(path.read_text(encoding="utf-8"))
    # optional page markers: "--- صفحة 3 ---"
    parts = PAGE_MARK_RX.split(raw)
    pages: list[tuple[int, str]] = []
    if len(parts) > 1:
        if parts[0].strip():
            pages.append((1, parts[0]))
        for n, body in zip(parts[1::2], parts[2::2]):
            pages.append((int(n), body))
    else:
        pages = [(1, raw)]
    tables, clean_pages = [], []
    for pno, body in pages:
        for m in MD_TABLE_RX.finditer(body):
            lines = [l.strip() for l in m.group(1).strip().split("\n")]
            grid = [[c.strip() for c in l.strip("|").split("|")] for l in lines if not re.match(r"^\|?\s*:?-{2,}", l)]
            rows = _table_to_rows(grid)
            if rows:
                tables.append((pno, rows))
        clean_pages.append((pno, MD_TABLE_RX.sub("\n", body)))
    return clean_pages, tables


# ---------------------------------------------------------------- chunking
def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    # paragraph -> sentence boundaries
    units = re.split(r"(?<=[\.\!\؟\?\n؛:])\s+", text)
    chunks, cur = [], ""
    for u in units:
        if len(cur) + len(u) + 1 <= max_chars:
            cur = f"{cur} {u}".strip()
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if overlap and cur else ""
            if " " in tail:  # start the overlap on a word boundary
                tail = tail[tail.index(" ") + 1:]
            cur = f"{tail} {u}".strip() if len(u) < max_chars else u
            while len(cur) > max_chars:  # very long sentence
                chunks.append(cur[:max_chars])
                cur = cur[max_chars - overlap:]
    if cur:
        chunks.append(cur)
    return chunks


def _strip_headers_footers(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop short lines repeated on most pages (running headers, footers, page numbers)."""
    if len(pages) < 3:
        return pages
    from collections import Counter

    key = lambda l: re.sub(r"\d+", "#", l.strip())
    counts = Counter(k for _, t in pages for k in {key(l) for l in t.split("\n") if 0 < len(l.strip()) < 80})
    limit = max(3, len(pages) // 2)
    noisy = {k for k, n in counts.items() if n >= limit}
    out = []
    for pno, t in pages:
        lines = [l for l in t.split("\n") if key(l) not in noisy and not re.fullmatch(r"\s*\d{1,3}\s*", l)]
        out.append((pno, "\n".join(lines)))
    return out


def _page_at(offsets: list[tuple[int, int]], pos: int) -> int | None:
    page = offsets[0][1] if offsets else None
    for start, pno in offsets:
        if start <= pos:
            page = pno
        else:
            break
    return page


def chunk_document(path: Path) -> list[Chunk]:
    name = path.name
    title = re.sub(r"[_\-]+", " ", path.stem).strip()
    doc_type = guess_doc_type(name)
    if path.suffix.lower() == ".pdf":
        pages, tables = load_pdf(path)
    else:
        pages, tables = load_text(path)

    pages = _strip_headers_footers(pages)
    year_m = YEAR_RX.search(name) or YEAR_RX.search(pages[0][1][:500] if pages else "")
    year = int(year_m.group(1)) if year_m else None
    doc_id = hashlib.md5(name.encode()).hexdigest()[:8]

    full, offsets = "", []
    for pno, txt in pages:
        offsets.append((len(full), pno))
        full += txt.strip() + "\n\n"

    # section headings -> nearest heading for each position
    headings = [(m.start(), m.group(0).lstrip("# ").strip()) for m in HEADING_RX.finditer(full)]

    def heading_at(pos: int) -> str | None:
        h = None
        for start, txt in headings:
            if start <= pos:
                h = txt
            else:
                break
        return h

    chunks: list[Chunk] = []
    arts = list(ARTICLE_RX.finditer(full))
    segments: list[tuple[int, int, str | None]] = []
    if len(arts) >= 2:  # regulations: one segment per numbered article
        if arts[0].start() > 0:
            segments.append((0, arts[0].start(), None))
        for a, b in zip(arts, arts[1:] + [None]):
            segments.append((a.start(), b.start() if b else len(full), a.group(1)))
    elif len(headings) >= 2:  # guides: one segment per heading / section
        starts = [0] + [s for s, _ in headings if s > 0]
        for s, e in zip(starts, starts[1:] + [len(full)]):
            segments.append((s, e, None))
    else:
        segments.append((0, len(full), None))

    for start, end, art in segments:
        seg = full[start:end]
        cursor = start
        for piece in _split_long(seg, config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP):
            if len(piece) < 40 or ("\n" not in piece.strip() and piece.lstrip().startswith("#")):
                continue  # skip empty fragments and heading-only lines
            idx = full.find(piece[:40], cursor)
            pos = idx if idx >= 0 else cursor
            cursor = max(cursor, pos)
            p_start, p_end = _page_at(offsets, pos), _page_at(offsets, pos + len(piece) - 1)
            chunks.append(Chunk(
                id=f"{doc_id}-{len(chunks)}", text=piece.strip(), source=name, title=title,
                doc_type=doc_type, page=p_start, page_end=p_end, article=art,
                section=heading_at(pos), version_year=year,
            ))

    for pno, rows in tables:
        for i in range(0, len(rows), config.TABLE_ROWS_PER_CHUNK):
            block = rows[i:i + config.TABLE_ROWS_PER_CHUNK]
            chunks.append(Chunk(
                id=f"{doc_id}-t{len(chunks)}", text="جدول:\n" + "\n".join(block), source=name,
                title=title, doc_type=doc_type, page=pno, page_end=pno, chunk_type="table",
                section="جدول", version_year=year,
            ))
    log.info("%s -> %d chunks (%d table chunks)", name, len(chunks), sum(c.chunk_type == "table" for c in chunks))
    return chunks


def ingest_folder(folder: Path) -> list[Chunk]:
    files = sorted(p for p in folder.glob("**/*") if p.suffix.lower() in {".pdf", ".md", ".txt"})
    out: list[Chunk] = []
    for f in files:
        try:
            out.extend(chunk_document(f))
        except Exception as e:
            log.error("failed to ingest %s: %s", f, e)
    return out


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    return [Chunk(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
