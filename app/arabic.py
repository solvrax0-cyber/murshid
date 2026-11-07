"""Arabic text utilities: cleaning PDF artefacts, search normalisation, light stemming,
and a Palestinian/Levantine dialect -> MSA mapper used for query understanding."""
import re
import unicodedata

TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ARABIC_CHAR = re.compile(r"[؀-ۿ]")
PUNCT = re.compile(r"[^\w\s؀-ۿ]|[_،؛؟٪٫٬٭۔]")  # non-word chars + Arabic punctuation
WS = re.compile(r"[ \t ‏‎‪-‮]+")


def clean_text(text: str) -> str:
    """Display-safe cleanup: NFKC fixes Arabic presentation forms that PDFs often contain,
    removes tatweel / bidi control marks, unifies digits, collapses spaces."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(TATWEEL, "").translate(DIGITS)
    text = WS.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


ARABIC_ANY = re.compile(r"[؀-ۿﭐ-﷿ﹰ-﻿]")  # incl. presentation forms


def reversed_score(text: str) -> int:
    """>0 means the text looks like *visual* order (reversed): many words end in 'لا' (= reversed 'ال')."""
    words = re.findall(r"[؀-ۿ]+", unicodedata.normalize("NFKC", text))
    fwd = sum(1 for w in words if w.startswith("ال") and len(w) > 3)
    rev = sum(1 for w in words if w.endswith("لا") and len(w) > 3)
    return rev - fwd


def _reverse_line(line: str) -> str:
    rev = line[::-1]
    # re-reverse latin/digit runs so "PDF", "COMP231", "2024", "15%" stay correct
    rev = re.sub(r"[A-Za-z0-9.,:/%\-_@]+", lambda m: m.group(0)[::-1], rev)
    return rev.translate(str.maketrans("()[]{}«»<>", ")(][}{»«><"))


def fix_visual_order(text: str, force: bool | None = None) -> str:
    """Some extractors (e.g. pdfplumber/pdfminer) return Arabic in *visual* order.
    Decide once for the whole page (more robust than per line), then reverse every Arabic line.
    Must run BEFORE NFKC so ligature glyphs (لا، ير، ين...) expand in the correct order."""
    if force is None:
        force = reversed_score(text) >= 2
    if not force:
        return text
    return "\n".join(_reverse_line(l) if ARABIC_ANY.search(l) else l for l in text.split("\n"))


def normalize(text: str) -> str:
    """Aggressive normalisation for matching only (never shown to users)."""
    text = clean_text(text).lower()
    text = TASHKEEL.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    text = PUNCT.sub(" ", text)
    return WS.sub(" ", text).strip()


STOPWORDS = set(normalize(w) for w in """
في من على الى إلى عن مع هل ما ماذا هذا هذه ذلك تلك التي الذي الذين او أو ثم قد لا لم لن ان أن إن كان كانت
يكون تكون هو هي هم انا أنا انت أنت نحن كل بعد قبل عند اذا إذا كيف كم اي أي بين حتى لكن غير يجب يمكن
لي له لها لك بها به فيها فيه منها منه عليه عليها ايضا أيضا و يا شو بدي اللي
""".split())

PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ف", "ب", "ل")
SUFFIXES = ("ات", "ون", "ين", "ها", "هم", "كم", "نا", "ه", "ي")


def light_stem(token: str) -> str:
    for p in PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 3:
            token = token[len(p):]
            break
    for s in SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= 3:
            token = token[: -len(s)]
            break
    return token


def tokenize(text: str) -> list[str]:
    toks = []
    for t in normalize(text).split():
        if t in STOPWORDS or len(t) < 2:
            continue
        toks.append(light_stem(t) if ARABIC_CHAR.search(t) else t)
    return toks


# ---- Dialect (Palestinian / Levantine) -> MSA. Order matters: longer phrases first. ----
DIALECT_MAP = [
    (r"شو\s+(?:ضايل|ضل|بقي|باقي)\s+(?:لي|إلي|الي)", "ما المتبقي علي"),
    (r"كيف\s+بدي", "كيف"),
    (r"(?:شو|ايش|إيش|اش)", "ما"),
    (r"(?:ليش|ليه)", "لماذا"),
    (r"(?:قديش|أديش|اديش|كديش)", "كم"),
    (r"(?:امتى|إمتى|ايمتى|إيمتى)", "متى"),
    (r"وين", "أين"),
    (r"(?:بقدر|بأقدر|أقدر|اقدر|بنقدر|بتقدر)", "أستطيع"),
    (r"(?:بدي|بدّي|بدنا)", "أريد"),
    (r"(?:بصير|بيصير|رح\s+يصير|راح\s+يصير)", "يحدث"),
    (r"(?:بتأثر|بيأثر|بأثر|بتاثر)", "يؤثر"),
    (r"(?:عشان|علشان|منشان)", "لكي"),
    (r"(?:اللي|إللي|الّي|اليل)", "الذي"),
    (r"(?:هسا|هسّا|هلأ|هلق|هلّق)", "الآن"),
    (r"(?:مش|مو|ماش)", "ليس"),
    (r"(?:كتير|كثير)", "كثير"),
    (r"(?:منيح|مليح)", "جيد"),
    (r"غبت", "تغيبت"),
    (r"(?:أسحب|اسحب|بسحب)", "أنسحب من"),
    (r"سحب", "انسحاب"),
    (r"(?:أسجل|اسجل|بسجل)", "أسجل"),
    (r"(?:تخرج|اتخرج|أتخرج)", "أتخرج"),
]
_DIALECT_RX = [(re.compile(rf"(?<![؀-ۿ]){p}(?![؀-ۿ])"), r) for p, r in DIALECT_MAP]

# Academic synonym expansion for keyword search (helps BM25 bridge dialect -> regulation wording)
SYNONYMS = {
    "انسحاب": "الانسحاب سحب مادة",
    "تغيبت": "الغياب الحضور نسبة الغياب حرمان",
    "غياب": "الغياب الحضور حرمان",
    "معدل": "المعدل التراكمي",
    "أتخرج": "متطلبات التخرج الساعات المعتمدة المتبقية",
    "رسبت": "الرسوب إعادة المادة متطلب سابق",
    "إنذار": "الإنذار الأكاديمي",
    "ساعة": "الساعات المعتمدة العبء الدراسي",
    "أسجل": "التسجيل العبء الدراسي الحد الأقصى",
    "شرط": "متطلب سابق",
}


def dialect_to_msa(question: str) -> str:
    q = clean_text(question)
    for rx, rep in _DIALECT_RX:
        q = rx.sub(rep, q)
    return q


def expand_query(question: str) -> str:
    q = dialect_to_msa(question)
    extra = [v for k, v in SYNONYMS.items() if k in q]
    return (q + " " + " ".join(extra)).strip()


PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[بريد]"),
    (re.compile(r"(?:\+?970|\+?972|0)5\d[\s-]?\d{3}[\s-]?\d{4}"), "[هاتف]"),
    (re.compile(r"\b\d{8,10}\b"), "[رقم]"),  # student / ID numbers
]


def redact_pii(text: str) -> str:
    for rx, rep in PII_PATTERNS:
        text = rx.sub(rep, text)
    return text


_LATIN = re.compile(r"[A-Za-z]")
_ARABIC = re.compile(r"[؀-ۿ]")


def detect_lang(text: str) -> str:
    """"en" if Latin letters are more than 60% of the letters, else "ar" (Arabic is the default)."""
    lat, ara = len(_LATIN.findall(text or "")), len(_ARABIC.findall(text or ""))
    return "en" if lat + ara and lat / (lat + ara) > 0.6 else "ar"
