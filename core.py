"""Evidence pipeline core: ingest, verify, and the two LLM calls.

Python owns the text at both ends. It indexes the files before the LLM sees
them and checks every excerpt the LLM returns against the raw files after.
The LLM only clusters and cites; it never decides what counts as verified.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# M0 Ingest and index
# ---------------------------------------------------------------------------

BULLET_RE = re.compile(r"^\s*[-*•]\s+(.*\S)\s*$")
HEADER_SPEAKER_RE = re.compile(r"\|\s*(.+?)\s*$")
DIRECT_SPEECH_CHARS = ('"', "“", "”")


@dataclass
class Excerpt:
    id: str
    text: str
    speaker: str
    file: str
    line: int
    section: str
    type: str  # "direct_speech" | "interviewer_note"


@dataclass
class Corpus:
    raw: dict[str, str] = field(default_factory=dict)  # file name -> full text
    catalog: dict[str, Excerpt] = field(default_factory=dict)
    speakers: dict[str, str] = field(default_factory=dict)  # file name -> speaker
    warnings: list[str] = field(default_factory=list)


def _is_section_header(line: str) -> bool:
    s = line.strip()
    return bool(s) and any(c.isalpha() for c in s) and s == s.upper() and not BULLET_RE.match(s)


def speaker_from_filename(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r"^interview[_\s-]*\d+[_\s-]*", "", stem, flags=re.I)
    words = [w for w in re.split(r"[_\s-]+", stem) if w]
    return " ".join(words) or "Unknown"


def ingest(files: dict[str, str]) -> Corpus:
    """Index every excerpt in every file. `files` maps file name -> text."""
    corpus = Corpus()
    for n, name in enumerate(sorted(files), start=1):
        text = files[name].replace("\r\n", "\n")
        corpus.raw[name] = text
        lines = text.split("\n")

        first = next((l for l in lines if l.strip()), "")
        m = None if BULLET_RE.match(first) else HEADER_SPEAKER_RE.search(first)
        has_header = bool(m)
        speaker = m.group(1) if m else speaker_from_filename(name)
        if not has_header:
            corpus.warnings.append(f"{name}: no '| Speaker' header line; speaker taken from file name ({speaker}).")
        if speaker == "Unknown":
            corpus.warnings.append(f"{name}: speaker could not be determined.")
        corpus.speakers[name] = speaker

        has_bullets = any(BULLET_RE.match(l) for l in lines)
        if not has_bullets:
            corpus.warnings.append(f"{name}: no bullet lines found; each non-empty line is treated as an excerpt.")

        section = ""
        header_skipped = False
        for i, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            if has_header and not header_skipped:
                header_skipped = True
                continue
            if _is_section_header(line):
                section = line.strip()
                continue
            bm = BULLET_RE.match(line)
            if bm:
                body = bm.group(1)
            elif not has_bullets:
                body = line.strip()
            else:
                continue
            eid = f"I{n}-L{i}"
            corpus.catalog[eid] = Excerpt(
                id=eid,
                text=body,
                speaker=speaker,
                file=name,
                line=i,
                section=section,
                type="direct_speech" if any(c in body for c in DIRECT_SPEECH_CHARS) else "interviewer_note",
            )
    return corpus


def catalog_for_prompt(corpus: Corpus) -> str:
    return "\n".join(
        f"[{e.id}] ({e.speaker} | {e.section or 'no section'}) {e.text}" for e in corpus.catalog.values()
    )


# ---------------------------------------------------------------------------
# M2 Verifier
# ---------------------------------------------------------------------------

VERIFIED = "VERIFIED"
WRONG_LINE = "WRONG_LINE"
NORMALISED_ONLY = "NORMALISED_ONLY"
NOT_FOUND = "NOT_FOUND"
BAD_ID = "BAD_ID"
TOO_SHORT = "TOO_SHORT"
FAILURE_STATUSES = (WRONG_LINE, NORMALISED_ONLY, NOT_FOUND, BAD_ID, TOO_SHORT)
MIN_WORDS = 3

# Only quote marks and whitespace are normalised, and only to diagnose a
# near-miss. Dashes and ellipses are never touched, and a normalised match
# never counts as verified.
_QUOTE_MAP = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.translate(_QUOTE_MAP)).strip()


@dataclass
class Check:
    quote_id: str
    quote_text: str
    status: str = ""
    speaker: str = ""
    file: str = ""
    line: int | None = None
    type: str = ""
    found_in: list[str] = field(default_factory=list)  # ids holding the text when the cited line doesn't

    @property
    def ok(self) -> bool:
        return self.status == VERIFIED


def verify(quote_id: str, quote_text: str, corpus: Corpus) -> Check:
    qt = (quote_text or "").strip()
    excerpt = corpus.catalog.get(quote_id)
    check = Check(quote_id=quote_id, quote_text=quote_text)
    if excerpt:
        check.speaker, check.file, check.line, check.type = excerpt.speaker, excerpt.file, excerpt.line, excerpt.type

    if len(qt.split()) < MIN_WORDS:
        check.status = TOO_SHORT
        return check
    if excerpt is None:
        check.status = BAD_ID
        return check

    raw = corpus.raw[excerpt.file]
    if qt in raw and qt in excerpt.text:
        check.status = VERIFIED
        return check

    if any(qt in r for r in corpus.raw.values()):
        check.status = WRONG_LINE
        check.found_in = [e.id for e in corpus.catalog.values() if qt in e.text]
        return check

    nq = _normalise(qt)
    if any(nq in _normalise(r) for r in corpus.raw.values()):
        check.status = NORMALISED_ONLY
    else:
        check.status = NOT_FOUND
    return check


def summarise(checks: list[Check]) -> dict:
    counts = Counter(c.status for c in checks)
    failures = sum(counts[s] for s in FAILURE_STATUSES)
    return {
        "extracted": len(checks),
        "verified": counts[VERIFIED],
        "failures": failures,
        "by_status": {s: counts[s] for s in (VERIFIED, *FAILURE_STATUSES)},
    }


def banner_text(s: dict) -> str:
    detail = ", ".join(f"{v} {k.lower().replace('_', ' ')}" for k, v in s["by_status"].items() if k != VERIFIED and v)
    tail = f" ({detail})" if detail else ""
    return (
        f"{s['extracted']} excerpts extracted · {s['verified']} verified word-for-word "
        f"· {s['failures']} failures{tail}"
    )


def tamper(text: str) -> str:
    """Change exactly one character, for the tamper self-test."""
    if "—" in text:
        return text.replace("—", "-", 1)
    for i, c in enumerate(text):
        if c.isalpha():
            return text[:i] + c.swapcase() + text[i + 1 :]
    return text + "."


# ---------------------------------------------------------------------------
# LLM schemas and prompts (M1, M3)
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    quote_id: str = Field(description="The excerpt id exactly as given, e.g. I2-L17")
    quote_text: str = Field(description="Copied character-for-character from that excerpt: the whole excerpt or a contiguous span of at least 3 words")
    stance: Literal["blocker", "friction", "enabler"] = Field(description="blocker: works against change; friction: day-to-day cost or confusion; enabler: a strength or bright spot")


class Theme(BaseModel):
    name: str
    one_line_summary: str
    evidence: list[Evidence]


class ThemeSet(BaseModel):
    themes: list[Theme]


class QAResult(BaseModel):
    is_answerable: bool
    answer: str
    evidence: list[Evidence]


COPY_RULES = """Rules for every piece of evidence:
- quote_id must be one of the ids in square brackets, exactly as written.
- quote_text must be copied character-for-character from that excerpt: either the whole excerpt or one contiguous span of at least 3 words.
- Do not paraphrase, correct spelling, change punctuation or dashes, add or remove quotation marks, or insert ellipses."""

THEME_PROMPT = """You are analysing interview notes for a consulting engagement. Below are numbered excerpts; each shows [id] (speaker | section) text.

Identify every major theme that recurs across these notes (typically 4 to 8). Themes must come from the text itself. For each theme give a short name, a one-line summary, and EVERY excerpt that supports it. An excerpt may support more than one theme.

{rules}

EXCERPTS:
{catalog}"""

QA_PROMPT = """Answer the question using ONLY the numbered interview excerpts below.

- Answer only what the excerpts directly state. Do not infer, extrapolate, or combine excerpts into claims they do not make.
- If the excerpts do not directly answer the question, set is_answerable to false, answer to an empty string, and evidence to an empty list.
- If they do, write a short answer and cite every excerpt you relied on.

{rules}

QUESTION: {question}

EXCERPTS:
{catalog}"""

NOT_IN_INTERVIEWS = "Not in the interviews."
DEFAULT_MODEL = "gemini-3.8-flash"


RETRY_CODES = (429, 500, 503)
RETRY_WAITS = (3, 8, 15)  # seconds; Gemini's busy/rate-limit errors are usually brief


def _with_retries(call, waits=RETRY_WAITS, sleep=time.sleep):
    """Retry transient Gemini errors (busy, rate-limited); raise anything else at once."""
    for wait in (*waits, None):
        try:
            return call()
        except Exception as e:
            if wait is None or getattr(e, "code", None) not in RETRY_CODES:
                raise
            sleep(wait)


def _generate(prompt: str, schema: type[BaseModel], api_key: str, model: str) -> BaseModel:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema, temperature=0)
    resp = _with_retries(lambda: client.models.generate_content(model=model, contents=prompt, config=config))
    if resp.parsed is not None:
        return resp.parsed
    return schema.model_validate_json(resp.text)


def extract_themes(corpus: Corpus, api_key: str, model: str = DEFAULT_MODEL) -> ThemeSet:
    prompt = THEME_PROMPT.format(rules=COPY_RULES, catalog=catalog_for_prompt(corpus))
    return _generate(prompt, ThemeSet, api_key, model)


def ask(question: str, corpus: Corpus, api_key: str, model: str = DEFAULT_MODEL) -> QAResult:
    prompt = QA_PROMPT.format(rules=COPY_RULES, question=question, catalog=catalog_for_prompt(corpus))
    return _generate(prompt, QAResult, api_key, model)


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------


def verify_themes(themes: ThemeSet, corpus: Corpus) -> list[tuple[Theme, list[Check]]]:
    return [(t, [verify(e.quote_id, e.quote_text, corpus) for e in t.evidence]) for t in themes.themes]


@dataclass
class Answer:
    text: str
    answered: bool
    kept: list[Check]
    dropped: list[Check]
    note: str = ""


def resolve_answer(result: QAResult, corpus: Corpus) -> Answer:
    """Hard override: only verified evidence can carry an answer."""
    if not result.is_answerable:
        return Answer(NOT_IN_INTERVIEWS, False, [], [])
    checks = [verify(e.quote_id, e.quote_text, corpus) for e in result.evidence]
    kept = [c for c in checks if c.ok]
    dropped = [c for c in checks if not c.ok]
    if not kept:
        note = "The model proposed an answer, but none of its citations passed verification." if checks else "The model proposed an answer with no citations."
        return Answer(NOT_IN_INTERVIEWS, False, [], dropped, note)
    return Answer(result.answer, True, kept, dropped)


def save_run(path: str, themes: ThemeSet, model: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": model, "themes": themes.model_dump()}, f, ensure_ascii=False, indent=2)


def load_run(path: str) -> ThemeSet:
    with open(path, encoding="utf-8") as f:
        return ThemeSet.model_validate(json.load(f)["themes"])


def matrix_rows(verified: list[tuple[Theme, list[Check]]], corpus: Corpus) -> list[dict]:
    rows = []
    for theme, checks in verified:
        for ev, c in zip(theme.evidence, checks):
            rows.append(
                {
                    "theme": theme.name,
                    "speaker": c.speaker,
                    "excerpt": c.quote_text,
                    "type": c.type,
                    "stance": ev.stance,
                    "source": f"{c.file}:{c.line}" if c.line else "",
                    "quote_id": c.quote_id,
                    "status": c.status,
                }
            )
    return rows

