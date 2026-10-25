"""Tests for ingest and the verifier. No LLM calls; no case data required.

If the interview files are present in ./data (git-ignored), the extra
case-file checks run too.
"""

import glob
import os

import pytest

import core
from core import (
    BAD_ID,
    NORMALISED_ONLY,
    NOT_FOUND,
    TOO_SHORT,
    VERIFIED,
    WRONG_LINE,
    Evidence,
    QAResult,
    ingest,
    resolve_answer,
    tamper,
    verify,
)

SAMPLE = """ACME — INTERVIEW NOTES (1 of 2) | Chief Widget Officer

PAIN POINTS
- Approvals take far too long — everything goes to the top
- "We are one team on paper but three teams in practice"

STRENGTHS
- People care deeply about the mission
"""

HEADERLESS = """Decisions wait two weeks for the council
Nobody knows who owns the budget line
"""


@pytest.fixture
def corpus():
    return ingest({"interview_1_Chief_Widget_Officer.txt": SAMPLE, "interview_2_Head_of_Things.txt": HEADERLESS})


def test_ingest_indexes_bullets_with_real_line_numbers(corpus):
    ids = list(corpus.catalog)
    assert ids[:3] == ["I1-L4", "I1-L5", "I1-L8"]
    e = corpus.catalog["I1-L4"]
    assert e.speaker == "Chief Widget Officer"
    assert e.section == "PAIN POINTS"
    assert e.text == "Approvals take far too long — everything goes to the top"
    assert SAMPLE.split("\n")[e.line - 1].endswith(e.text)


def test_direct_speech_vs_interviewer_note(corpus):
    assert corpus.catalog["I1-L5"].type == "direct_speech"
    assert corpus.catalog["I1-L4"].type == "interviewer_note"


def test_headerless_file_falls_back_to_filename_and_lines(corpus):
    assert corpus.speakers["interview_2_Head_of_Things.txt"] == "Head of Things"
    assert corpus.catalog["I2-L1"].text == "Decisions wait two weeks for the council"
    assert any("no '| Speaker' header" in w for w in corpus.warnings)


def test_verified_whole_and_span(corpus):
    assert verify("I1-L4", "Approvals take far too long — everything goes to the top", corpus).status == VERIFIED
    assert verify("I1-L4", "everything goes to the top", corpus).status == VERIFIED


def test_tamper_one_character_fails(corpus):
    real = corpus.catalog["I1-L4"].text
    fake = tamper(real)
    assert fake != real and len(fake) == len(real)
    assert verify("I1-L4", fake, corpus).status == NOT_FOUND


def test_wrong_line_is_caught_and_located(corpus):
    c = verify("I1-L8", "everything goes to the top", corpus)
    assert c.status == WRONG_LINE
    assert c.found_in == ["I1-L4"]


def test_curly_quotes_are_normalised_only_never_verified(corpus):
    c = verify("I1-L5", "“We are one team on paper but three teams in practice”", corpus)
    assert c.status == NORMALISED_ONLY
    assert not c.ok


def test_short_whole_line_is_verified_but_short_fragment_is_not():
    c = ingest({"interview_1_X.txt": "NOTES | Head of Things\n\nSKILLS\n- Writing\n- Critical thinking and diplomacy\n"})
    assert verify("I1-L4", "Writing", c).status == VERIFIED
    assert verify("I1-L5", "Critical thinking", c).status == TOO_SHORT


def test_bad_id_and_too_short(corpus):
    assert verify("I9-L1", "People care deeply about the mission", corpus).status == BAD_ID
    assert verify("I1-L8", "mission", corpus).status == TOO_SHORT


def test_qa_override_refuses_when_no_citation_survives(corpus):
    fake = QAResult(is_answerable=True, answer="Budget is $5m", evidence=[Evidence(quote_id="I1-L4", quote_text="Budget is five million dollars", stance="friction")])
    a = resolve_answer(fake, corpus)
    assert not a.answered and a.text == core.NOT_IN_INTERVIEWS and len(a.dropped) == 1


def test_qa_keeps_verified_and_lists_dropped(corpus):
    res = QAResult(
        is_answerable=True,
        answer="Approvals go to the top.",
        evidence=[
            Evidence(quote_id="I1-L4", quote_text="everything goes to the top", stance="blocker"),
            Evidence(quote_id="I1-L4", quote_text="everything goes to the very top", stance="blocker"),
        ],
    )
    a = resolve_answer(res, corpus)
    assert a.answered and len(a.kept) == 1 and len(a.dropped) == 1


def test_unanswerable_is_plain_refusal(corpus):
    a = resolve_answer(QAResult(is_answerable=False, answer="", evidence=[]), corpus)
    assert a.text == "Not in the interviews."


DATA = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "*.txt")))


@pytest.mark.skipif(len(DATA) != 5, reason="case files not present in ./data")
def test_case_files_ingest():
    files = {os.path.basename(p): open(p, encoding="utf-8").read() for p in DATA}
    c = ingest(files)
    assert len(c.speakers) == 5 and not c.warnings
    assert len(c.catalog) == 163
    assert sum(e.type == "direct_speech" for e in c.catalog.values()) == 13
    # every catalog entry is itself verifiable against its own line
    assert all(verify(e.id, e.text, c).status == VERIFIED for e in c.catalog.values() if len(e.text.split()) >= 3)


class _Busy(Exception):
    def __init__(self, code):
        self.code = code


def test_retries_transient_errors_then_succeeds():
    calls, waits = [], []

    def call():
        calls.append(1)
        if len(calls) < 3:
            raise _Busy(503)
        return "ok"

    assert core._with_retries(call, waits=(1, 2, 3), sleep=waits.append) == "ok"
    assert waits == [1, 2]


def test_does_not_retry_permanent_errors():
    waits = []

    def call():
        raise _Busy(404)

    with pytest.raises(_Busy):
        core._with_retries(call, waits=(1, 2), sleep=waits.append)
    assert waits == []


def test_gives_up_after_last_retry():
    waits = []

    def call():
        raise _Busy(503)

    with pytest.raises(_Busy):
        core._with_retries(call, waits=(1, 2), sleep=waits.append)
    assert waits == [1, 2]
