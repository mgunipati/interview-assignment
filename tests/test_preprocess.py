"""Redaction must not leak IDs or customer names into anything sent to the LLM."""
import re

import pytest

from transcript_intel.data import load_tables
from transcript_intel.llm import quote_grounded
from transcript_intel.preprocess import mask_ids, redact


@pytest.fixture(scope="module")
def red():
    _, sentences, _ = load_tables()
    return sentences, redact(sentences)


def test_no_id_leaks(red):
    _, r = red
    assert not r.clean_text.str.contains(r"\b[A-Z]{2,4}-\d{4,}").any()
    assert not r.clean_text.str.contains(r"(?i)\b(?:invoice|case number|ticket number)\s*\d{3,}").any()


def test_no_customer_name_leaks(red):
    s, r = red
    for mid, g in r.groupby("meeting_id"):
        for name in set(g.loc[g.side == "customer", "speaker"]):
            first, last = name.split()[0], name.split()[-1]
            token = last if first.endswith(".") else first
            assert not g.clean_text.str.contains(rf"\b{re.escape(token)}\b").any(), (mid, name)


def test_staff_names_kept(red):
    _, r = red
    assert (r.loc[r.side == "aegis", "speaker_label"] == r.loc[r.side == "aegis", "speaker"]).all()


@pytest.mark.parametrize("raw,expected", [
    ("account number BRI-04471", "account number [ID]"),
    ("it's INV-2026-00847.", "it's [ID]."),
    ("invoice 8847 and 8901", "invoice [ID] and [ID]"),
    ("case number 9279", "case number [ID]"),
    ("we pay $18,500 a year", "we pay $18,500 a year"),
])
def test_mask_ids(raw, expected):
    assert mask_ids(raw) == expected


def test_quote_grounding():
    t = "[CUSTOMER] Customer 1: We have zero threat visibility, zero, and it has been three hours."
    assert quote_grounded("we have zero threat visibility, zero", t)
    assert quote_grounded("We have zero threat visibility ... it has been three hours", t)
    assert not quote_grounded("we have no visibility into threats", t)
