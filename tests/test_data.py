"""Guards for the assumptions the data audit established (see docs/data_audit.md)."""
import pytest

from transcript_intel.data import call_type, load_tables, speaker_email


@pytest.fixture(scope="module")
def tables():
    return load_tables()


def test_loads_all_calls(tables):
    calls, sentences, moments = tables
    assert len(calls) == 100
    assert len(sentences) == 4313
    assert len(moments) == 402


def test_call_type_mix(tables):
    calls, _, _ = tables
    assert calls.call_type.value_counts().to_dict() == {"external": 40, "support": 30, "internal": 30}


def test_every_speaker_maps_to_an_attendee(tables):
    _, sentences, _ = tables
    assert sentences.speaker_email.notna().all()
    assert set(sentences.side) == {"aegis", "customer"}


@pytest.mark.parametrize("title,emails,expected", [
    ("Weekly Engineering Standup", ["a.b@aegiscloud.com", "c.d@aegiscloud.com"], "internal"),
    ("Support Case #1 - Foo", ["a.b@aegiscloud.com", "x.y@foo.com"], "support"),
    ("URGENT: Foo - Dashboard Down", ["a.b@aegiscloud.com", "x.y@foo.com"], "support"),
    ("Aegis / Foo - Renewal Discussion", ["a.b@aegiscloud.com", "x.y@foo.com"], "external"),
])
def test_call_type_rules(title, emails, expected):
    assert call_type(title, emails) == expected


def test_speaker_email_conventions():
    emails = ["kevin.obrien@aegiscloud.com", "t.hargrove@redwoodclinical.com"]
    assert speaker_email("Kevin O'Brien", emails) == "kevin.obrien@aegiscloud.com"
    assert speaker_email("Thomas Hargrove", emails) == "t.hargrove@redwoodclinical.com"
