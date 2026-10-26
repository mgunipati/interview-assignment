"""Load the raw transcript dataset into pandas DataFrames.

Each call lives in dataset/<meetingId>/ with six JSON files. This module flattens
them into three tables that the rest of the project builds on:

    calls      one row per meeting (metadata + provided summary fields)
    sentences  one row per transcript line
    moments    one row per provided key moment
"""
import json
import re
from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
INTERNAL_DOMAIN = "aegiscloud.com"
CORE_PRODUCTS = ["Detect", "Protect", "Identity", "Comply"]
FILES = ["meeting-info", "events", "speaker-meta", "speakers", "summary", "transcript"]

# Support calls are recognisable by title; everything else with a customer attendee
# is an account-management ("external") call. See notebooks/01_data_audit.ipynb and docs/data_audit.md.
SUPPORT_TITLE = re.compile(r"^(Support Case #|URGENT|ESCALATION)", re.I)
INCIDENT_TITLE = re.compile(r"urgent|outage|incident|concern|escalat", re.I)
HONORIFIC = re.compile(r"^(Dr|Mr|Mrs|Ms|Prof)\.?\s+", re.I)


def load_raw(dataset_dir=DATASET_DIR):
    """Return {meetingId: {file_name: parsed_json}} for every call folder."""
    raw = {}
    for d in sorted(Path(dataset_dir).iterdir()):
        if d.is_dir():
            raw[d.name] = {f: json.loads((d / f"{f}.json").read_text()) for f in FILES}
    return raw


def call_type(title, emails):
    """Rule-based call type: internal / support / external."""
    if all(e.endswith("@" + INTERNAL_DOMAIN) for e in emails):
        return "internal"
    if SUPPORT_TITLE.match(title):
        return "support"
    return "external"


def _norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def speaker_email(name, emails):
    """Match a display name to an attendee email.

    Aegis staff use first.last@, customers use f.last@ - both are tried.
    Honorifics ("Dr. Linda Cheng") are dropped first.
    """
    name = HONORIFIC.sub("", name)
    first, *_, last = name.split()
    for e in emails:
        local = _norm(e.split("@")[0])
        if local in (_norm(name), _norm(first[0] + last)):
            return e
    return None


def build_tables(raw):
    calls, sentences, moments = [], [], []
    for mid, f in raw.items():
        mi, sm, tr = f["meeting-info"], f["summary"], f["transcript"]["data"]
        emails = mi["allEmails"]
        ctype = call_type(mi["title"], emails)
        accounts = sorted({e.split("@")[1] for e in emails} - {INTERNAL_DOMAIN})
        text = " ".join(s["sentence"] for s in tr)
        spoken_end = tr[-1]["endTime"]
        calls.append({
            "meeting_id": mid,
            "title": mi["title"],
            "start": pd.Timestamp(mi["startTime"]),
            "call_type": ctype,
            "incident_flag": bool(INCIDENT_TITLE.search(mi["title"])),
            "account": accounts[0] if accounts else None,
            "n_attendees": len(emails),
            "n_speakers": len(f["speaker-meta"]),
            "scheduled_min": mi["duration"],
            "spoken_min": spoken_end / 60,
            "coverage": spoken_end / (mi["duration"] * 60),
            "n_turns": len(tr),
            "n_words": len(text.split()),
            "products": [p for p in CORE_PRODUCTS if re.search(rf"\b{p}\b", text)],
            "overall_sentiment": sm["overallSentiment"],
            "sentiment_score": sm["sentimentScore"],
            "topics": sm["topics"],
            "action_items": sm["actionItems"],
            "summary": sm["summary"],
            "emails": emails,
        })
        email_of = {n: speaker_email(n, emails) for n in f["speaker-meta"].values()}
        for s in tr:
            em = email_of[s["speaker_name"]]
            sentences.append({
                "meeting_id": mid,
                "call_type": ctype,
                "idx": s["index"],
                "speaker": s["speaker_name"],
                "speaker_email": em,
                "side": None if em is None else ("aegis" if em.endswith(INTERNAL_DOMAIN) else "customer"),
                "start_s": s["time"],
                "end_s": s["endTime"],
                "text": s["sentence"],
                "sentiment": s["sentimentType"],
                "asr_conf": s["averageConfidence"],
            })
        for m in sm["keyMoments"]:
            moments.append({"meeting_id": mid, "call_type": ctype, "time_s": m["time"],
                            "type": m["type"], "speaker": m["speaker"], "text": m["text"],
                            "beyond_transcript": m["time"] > spoken_end + 5})
    return pd.DataFrame(calls), pd.DataFrame(sentences), pd.DataFrame(moments)


def load_tables(dataset_dir=DATASET_DIR):
    return build_tables(load_raw(dataset_dir))
