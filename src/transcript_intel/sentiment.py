"""Sentiment features built from line-level labels.

Net sentiment = mean over lines of (+1 positive, 0 neutral, -1 negative), so it runs from -1 to +1.
It is computed per speaker side because Aegis staff are polite by default and dilute customer signal.
"""
import numpy as np
import pandas as pd

VALUE = {"positive": 1, "neutral": 0, "negative": -1}
OUTAGE_START = pd.Timestamp("2026-03-10", tz="UTC")
OUTAGE_END = pd.Timestamp("2026-03-19", tz="UTC")   # last outage-response call is 18 Mar
COMPLY_V2_GA = pd.Timestamp("2026-04-01", tz="UTC")
PERIODS = ["pre-outage", "outage", "post-outage"]


def period(ts):
    if ts < OUTAGE_START:
        return "pre-outage"
    return "outage" if ts < OUTAGE_END else "post-outage"


def add_line_values(sentences, label_col="sentiment"):
    s = sentences.copy()
    s["value"] = s[label_col].map(VALUE)
    # position in the call, 0..1, and which third of the call the line falls in
    s["pos"] = s.groupby("meeting_id").idx.transform(lambda x: x / x.max())
    s["third"] = pd.cut(s.pos, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["start", "middle", "end"])
    return s


def call_sentiment(calls, sentences, label_col="sentiment"):
    """One row per call: provided score, net sentiment overall and per side, politeness gap, period."""
    s = add_line_values(sentences, label_col)
    side = s.groupby(["meeting_id", "side"]).value.mean().unstack().rename(
        columns={"customer": "net_customer", "aegis": "net_aegis"})
    out = calls.join(s.groupby("meeting_id").value.mean().rename("net_all"), on="meeting_id").join(side, on="meeting_id")
    out["gap"] = out.net_aegis - out.net_customer          # how much staff tone masks customer tone
    out["period"] = pd.Categorical(out.start.map(period), PERIODS, ordered=True)
    out["week"] = out.start.dt.tz_localize(None).dt.to_period("W").dt.start_time
    return out


def call_arcs(sentences, label_col="sentiment"):
    """Net sentiment per call, side and third of the call (start / middle / end)."""
    s = add_line_values(sentences, label_col)
    return s.groupby(["meeting_id", "side", "third"], observed=True).value.mean().unstack().reset_index()
