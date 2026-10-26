"""Hand-labelled validation set.

The 25 calls are chosen *before* any LLM output exists (fixed seed, stratified), so the
sample can't be biased toward calls the model gets right. Labels are filled in by a human
from the redacted transcripts in labeling/transcripts/, blind to the model's answers.
"""
from pathlib import Path

import pandas as pd

from .preprocess import render_transcript

ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "labeling"
SEED = 7
OUTAGE = (pd.Timestamp("2026-03-10", tz="UTC"), pd.Timestamp("2026-03-19", tz="UTC"))
# Per call type: how many to draw. Mirrors the 30/40/30 mix, rounded to 25.
QUOTA = {"support": 8, "external": 10, "internal": 7}


def period(ts):
    if ts < OUTAGE[0]:
        return "pre-outage"
    return "outage" if ts < OUTAGE[1] else "post-outage"


def select_sample(calls):
    """Stratify by call type x period; always include the 4 incident-flagged external calls."""
    c = calls.assign(period=calls.start.map(period))
    must = c[(c.call_type == "external") & c.incident_flag]
    picked = [must]
    for t, n in QUOTA.items():
        pool = c[(c.call_type == t) & ~c.meeting_id.isin(must.meeting_id)]
        n -= len(must) if t == "external" else 0
        # proportional allocation across periods, at least one per period
        share = pool.period.value_counts(normalize=True)
        alloc = (share * n).round().clip(lower=1).astype(int)
        while alloc.sum() > n:
            alloc[alloc.idxmax()] -= 1
        while alloc.sum() < n:
            alloc[alloc.idxmin()] += 1
        for p, k in alloc.items():
            picked.append(pool[pool.period == p].sample(k, random_state=SEED))
    return pd.concat(picked).sort_values("start")


def write_label_sheet(sample, redacted):
    """Write labeling/label_sheet.csv (blank label columns) and one transcript file per call."""
    tdir = LABEL_DIR / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(sample.itertuples(), 1):
        fname = f"{i:02d}_{r.meeting_id}.txt"
        g = redacted[redacted.meeting_id == r.meeting_id]
        (tdir / fname).write_text(f"{r.title}\n{r.start:%Y-%m-%d}\n\n{render_transcript(g)}\n")
        rows.append({"n": i, "meeting_id": r.meeting_id, "file": f"transcripts/{fname}", "title": r.title,
                     "date": r.start.date(), "period": r.period,
                     "call_type": "", "primary_theme": "", "other_themes": "", "notes": ""})
    sheet = pd.DataFrame(rows)
    path = LABEL_DIR / "label_sheet.csv"
    if path.exists():
        raise FileExistsError(f"{path} exists - refusing to overwrite human labels")
    sheet.to_csv(path, index=False)
    return sheet


def load_labels():
    df = pd.read_csv(LABEL_DIR / "label_sheet.csv", dtype=str).fillna("")
    df["other_themes"] = df.other_themes.str.split(";").apply(lambda xs: [x.strip() for x in xs if x.strip()])
    return df
