"""Clean, speaker-normalise and PII-redact transcripts before modelling.

Decisions come from the data audit (docs/data_audit.md):
- IDs (account / ticket / case / invoice numbers) are masked: no analytic value, real exposure.
- Customer contacts are pseudonymised per call ("Customer 1"); Aegis staff keep their names
  (internal, and needed for role/ownership insights). Company names and dollar amounts are kept.
- Greeting / sign-off boilerplate is flagged so clustering can ignore it; the LLM still sees it.
"""
import re

import pandas as pd

ID_PATTERNS = [
    re.compile(r"\b[A-Z]{2,4}-\d[\d-]{3,}\b"),                                   # BRI-04471, INV-2026-00847
    re.compile(r"(?i)\b((?:case|ticket|invoice|account)\s*(?:number|no\.?|#)?\s*)\d{3,}\b"),
    re.compile(r"(?i)(\binvoices?\s+\[ID\]\s+and\s+)\d{3,}\b"),                   # "invoice 8847 and 8901"
]
# Short opening/closing pleasantries ("Can you hear me?", "Thanks, bye")
BOILERPLATE = re.compile(
    r"(?i)^(hey|hi|hello|morning|afternoon|thanks|thank you|bye|cheers|talk soon|sounds good|"
    r"great|perfect|alright|okay|yeah|can you hear|you're on mute|loud and clear|"
    r"thank you for calling)\b"
)
EDGE_LINES = 4          # boilerplate is only looked for in the first/last N lines
MAX_BOILERPLATE_WORDS = 25


def mask_ids(text):
    text = ID_PATTERNS[0].sub("[ID]", text)
    text = ID_PATTERNS[1].sub(lambda m: m.group(1) + "[ID]", text)
    return ID_PATTERNS[2].sub(lambda m: m.group(1) + "[ID]", text)


def _name_regex(names):
    """Match full names and first names, longest first, with optional possessive."""
    parts = sorted({p for n in names for p in (n, n.split()[-1] if n.split()[0].endswith(".") else n.split()[0])},
                   key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(map(re.escape, parts)) + r")\b") if parts else None


def redact(sentences):
    """Return a copy of `sentences` with `speaker_label` and `clean_text` columns."""
    out = []
    for mid, g in sentences.groupby("meeting_id", sort=False):
        customers = list(dict.fromkeys(g.loc[g.side == "customer", "speaker"]))
        alias = {n: f"Customer {i + 1}" for i, n in enumerate(customers)}
        for n in customers:  # first name alone, and surname after an honorific ("Dr. Cheng")
            first, last = n.split()[0], n.split()[-1]
            alias.setdefault(last if first.endswith(".") else first, alias[n])
        rx = _name_regex(alias)
        g = g.copy()
        g["speaker_label"] = [alias.get(s, s) for s in g.speaker]
        g["clean_text"] = [mask_ids(rx.sub(lambda m: alias[m.group(1)], t) if rx else t) for t in g.text]
        n = len(g)
        pos = range(n)
        g["is_boilerplate"] = [
            (i < EDGE_LINES or i >= n - EDGE_LINES)
            and len(t.split()) <= MAX_BOILERPLATE_WORDS
            and bool(BOILERPLATE.match(t))
            for i, t in zip(pos, g.text)
        ]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def render_transcript(g, include_boilerplate=True):
    """Format one call's redacted lines as 'ROLE Name: text' for an LLM prompt."""
    if not include_boilerplate:
        g = g[~g.is_boilerplate]
    return "\n".join(f"[{'AEGIS' if s == 'aegis' else 'CUSTOMER'}] {lbl}: {t}"
                     for s, lbl, t in zip(g.side, g.speaker_label, g.clean_text))


def call_documents(red, include_boilerplate=False):
    """One redacted text per call (used for embeddings / clustering)."""
    return red.groupby("meeting_id", sort=False).apply(
        lambda g: " ".join(g.loc[include_boilerplate | ~g.is_boilerplate, "clean_text"]))
