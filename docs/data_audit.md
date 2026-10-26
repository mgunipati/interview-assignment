# Data Audit — Findings and Decisions

**Dataset:** 100 call transcripts, 3 Feb – 28 Apr 2026, one folder per call (`dataset/<meetingId>/`).
**Reproduce:** `notebooks/01_data_audit.ipynb` (every number below is computed there). Loader: `src/transcript_intel/data.py`.
**Format of this doc:** for each audit question — *what I found* → *what I decided because of it*.

---

## TL;DR

The data is clean, internally consistent and clearly **synthetic**. It contains all three call types the brief describes
(30 support · 40 external · 30 internal), so no synthetic augmentation is needed. The two issues that shape the pipeline are:

1. **Transcripts are partial** — they cover a median 29% of the scheduled meeting, and 27% of the provided key moments
   point past the end of the text. → Transcript text is the source of truth; provided summaries are treated as metadata.
2. **Provided topic tags are unusable as categories** — 351 distinct free-text tags, 76% used once.
   → Topic categorisation needs a clustering + LLM-naming step.

---

## 1. Format

| Check | Result |
|---|---|
| Files per call | 6/6 in all 100 calls, all valid JSON |
| Schema consistency | 1 schema per object type (meeting-info, summary, transcript line, event, key moment) |
| meetingId matches folder | ✅ |
| Line index contiguous & time-ordered | ✅ |
| Overlapping turns / empty lines | none |
| `speakers.json` vs transcript turns | identical (redundant file) |
| Line-level sentiment labels | 56% neutral · 27% positive · 17% negative |
| Provided overall label → score | clean bands (very-negative 1.4–1.6 … very-positive 4.7–4.9) |
| Line sentiment vs overall score | r = 0.94 |
| **Transcript coverage (spoken ÷ scheduled)** | **min 15% · median 29% · max 90%** |
| **Key moments timestamped after transcript ends** | **107 / 402** |

**Found:** schema is perfectly consistent, but transcripts are excerpts — e.g. a 35-minute meeting has ~8.6 minutes of text — and many provided key moments reference content we don't have.
**Decided:**
- Treat **transcript text as the source of truth**; use provided `summary` / `keyMoments` / `topics` as unverifiable metadata. Never rely on key-moment timestamps.
- Present `duration` as *scheduled* length, never as talk time.

**Found:** the provided overall sentiment agrees with the line-level labels (r = 0.94).
**Decided:** reuse provided sentiment as the baseline rather than re-scoring from scratch; add value by slicing it — speaker side, call phase, time, account.

---

## 2. Speaker labels

| Check | Result |
|---|---|
| Generic labels ("Speaker 1") | 0 — every line has a real name |
| Speakers per call | 2: 30 · 3: 38 · 4: 23 · 5: 9 |
| Attendee count = speaker count | every call |
| Speakers matched to attendee email | 311 / 311 |
| Side | 224 Aegis-staff appearances · 87 customer appearances |
| Top speaker's talk share | 29%–66% (median 44%) — no monologues |

Email conventions differ by side: Aegis staff use `first.last@aegiscloud.com`, customers use `f.last@customer.com`.
Two speakers carry an honorific ("Dr. Linda Cheng") that the matcher strips.

**Decided:** add a **`side` column (aegis / customer)** to every transcript line. Our staff are polite and positive by default;
measuring *customer* sentiment separately is what makes sentiment actionable.

---

## 3. Actual call-type mix

There is **no call-type field**. Derived with two transparent rules:

1. All attendees on `aegiscloud.com` → **internal**
2. Otherwise title starts `Support Case #` / `URGENT` / `ESCALATION` → **support**, else → **external**

| Type | Calls | Title patterns | Feb | Mar | Apr |
|---|---|---|---|---|---|
| Support | 30 | Support Case # (27), URGENT (2), ESCALATION (1) | 8 | 13 | 9 |
| External | 40 | Aegis / {Customer} – {topic} | 10 | 13 | 17 |
| Internal | 30 | Detect Outage (5), Comply v2 (5), sprint/standup/all-hands… | 8 | 10 | 12 |

Edge cases — external calls that are really incident handling, run by account managers:
Northstar Pharma *Urgent: Detect Outage Impact* · Quantum Edge *Renewal Concerns* · Summit Trust *Platform Concerns* · Helix Data *Post-Incident Review* (scores 2.1–2.4).

**Decided:**
- All three types are present and balanced → **no synthetic data generated**.
- Keep the **rule-based label** — 100% explainable, no model needed, unit-tested.
- Don't relabel the 4 edge cases; add an **`incident_flag`**. Account managers absorbing outage escalations is itself a finding.

---

## 4. Length distribution

| Type | Scheduled min (median) | Words (median) | Lines (median) | Words/min |
|---|---|---|---|---|
| Support | 17.2 | 1,333 | 41 | 157 |
| External | 39.4 | 1,521 | 44 | 158 |
| Internal | 30.4 | 1,160 | 41 | 154 |

- Words per transcript: 952 – 2,711. Outliers (|z| > 2): Harborview *Threat Detection Tuning* (2,711), Blackridge *Renewal Discussion* (2,126), Summit Trust *Platform Concerns* (2,018).
- **Synthetic tells:** speaking rate 149–161 wpm across every call; ASR confidence only 0.88–0.97; 27% of meetings on weekends.

**Decided:**
- Every transcript fits in a single LLM call → **no chunking or truncation**.
- **No insights built on pace, ASR confidence or day-of-week** — they describe the data generator, not the business.

---

## 5. Metadata

| Field | Status | Source / workaround |
|---|---|---|
| Date / time | present | `meeting-info.startTime` |
| Customer account | derivable | attendee email domain — always matches the title |
| Agent / staff | present | 27 `aegiscloud.com` people, named per line |
| Product | derivable | keyword match: Detect 74 · Comply 64 · Protect 38 · Identity 37 calls |
| Call type | derivable | rules in §3 |
| Topics | present but unusable | 351 distinct tags, 266 used once |
| Staff role / team | missing | infer from call-type participation |
| ARR, plan, renewal date | missing | only appears in speech, if at all |
| CSAT, ticket status / resolution | missing | — |

- **32 customer accounts**; calls per account: 1 (8 accounts) · 2 (14) · 3 (6) · 4 (4).
- **20 accounts have both support and external calls.**
- **No staff member appears in all three call types** — e.g. `david.kim` and `elena.vasquez` only take support calls; `aisha.johnson`, `daniel.okafor` are mostly external.

**Decided:**
- Build **account-level timelines** (support pain → renewal conversation) as a headline cross-call-type insight.
- Infer staff role (support / account management / engineering) from participation rather than inventing a field.
- Provided topic tags are too fragmented to be categories → **clustering + LLM naming** for topic categorisation (notebook 02), with the tags used only as a cross-check.

---

## 6. PII

Scanned all transcript lines and provided summaries; every category with hits was checked in context.

| Type | Hits | Examples / context |
|---|---|---|
| Email, phone, SSN, credit card, IP, street address, API key | **0** | — |
| Account / ticket / invoice IDs | 15 | `BRI-04471`, `AGS-229183`, `INV-2026-00847` |
| Case / invoice numbers | 32 | "case number 9279", "invoice 8847 and 8901" |
| Money amounts | 28 | contract values, overage charges ($18,500, $480) |
| Health terms | 15 | "patient portal", "patient data backups" — compliance context, **no actual PHI** |
| Third-party names | 63 | colleagues named but not on the call: Priya (29), Marcus (21), Raj (6) |
| By design | — | full names in every line; corporate emails in metadata; customer company names in titles |

**Decided:** before any text is sent to the Claude API,
- **mask** account / ticket / case / invoice IDs,
- **pseudonymise** customer contact names,
- **keep** company names (account-level insights depend on them) and dollar amounts (pricing signal).
- LLM response cache (`cache/`) is git-ignored because it contains transcript text.

---

## 7. Duplicates and near-duplicates

| Check | Result |
|---|---|
| Exact duplicate transcripts (hash) | 0 |
| Same start time + attendees | 0 |
| Near-duplicates (5-word-shingle Jaccard, 4,950 pairs) | max 0.015; 0 pairs > 0.10 |
| Summary similarity (3-gram Jaccard) | max 0.10 |
| Repeated titles | *Weekly Engineering Standup* ×2, *Detect Team – Sprint Planning* ×2 — recurring meetings, different content |
| Verbatim sentence reuse (≥ 6 words) | 2 sentences — both greetings ("Thank you for calling Aegis Cloud Security support…") |

**Decided:**
- **Keep all 100 calls**; no deduplication.
- **Strip opening/closing pleasantries before clustering** so support calls don't cluster on boilerplate.

---

## Story the data is built around

Visible in titles and scores even before modelling — the pipeline should surface it, not assume it:

- **Feb:** baseline — onboarding, renewals, sprint planning; support calls on SSO/MFA/billing (scores 2.2–3.8).
- **10–18 Mar: Detect outage.** Internal war room → escalation bridge → customer impact assessment → RCA → post-incident review (internal scores 1.8–2.8); simultaneous URGENT support calls (1.4–1.8) and account calls about renewal risk (Quantum Edge, Nova Retail, Summit Trust, Meridian). Followed by *Competitive Threat Assessment – Post Outage*.
- **Apr: Comply v2 launch.** Internal launch calls and customer previews score 4.7–4.8, while *Detect Reliability – 30-Day Review*, *Win/Loss Analysis – Q1* and *Protect Performance – Scalability Concerns* stay at 2.2–2.8.

---

## Summary of decisions

| Finding | Decision |
|---|---|
| Transcripts cover ~29% of scheduled time; 107 key moments point past the text | Transcript text is source of truth; provided summaries = metadata |
| Provided sentiment agrees with line labels (r = 0.94) | Reuse as baseline; add speaker-side / phase / time / account slices |
| All 311 speakers map to attendee emails | `side` column → customer vs Aegis sentiment |
| 30 support / 40 external / 30 internal (derived) | Rule-based call type; no synthetic data; `incident_flag` for 4 edge cases |
| Uniform pace, narrow ASR, 27% weekend calls → synthetic | No insights on pace, ASR or weekday |
| 351 free-text topic tags, 76% singletons | Clustering + LLM naming for categories |
| 20 accounts have support + external calls | Account timelines as a headline insight |
| IDs, names, amounts in text; no hard PII | Mask IDs, pseudonymise contacts before LLM calls |
| No duplicates; only greeting boilerplate | Keep all 100; strip boilerplate before clustering |
