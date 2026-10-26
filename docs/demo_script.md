# Demo video — script and checklist (target 8–9 minutes)

The brief asks for: the pipeline running, a walk through the outputs, and the insights / visualisations.
Structure: **frame (1 min) → run it (2.5 min) → outputs (4 min) → trust & close (1 min)**.
Speak to a product/engineering leader: what it shows and why it matters, not the code line by line.

---

## Before you record

**Set up (10 min)**
- [ ] Notebooks are already executed with outputs saved — do **not** re-run them on camera (UMAP takes time; outputs are what matter).
- [ ] Open in this order, each in its own tab/window:
  1. Deck, slide 2 ("Four findings, three decisions")
  2. PyCharm with the project open, `README.md` showing
  3. Terminal at the project root with the venv active: `source .venv/bin/activate && clear`
  4. `notebooks/02_topic_pipeline.ipynb` · `03_sentiment.ipynb` · `04_stakeholder_insights.ipynb` (Jupyter or PyCharm)
- [ ] Terminal font ≥ 16pt, editor/notebook zoom ~125%, window ≤ 1440 px wide so text is readable.
- [ ] Do a dry run of the live command once (costs ~$0.04): `python -m transcript_intel.run --live "Northstar Pharma - Urgent"`
- [ ] Do Not Disturb on; close Slack/mail.

**Never show on screen:** the `.env` file, `cat .env`, or any terminal history containing the key. Collapse `.env` in the PyCharm project tree (or it's fine as a filename, just never open it).

**Recording:** macOS `⌘⇧5` → *Record Entire Screen* (or Selected Portion), Options → your microphone. Loom works too.

---

## 0:00 – 1:00 · Frame the problem  — *Deck, slide 2*

> "This is Transcript Intelligence: a pipeline that turns call transcripts into decisions for support, sales, product and engineering leaders. I'll show it running on the 100 sample calls, walk through the outputs, and end on how much to trust them.
>
> The headline from the data: the March Detect outage was visible in our own calls five weeks before it happened; it stayed in customer conversations for 49 days, not six hours; a successful Comply launch hid that from the averages; and Protect is showing the same warning pattern today. Let me show you where that comes from."

## 1:00 – 1:45 · What's in the repo  — *PyCharm, README.md*

Scroll slowly through the Layout block.

> "It's a normal Python package. The `dataset` folder is the 100 calls as provided. `src/transcript_intel` is the pipeline — loading, redaction, the LLM steps, clustering, sentiment, insights. Four notebooks tell the story in order: the data audit, topics, sentiment, and stakeholder insights. And `docs/data_audit.md` records every decision the audit drove — for example, there's no call-type field, so I derive it; and transcripts are excerpts, so the transcript text is my source of truth."

Optional (10 s): open `dataset/<any id>/transcript.json` to show a line with speaker, time and sentiment label.

## 1:45 – 3:00 · Run the pipeline  — *Terminal*

```bash
python -m transcript_intel.run
```

It finishes in ~6 seconds. While it prints, narrate each stage as it appears:

> "Seven stages. One: load and redact — account and ticket IDs are masked and customer contacts pseudonymised before anything goes to an LLM.
> Two: call type. Rules on attendee domains and titles give 30 support, 40 external, 30 internal, and Claude reading the content agrees on all 100. But look at the incident line — titles flag 15 calls, content finds 37. That's the first sign of how far the outage reached.
> Three: Claude extracts 529 issues, each with a verbatim quote — and every quote is machine-checked against the transcript: 100% found.
> Four: those issues are embedded and clustered — 24 themes — which I curated into a fixed taxonomy.
> Five: every call is classified into that taxonomy. Six: sentiment — note the customer column versus the Aegis column. Seven: the insights — the risk-radar backtest, the account watch list, and the top ask from at-risk accounts.
> It ran in seconds because every LLM response is cached — re-runs are free and reproducible."

## 3:00 – 3:45 · Watch Claude do it live  — *Terminal*

```bash
python -m transcript_intel.run --live "Northstar Pharma - Urgent"
```

~10 seconds, ~$0.04. While waiting:

> "That was the cached run. Here's one call going to the Claude API live, so you can see what the model actually does. This is an account call during the outage."

When it prints:

> "Each issue is generic enough to compare across calls, tagged with product and severity, and backed by a quote — and the tool checks every quote is really in the transcript. It also picked up that the customer's CISO asked for a competitive evaluation of SentinelShield. That structure is what everything downstream is built on."

*(If the API is slow or errors, skip this segment — the cached run already showed the pipeline.)*

## 3:45 – 5:00 · Topics: why hybrid  — *notebook 02*

1. **§3 baseline** — the silhouette chart.
   > "First I tried clustering whole calls. Separation is weak, and each call gets exactly one label — but calls are about several things."
2. **§5** — the UMAP scatter of 529 issue statements.
   > "Clustering *issues* instead of calls gives sharp themes. But the clusters move between random seeds, and the outage splits into five clusters — so clusters are for discovery, and a human curates the final taxonomy."
3. **§7** — the theme × call-type heatmap, then the *Examples per theme* table.
   > "19 themes in five groups, each with an owner. The two outage themes lead, across all three call types — and every label comes with the quote that justifies it."

## 5:00 – 6:15 · Sentiment: what the trends mean  — *notebook 03*

1. **§3** — the "Staff tone masks customer tone" bar chart.
   > "On support calls customers are at minus 0.37, our staff at zero. A whole-call score would flatter support, so every trend uses customer lines only."
2. **§4** — the weekly line chart with the outage band and Comply v2 line.
   > "Every call type collapses in outage week, then the Comply launch is the high point of the quarter."
3. **§5** — the post-outage scatter (red vs green).
   > "This is the key chart. After the outage, calls still about it score 3.17; Comply calls 4.39. The average says 'recovered'. It wasn't — those red dots go to the end of April."
4. **§8** — robustness table, 5 seconds.
   > "And every comparison holds when I swap in an independent sentiment model."

## 6:15 – 8:00 · Beyond the brief  — *notebook 04*

1. **§1a** — the precursor table and the printed quotes.
   > "Starting from the root-cause analysis's own words and searching backwards: an engineer described the exact failure on 5 February; a January failure had the same root cause; on 16 February the fix was 'sixty-forty'; on 2 March it was cut to a design spike. Five calls, five weeks."
2. **§1b** — the two backtest tables.
   > "Could a tool have caught it? Counting mentions ranks it sixth — the warning wasn't loud. Having Claude read each internal call for unresolved risks ranks it first, eight times the next risk, using only data from before the outage. One incident is one backtest — but the method works."
3. **§1c** — today's radar.
   > "Run it on today's data and Protect looks like Detect did in February: a critical scheduler risk customers are already feeling, and 'single points of failure like Detect' discussed but not scheduled."
4. **§3** — watch-list table (scroll to the open-loop column).
   > "For CS: 14 of 32 accounts flagged, each point a readable reason; six are open loops — a bad support call and no follow-up since."
5. **§4a** — the demand bar chart.
   > "And what at-risk customers ask for isn't features — it's RCA reports, outage notifications and SLAs."

## 8:00 – 8:45 · How much to trust it, and close  — *Terminal, then deck slide 16*

```bash
pytest -q
```

> "Tests guard the assumptions — redaction leaks, call-type rules, speaker mapping.
> What's validated: every one of 987 evidence quotes is in its transcript; call type agrees 100 out of 100; sentiment conclusions hold under an independent model. What isn't yet: the topic labels against 25 calls I labelled by hand — *[say the result]*; the radar is one backtest; there are no renewal outcomes to validate the watch list. The whole thing costs about nine cents a call, so it can run on every call, every week. Thanks."

---

## If you're running long
Cut, in this order: the dataset JSON peek (1:00), §8 robustness (notebook 03), §4a demand (notebook 04). Don't cut the live run or the §5 recovery chart — they carry "show it running" and the main insight.

## Likely questions to have ready (for the live Q&A)
- *Why not just use the provided topic tags?* — 351 tags, 76% used once (audit §5).
- *How do you know the LLM isn't hallucinating?* — quote grounding (100% of 987), call-type agreement, independent sentiment model.
- *Why Claude for labels but rules for call type?* — rules matched the LLM 100/100 and are free and deterministic; the LLM is the audit.
- *What would you do with real data?* — validate the radar on incident history, the watch score on renewals; weekly re-clustering to catch new themes.
- *The Keystone "redundancy in place" quote — is that fair?* — it's a gap between customer messaging and engineering status, shown neutrally; the war room owned the prioritisation.
