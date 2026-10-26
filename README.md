# Transcript Intelligence

Take-home assignment: categorise call transcripts by topic, analyse sentiment across call types,
and surface insights for support, sales, product and engineering leaders.

## Layout

```
dataset/                 100 raw calls, one folder each (provided, unmodified)
src/transcript_intel/    pipeline package
    data.py              load raw JSON → calls / sentences / moments DataFrames; call-type + speaker-side rules
    preprocess.py        PII redaction (IDs masked, customer contacts pseudonymised), boilerplate flags
    embed.py             local sentence embeddings (all-mpnet-base-v2), chunk-mean for long transcripts
    cluster.py           TF-IDF / embedding baselines, UMAP + HDBSCAN, class-based TF-IDF terms
    schemas.py           Pydantic schemas for every LLM step
    prompts.py           prompts for every LLM step
    llm.py               Claude client: structured output, refusal fallback, disk cache, quote grounding
    topics.py            hybrid topic pipeline (call-type check → extraction → clusters → taxonomy → classify → evaluate)
    validation.py        stratified 25-call sample + blind human label sheet
    sentiment.py         net sentiment per call / speaker side / call third; outage and launch periods
    insights.py          risk triage + radar, precursor search, feature demand, competitor mentions, account health
    run.py               end-to-end runner (python -m transcript_intel.run)
notebooks/
    01_data_audit.ipynb  data audit: format, speakers, call-type mix, length, metadata, PII, duplicates
    02_topic_pipeline.ipynb  hybrid topic categorisation: rules → LLM extraction → clusters → curated taxonomy → classify → validate
    03_sentiment.ipynb   sentiment by call type, speaker side, time, call arc and topic; robustness vs an independent model
    04_stakeholder_insights.ipynb  risk radar (backtested), outage blast radius, account watch list, demand & competitors
docs/
    data_audit.md        audit findings and the decisions they drove (reference doc)
    demo_script.md       narration script + checklist for the demo video
taxonomy/                taxonomy_draft.json (LLM) → taxonomy.json (human-curated)
labeling/                label_sheet.csv + redacted transcripts for the 25 hand-labelled calls
tests/                   guards for audit assumptions (call-type rules, speaker mapping, counts)
reports/                 call_topics.csv (one row per call), issues.csv, call_theme_labels.csv, call_sentiment.csv,
                         account_watchlist.csv, risk_radar.csv, feature_demand.csv, competitor_mentions.csv
cache/                   LLM + embedding cache (git-ignored) — notebooks re-run offline once populated
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,llm]"
.venv/bin/pytest
.venv/bin/jupyter lab notebooks/
```

LLM steps use the Claude API: `export ANTHROPIC_API_KEY=...`.

## Run the pipeline

```bash
.venv/bin/python -m transcript_intel.run                     # all 7 stages from cache (~6 s, free)
.venv/bin/python -m transcript_intel.run --live "Northstar"  # one call through the Claude API, live (~$0.04)
```

## Quick use

```python
from transcript_intel import load_tables
calls, sentences, moments = load_tables()
calls.groupby("call_type").sentiment_score.mean()
```

## Key data decisions

See [docs/data_audit.md](docs/data_audit.md). In short: transcripts are partial excerpts, so transcript text is the
source of truth; call type is derived by rules (30 support / 40 external / 30 internal); every line is tagged
customer vs Aegis; IDs and contact names are masked before any LLM call.
