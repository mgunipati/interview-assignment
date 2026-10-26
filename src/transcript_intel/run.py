"""Run the whole pipeline end to end: `python -m transcript_intel.run`.

Uses the LLM cache in cache/llm/, so a re-run is free and offline; calls not in the cache go to the
Claude API (needs ANTHROPIC_API_KEY in .env). Prints what each stage produced and writes reports/.
"""
import argparse
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
ASOF = pd.Timestamp("2026-04-29", tz="UTC")


def stage(n, title):
    print(f"\n[{n}/7] {title}")
    return time.time()


def done(t0, msg):
    print(f"      {msg}  ({time.time() - t0:.1f}s)")


def live_example(title_fragment):
    """Send one call to Claude fresh (no cache) and show the extraction next to the grounding check."""
    from . import llm, prompts
    from .data import load_tables
    from .preprocess import redact, render_transcript
    from .schemas import CallExtraction

    calls, sentences, _ = load_tables()
    match = calls[calls.title.str.contains(title_fragment, case=False, regex=False)]
    if match.empty:
        raise SystemExit(f"No call title contains {title_fragment!r}")
    r = match.iloc[0]
    transcript = render_transcript(redact(sentences[sentences.meeting_id == r.meeting_id]))
    print(f"Live extraction (no cache): {r.title}  [{r.call_type}]\n"
          f"Sending {len(transcript.split()):,} words of redacted transcript to {llm.MODEL}...")
    t0 = time.time()
    x = llm.call("extract", r.meeting_id, *prompts.extraction_prompt(r.title, r.call_type, transcript),
                 CallExtraction, effort="medium", use_cache=False)
    print(f"Done in {time.time() - t0:.0f}s · ${llm.cost_so_far():.3f}\n\nPurpose: {x.primary_purpose}\n"
          f"Churn risk: {x.churn_risk} · resolution: {x.resolution}\n")
    for i in x.issues:
        ok = "FOUND IN TRANSCRIPT" if llm.quote_grounded(i.evidence_quote, transcript) else "NOT FOUND"
        print(f"- [{i.product} · {i.severity}] {i.statement}\n    \u201c{i.evidence_quote}\u201d  -> {ok}")
    if x.competitors:
        print("\nCompetitors: " + "; ".join(f"{c.name} ({c.context})" for c in x.competitors))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", metavar="TITLE", help="extract one call live from the Claude API (bypasses the cache)")
    args = ap.parse_args()
    if args.live:
        return live_example(args.live)
    from . import insights, llm, topics
    from .data import load_tables
    from .preprocess import redact
    from .sentiment import call_sentiment

    start = time.time()
    REPORTS.mkdir(exist_ok=True)

    t = stage(1, "Load and redact")
    calls, sentences, _ = load_tables()
    red = redact(sentences)
    done(t, f"{len(calls)} calls, {len(sentences):,} lines; IDs masked, customers pseudonymised, "
            f"{red.is_boilerplate.sum()} boilerplate lines flagged")

    t = stage(2, "Call type: rules, checked by Claude")
    ct = topics.check_call_types(calls, red)
    done(t, f"{calls.call_type.value_counts().to_dict()}; rules vs LLM agree on {ct.agrees.sum()}/{len(ct)}; "
            f"incident-related: {ct.incident_flag.sum()} by title, {ct.llm_incident.sum()} by content")

    t = stage(3, "Extract issues with evidence quotes (Claude, Pydantic schema)")
    calls_x, issues = topics.extract(calls, red)
    done(t, f"{len(issues)} issues, {issues.grounded.mean():.0%} of quotes found verbatim in the transcript")

    t = stage(4, "Discover themes: embeddings + UMAP + HDBSCAN, named by Claude")
    iss, terms, _ = topics.discover_themes(issues, min_cluster_size=8, min_samples=2, n_neighbors=10)
    named = topics.name_clusters(iss, terms)
    done(t, f"{len(named)} clusters, {(iss.cluster == -1).mean():.0%} noise -> curated into taxonomy/taxonomy.json")

    t = stage(5, "Classify every call into the curated taxonomy (multi-label)")
    tax = topics.load_taxonomy()
    assigned = topics.classify(calls, red, tax)
    kept = assigned[(assigned.confidence >= 0.6) | assigned.is_primary]
    top = kept.groupby("theme").meeting_id.nunique().sort_values(ascending=False).head(3)
    done(t, f"{len(tax)} themes, {len(kept) / 100:.1f} labels/call, quotes grounded {assigned.grounded.mean():.0%}; "
            f"top: " + ", ".join(f"{k} ({v})" for k, v in top.items()))

    t = stage(6, "Sentiment by call type and speaker side")
    cs = call_sentiment(calls, sentences).merge(calls_x[["meeting_id", "churn_risk", "resolution"]])
    by = cs.groupby("call_type")[["sentiment_score", "net_customer", "net_aegis"]].mean().round(2)
    done(t, "\n      " + by.to_string().replace("\n", "\n      "))

    t = stage(7, "Stakeholder insights")
    risks = insights.triage_risks(calls, red)
    backtest = insights.risk_radar(risks, issues, asof=pd.Timestamp("2026-03-10", tz="UTC"))
    now = insights.risk_radar(risks, issues, asof=ASOF)
    comp = insights.competitor_mentions(calls_x)
    health = insights.account_health(cs, calls_x, comp, ASOF)
    _, demand = insights.feature_demand(calls_x)
    b = backtest.iloc[0]
    done(t, f"risk radar backtest (data before 10 Mar) #1: {b.risk[:70]}… "
            f"score {b.score:.0f} vs next {backtest.iloc[1].score:.0f}\n"
            f"      watch list: {(health.watch_score >= 5).sum()} of {len(health)} accounts, "
            f"{health[health.watch_score >= 5].open_loop.sum()} open loops\n"
            f"      top ask from at-risk accounts: {demand.index[0]}")

    pd.DataFrame(kept).to_csv(REPORTS / "call_theme_labels_kept.csv", index=False)
    health.to_csv(REPORTS / "account_watchlist.csv", index=False)
    now.drop(columns=["grounded"]).to_csv(REPORTS / "risk_radar.csv", index=False)
    demand.to_csv(REPORTS / "feature_demand.csv")
    u = llm.usage_totals
    print(f"\nDone in {time.time() - start:.0f}s · LLM responses: {u['cached']} from cache, {u['calls']} new "
          f"(${llm.cost_so_far():.2f}) · reports written to {REPORTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
