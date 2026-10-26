"""Stakeholder insights built on the extraction / classification / sentiment layers (notebook 04)."""
import numpy as np
import pandas as pd

from . import cluster as cl
from . import llm, prompts, topics
from .embed import embed
from .preprocess import render_transcript
from .schemas import RiskTriage

IMPACT = {"low": 1, "medium": 2, "high": 3, "critical": 4}
OPEN_STATUS = {"none", "discussed_only", "deferred", "scheduled"}   # not yet being built


# ---- 1. early-warning risk radar ------------------------------------------------------
def triage_risks(calls, redacted):
    """LLM pass over internal calls: unresolved operational risks with impact and mitigation status."""
    internal = calls[calls.call_type == "internal"]
    tx = {mid: render_transcript(g) for mid, g in redacted.groupby("meeting_id", sort=False)}
    res = llm.run_many("risk_triage", {r.meeting_id: r for r in internal.itertuples()},
                       lambda r: prompts.risk_triage_prompt(r.title, tx[r.meeting_id]), RiskTriage, effort="medium")
    rows = [{"meeting_id": mid, **rk.model_dump(), "grounded": llm.quote_grounded(rk.evidence_quote, tx[mid])}
            for mid, x in res.items() if x for rk in x.risks]
    return internal[["meeting_id", "start", "title"]].merge(pd.DataFrame(rows), on="meeting_id")


def risk_radar(risks, issues, asof, link=0.6, window_days=42):
    """Rank risks known as of `asof`.

    A risk scores high when it is severe, still open, raised in more than one internal call (persistence),
    and already echoed in customer calls. Recurrence/echo are found by embedding similarity.
    """
    r = risks[risks.start < asof].reset_index(drop=True)
    i = issues[(issues.start < asof) & (issues.call_type != "internal")].reset_index(drop=True)
    if r.empty:
        return r
    Er, Ei = embed(r.risk.tolist()), embed(i.statement.tolist())
    S_rr, S_ri = Er @ Er.T, Er @ Ei.T
    rows = []
    for k, x in r.iterrows():
        same = r[(S_rr[k] > link) & (r.meeting_id != x.meeting_id)]
        echo = i[S_ri[k] > link]
        rows.append({
            "internal_calls": 1 + same.meeting_id.nunique(),
            "customer_echoes": echo.meeting_id.nunique(),
            "echo_example": echo.statement.iloc[0] if len(echo) else "",
            "first_raised": min([x.start, *same.start]).date(),
        })
    out = pd.concat([r, pd.DataFrame(rows)], axis=1)
    recent = (asof - out.start).dt.days <= window_days
    out["score"] = (out.customer_impact_if_realised.map(IMPACT)
                    * np.where(out.mitigation_status.isin(OPEN_STATUS), 1.0, 0.3)
                    * out.internal_calls * (1 + out.customer_echoes) * np.where(recent, 1.0, 0.5))
    # one row per underlying risk: keep the highest-scoring phrasing of each near-duplicate group
    out = out.sort_values("score", ascending=False)
    E = embed(out.risk.tolist())
    keep, seen = [], []
    for k in range(len(out)):
        if all(E[k] @ E[j] <= link for j in seen):
            keep.append(k); seen.append(k)
    return out.iloc[keep]


def precursors(issues, seed_statements, before, top=8):
    """Issue statements before `before` most similar to the given root-cause statements."""
    q = embed(seed_statements).mean(axis=0)
    q /= np.linalg.norm(q)
    pre = issues[issues.start < before].copy()
    pre["similarity"] = embed(pre.statement.tolist()) @ q
    return pre.sort_values("similarity", ascending=False).head(top), pre.similarity


# ---- 2. product & competitive intelligence ---------------------------------------------
def competitor_mentions(calls_x):
    rows = [{"meeting_id": r.meeting_id, "start": r.start, "call_type": r.call_type, "account": r.account,
             "title": r.title, "competitor": c["name"], "context": c["context"], "quote": c["evidence_quote"]}
            for r in calls_x.itertuples() if not r.refused for c in r.competitors]
    return pd.DataFrame(rows)


def feature_demand(calls_x, min_cluster_size=5):
    """Cluster feature requests, name clusters with the LLM, and size demand by accounts and churn risk."""
    fr = pd.DataFrame([{"meeting_id": r.meeting_id, "call_type": r.call_type, "account": r.account,
                        "churn_risk": r.churn_risk, "request": f}
                       for r in calls_x.itertuples() if not r.refused for f in r.feature_requests])
    labels, _ = cl.umap_hdbscan(embed(fr.request.tolist()), n_neighbors=10,
                                min_cluster_size=min_cluster_size, min_samples=2)
    fr["cluster"] = labels
    terms = cl.ctfidf_terms(fr.request.tolist(), labels)
    named = topics.name_clusters(fr, terms, kind="feature requests", text_col="request", step="feature_cluster_name")
    fr = fr.merge(named[["cluster", "name"]], on="cluster", how="left")
    fr["at_risk_account"] = fr.account.where(fr.churn_risk.isin(["medium", "high"]))
    demand = (fr[fr.cluster >= 0]
              .groupby("name").agg(requests=("request", "size"),
                                   accounts=("account", "nunique"),
                                   at_risk_accounts=("at_risk_account", "nunique"),
                                   internal_mentions=("call_type", lambda s: (s == "internal").sum()),
                                   example=("request", "first"))
              .sort_values(["at_risk_accounts", "accounts"], ascending=False))
    return fr, demand


# ---- 3. account health -----------------------------------------------------------------
def account_health(call_sent, calls_x, competitors, asof):
    """One row per customer account with the signals a CSM would want on one screen."""
    d = (call_sent.dropna(subset=["account"]).drop(columns=["churn_risk", "resolution"], errors="ignore")
         .merge(calls_x[["meeting_id", "churn_risk", "resolution"]], on="meeting_id")
         .sort_values("start"))
    rows = []
    for acc, g in d.groupby("account"):
        last = g.iloc[-1]
        sup = g[g.call_type == "support"]
        bad = sup[sup.net_customer < 0]
        open_loop = len(bad) > 0 and not (g[g.start > bad.start.max()].call_type == "external").any()
        comps = competitors[competitors.account == acc].competitor.unique() if len(competitors) else []
        rows.append({
            "account": acc, "calls": len(g), "last_call": last.start.date(), "last_type": last.call_type,
            "days_since_last": (asof - last.start).days,
            "first_score": g.sentiment_score.iloc[0], "last_score": last.sentiment_score,
            "trend": last.sentiment_score - g.sentiment_score.iloc[0],
            "last_customer_net": last.net_customer, "churn_risk": last.churn_risk,
            "competitors": ", ".join(sorted(comps)),
            "unresolved_support": int((sup.resolution == "unresolved").sum()),
            "open_loop": open_loop,
        })
    h = pd.DataFrame(rows)
    risk_pts = {"high": 3, "medium": 2, "low": 1, "none": 0, "not_applicable": 0}
    # transparent additive score: each flag is one reason a CSM can read off the table
    h["watch_score"] = (h.churn_risk.map(risk_pts)
                        + (h.last_customer_net < 0).astype(int)
                        + (h.trend <= -1).astype(int)
                        + (h.competitors != "").astype(int)
                        + h.open_loop.astype(int)
                        + (h.unresolved_support > 0).astype(int))
    return h.sort_values(["watch_score", "last_score"], ascending=[False, True])
