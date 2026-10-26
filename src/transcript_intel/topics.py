"""Hybrid topic pipeline: rules -> LLM extraction -> embedding clusters -> LLM naming ->
human-curated taxonomy -> LLM multi-label classification -> validation.

Each step returns a DataFrame so the notebook can show (and question) every intermediate result.
"""
import json
from pathlib import Path

import pandas as pd

from . import cluster as cl
from . import llm, prompts
from .embed import embed
from .preprocess import render_transcript
from .schemas import CallExtraction, CallTypeCheck, ClusterName, make_classification_schema

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_DIR = ROOT / "taxonomy"


def _transcripts(redacted):
    return {mid: render_transcript(g) for mid, g in redacted.groupby("meeting_id", sort=False)}


# ---- 1. call-type second opinion ------------------------------------------------------
def check_call_types(calls, redacted):
    tx = _transcripts(redacted)
    items = {r.meeting_id: r for r in calls.itertuples()}

    def build(r):
        domains = sorted({e.split("@")[1] for e in r.emails})
        return prompts.call_type_prompt(r.title, domains, tx[r.meeting_id])

    res = llm.run_many("call_type", items, build, CallTypeCheck)
    out = pd.DataFrame([{"meeting_id": k, **(v.model_dump() if v else {"call_type": None})} for k, v in res.items()])
    out = calls[["meeting_id", "title", "call_type", "incident_flag"]].merge(
        out.rename(columns={"call_type": "llm_call_type", "incident_related": "llm_incident"}), on="meeting_id")
    out["agrees"] = out.call_type == out.llm_call_type
    return out


# ---- 2. per-call structured extraction ------------------------------------------------
def extract(calls, redacted):
    tx = _transcripts(redacted)
    items = {r.meeting_id: r for r in calls.itertuples()}
    res = llm.run_many("extract", items,
                       lambda r: prompts.extraction_prompt(r.title, r.call_type, tx[r.meeting_id]),
                       CallExtraction, effort="medium")
    records, issues = [], []
    for mid, x in res.items():
        if x is None:
            records.append({"meeting_id": mid, "refused": True})
            continue
        d = x.model_dump()
        records.append({"meeting_id": mid, "refused": False, **{k: v for k, v in d.items() if k != "issues"},
                        "churn_evidence_grounded": llm.quote_grounded(d["churn_evidence"], tx[mid])
                        if d["churn_evidence"] else None})
        for i, iss in enumerate(d["issues"]):
            issues.append({"meeting_id": mid, "rank": i, **iss,
                           "grounded": llm.quote_grounded(iss["evidence_quote"], tx[mid])})
    meta = calls[["meeting_id", "title", "call_type", "start", "account"]]
    return (meta.merge(pd.DataFrame(records), on="meeting_id"),
            meta.merge(pd.DataFrame(issues), on="meeting_id"))


# ---- 3. theme discovery over issue statements -----------------------------------------
def discover_themes(issues, min_cluster_size=6, min_samples=3, n_neighbors=10):
    E = embed(issues.statement.tolist())
    labels, _ = cl.umap_hdbscan(E, n_neighbors=n_neighbors, min_cluster_size=min_cluster_size,
                                min_samples=min_samples)
    issues = issues.assign(cluster=labels)
    terms = cl.ctfidf_terms(issues.statement.tolist(), labels)
    return issues, terms, E


def name_clusters(issues, terms, max_statements=40, kind="issue statements", text_col="statement", step="cluster_name"):
    groups = {int(c): g for c, g in issues[issues.cluster >= 0].groupby("cluster")}

    def build(g):
        st = g[text_col].sample(min(len(g), max_statements), random_state=0).tolist()
        return prompts.cluster_name_prompt(st, terms[int(g.cluster.iloc[0])], kind=kind)

    res = llm.run_many(step, groups, build, ClusterName, effort="medium")
    rows = [{"cluster": c, "size": len(groups[c]), "n_calls": groups[c].meeting_id.nunique(),
             "terms": ", ".join(terms[c][:6]), **res[c].model_dump()} for c in sorted(res) if res[c]]
    return pd.DataFrame(rows)


def draft_taxonomy(named):
    """Group LLM cluster names under their suggested parent -> taxonomy/taxonomy_draft.json.

    This is a *draft*: a human edits it into taxonomy/taxonomy.json (merge, rename, split,
    add definitions), which is what classification uses.
    """
    draft = {}
    for r in named.sort_values("size", ascending=False).itertuples():
        draft.setdefault(r.suggested_parent, []).append(
            {"name": r.name, "definition": r.description, "source_clusters": [int(r.cluster)],
             "coherence": r.coherence})
    TAXONOMY_DIR.mkdir(exist_ok=True)
    (TAXONOMY_DIR / "taxonomy_draft.json").write_text(json.dumps(draft, indent=2))
    return draft


def load_taxonomy(path=TAXONOMY_DIR / "taxonomy.json"):
    tax = json.loads(Path(path).read_text())
    rows = [{"l1": l1, "l2": t["name"], "definition": t["definition"]} for l1, ts in tax.items() for t in ts]
    return pd.DataFrame(rows)


def taxonomy_text(tax):
    return "\n".join(f"{l1}\n" + "\n".join(f"  - {r.l2}: {r.definition}" for r in g.itertuples())
                     for l1, g in tax.groupby("l1", sort=False))


# ---- 4. classify every call into the curated taxonomy ---------------------------------
def classify(calls, redacted, tax):
    tx = _transcripts(redacted)
    schema = make_classification_schema(tax.l2.tolist())
    text = taxonomy_text(tax)
    items = {r.meeting_id: r for r in calls.itertuples()}
    res = llm.run_many("classify", items,
                       lambda r: prompts.classification_prompt(text, r.title, r.call_type, tx[r.meeting_id]),
                       schema, effort="medium")
    l1_of = dict(zip(tax.l2, tax.l1))
    rows = []
    for mid, x in res.items():
        if x is None:
            continue
        for lab in x.labels:
            rows.append({"meeting_id": mid, "theme": lab.theme, "l1": l1_of[lab.theme],
                         "is_primary": lab.theme == x.primary_theme, "confidence": lab.confidence,
                         "evidence_quote": lab.evidence_quote,
                         "grounded": llm.quote_grounded(lab.evidence_quote, tx[mid])})
    return calls[["meeting_id", "title", "call_type", "start", "account"]].merge(pd.DataFrame(rows), on="meeting_id")


# ---- 5. validation against human labels -----------------------------------------------
def evaluate(labels_df, assigned):
    """Primary-theme accuracy and per-theme precision/recall vs the human label sheet."""
    pred_primary = assigned[assigned.is_primary].set_index("meeting_id").theme
    pred_sets = assigned.groupby("meeting_id").theme.agg(set)
    rows = []
    for r in labels_df.itertuples():
        human = {r.primary_theme, *r.other_themes} - {""}
        pred = pred_sets.get(r.meeting_id, set())
        rows.append({"meeting_id": r.meeting_id, "title": r.title, "human_primary": r.primary_theme,
                     "model_primary": pred_primary.get(r.meeting_id), "human": human, "model": pred,
                     "primary_match": pred_primary.get(r.meeting_id) == r.primary_theme,
                     "jaccard": len(human & pred) / len(human | pred) if human | pred else 1.0})
    per_call = pd.DataFrame(rows)
    themes = sorted(set().union(*per_call.human, *per_call.model))
    per_theme = pd.DataFrame([{
        "theme": t,
        "support": sum(t in h for h in per_call.human),
        "precision": _ratio(sum(t in h and t in m for h, m in zip(per_call.human, per_call.model)),
                            sum(t in m for m in per_call.model)),
        "recall": _ratio(sum(t in h and t in m for h, m in zip(per_call.human, per_call.model)),
                         sum(t in h for h in per_call.human)),
    } for t in themes])
    return per_call, per_theme


def _ratio(a, b):
    return a / b if b else float("nan")
