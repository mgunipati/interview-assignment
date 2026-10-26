"""Pydantic schemas for every LLM step. Claude's output is constrained to these shapes."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

CallType = Literal["support", "external", "internal"]
Product = Literal["Detect", "Protect", "Identity", "Comply", "Platform/Billing", "None"]
Severity = Literal["low", "medium", "high"]
Side = Literal["customer", "aegis"]


# ---- Step: call-type second opinion -------------------------------------------------
class CallTypeCheck(BaseModel):
    call_type: CallType = Field(description="support = customer reporting a problem to support; "
                                            "external = account manager / CSM with a customer; internal = Aegis staff only")
    confidence: float = Field(ge=0, le=1)
    incident_related: bool = Field(description="True if the call is substantially about an outage or incident")
    rationale: str = Field(description="One sentence")


# ---- Step: per-call structured extraction --------------------------------------------
class Issue(BaseModel):
    statement: str = Field(description="One generic sentence describing a single issue/theme, "
                                       "no company or person names, e.g. 'SIEM connector times out under load'")
    product: Product
    severity: Severity
    raised_by: Side
    evidence_quote: str = Field(description="Verbatim span copied from the transcript (5-40 words)")


class Competitor(BaseModel):
    name: str
    context: str = Field(description="Why it came up, one sentence")
    evidence_quote: str


class CallExtraction(BaseModel):
    one_line_summary: str
    primary_purpose: str = Field(description="Why the call happened, 3-8 words")
    issues: List[Issue] = Field(description="3-6 distinct issues or themes discussed, most important first")
    feature_requests: List[str] = Field(description="Concrete capabilities the customer or team asked for; empty if none")
    competitors: List[Competitor] = Field(description="Competitors mentioned; empty if none")
    churn_risk: Literal["none", "low", "medium", "high", "not_applicable"] = Field(
        description="Risk this customer leaves or downgrades; not_applicable for internal calls")
    churn_evidence: Optional[str] = Field(default=None, description="Verbatim quote supporting churn_risk, if any")
    resolution: Literal["resolved", "partially_resolved", "unresolved", "not_applicable"] = Field(
        description="For support/escalation calls: was the customer's problem fixed by the end?")


# ---- Step: name an issue cluster ------------------------------------------------------
class ClusterName(BaseModel):
    name: str = Field(description="2-5 word theme name, specific (not 'Various issues')")
    description: str = Field(description="One sentence: what belongs in this theme")
    suggested_parent: str = Field(description="Broader level-1 group this belongs under, 1-3 words")
    coherence: Literal["high", "medium", "low"] = Field(description="How well the statements fit one theme")


# ---- Step: classify a call into the curated taxonomy ----------------------------------
def make_classification_schema(l2_names):
    """Build a schema whose labels are restricted to the curated taxonomy."""
    L2 = Literal[tuple(l2_names)]

    class Label(BaseModel):
        theme: L2
        confidence: float = Field(ge=0, le=1)
        evidence_quote: str = Field(description="Verbatim span from the transcript (5-40 words)")

    class CallClassification(BaseModel):
        primary_theme: L2 = Field(description="The single theme that best describes the call")
        labels: List[Label] = Field(description="All themes substantially discussed (1-4), including the primary")

    return CallClassification


# ---- Step: operational-risk triage (internal calls) -----------------------------------
class OperationalRisk(BaseModel):
    risk: str = Field(description="One generic sentence: what could fail and how, e.g. 'single ingestion node has no failover'")
    product: Product
    customer_impact_if_realised: Literal["low", "medium", "high", "critical"] = Field(
        description="How bad it would be for customers if this risk materialised")
    mitigation_status: Literal["none", "discussed_only", "scheduled", "in_progress", "deferred", "done"] = Field(
        description="State of the fix as of the END of this call")
    evidence_quote: str = Field(description="Verbatim span from the transcript (5-40 words)")


class RiskTriage(BaseModel):
    risks: List[OperationalRisk] = Field(description="Unresolved technical or operational risks that could cause a "
                                                     "customer-visible failure. Exclude incidents that already happened "
                                                     "and are fully fixed. Empty if none.")
