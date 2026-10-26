"""Prompts for each LLM step. Kept stable (no timestamps / ids) so responses cache cleanly."""

CONTEXT = """You are analysing call transcripts from Aegis Cloud Security, a B2B security SaaS company.
Products: Detect (threat detection / SIEM), Protect (backup and recovery, incl. CloudPrime connector),
Identity (SSO, MFA, SCIM, LDAP), Comply (compliance reporting: SOC 2, HIPAA, PCI DSS, ISO 27001).
LogVault is a SIEM integration. Speaker lines are tagged [AEGIS] or [CUSTOMER].
Customer contacts are pseudonymised ("Customer 1") and IDs are masked ("[ID]") - do not try to recover them.
Transcripts may be excerpts of a longer meeting; describe only what is in the text."""


def call_type_prompt(title, attendee_domains, transcript):
    system = CONTEXT + """

Task: decide what kind of call this is, from content (not just the title).
- support: a customer reporting a problem or request to the support team
- external: an account manager / customer success conversation with a customer (renewals, adoption, reviews, feedback)
- internal: Aegis staff only (engineering, product, planning, incident response)"""
    prompt = f"Title: {title}\nAttendee email domains: {', '.join(attendee_domains)}\n\nTranscript:\n{transcript}"
    return system, prompt


def extraction_prompt(title, call_type, transcript):
    system = CONTEXT + """

Task: extract a structured record of this call.
- Issues must be distinct, generic (no company/person names) and each about ONE thing, so they can be
  compared across calls. Prefer the underlying problem over the surface symptom.
- Every evidence quote must be copied exactly from the transcript, without paraphrasing.
- Leave lists empty rather than guessing."""
    prompt = f"Title: {title}\nCall type: {call_type}\n\nTranscript:\n{transcript}"
    return system, prompt


def cluster_name_prompt(statements, terms, kind="issue statements"):
    system = CONTEXT + f"""

Task: these {kind} were grouped by an embedding-based clustering algorithm. Name the theme they share.
If they don't really share one, say so via coherence=low and name the dominant theme."""
    body = "\n".join(f"- {s}" for s in statements)
    prompt = f"Distinctive terms: {', '.join(terms)}\n\nStatements ({len(statements)}):\n{body}"
    return system, prompt


def classification_prompt(taxonomy_text, title, call_type, transcript):
    system = CONTEXT + f"""

Task: label the call with every theme from the taxonomy below that is substantially discussed (1-4 themes),
and pick the single primary theme. Use only theme names from the taxonomy. Confidence reflects how clearly
the transcript supports each label. Evidence quotes must be copied exactly from the transcript.

Taxonomy:
{taxonomy_text}"""
    prompt = f"Title: {title}\nCall type: {call_type}\n\nTranscript:\n{transcript}"
    return system, prompt


def risk_triage_prompt(title, transcript):
    system = CONTEXT + """

Task: this is an internal Aegis call. List the unresolved operational risks raised in it: technical debt, architectural
weaknesses, capacity limits, process gaps - anything that could cause a customer-visible failure if left alone.
Judge only from this call; do not speculate about what happens later. Quotes must be copied exactly."""
    return system, f"Title: {title}\n\nTranscript:\n{transcript}"
