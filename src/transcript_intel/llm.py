"""Thin Claude client: structured output, refusal fallback, on-disk cache, cost tracking.

Every response is cached under cache/llm/<step>/ keyed by a hash of (model, prompt, schema),
so notebooks re-run offline and for free once the pipeline has run once.
"""
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache" / "llm"
MODEL = "claude-opus-5-5"
PRICE_PER_MTOK = {"input": 4.00, "output": 20.00}

_client = None
_lock = threading.Lock()
usage_totals = {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0, "refusals": 0}


def _get_client():
    global _client
    if _client is None:
        import anthropic
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        _client = anthropic.Anthropic(max_retries=4)
    return _client


def _key(system, prompt, schema, effort):
    blob = json.dumps([MODEL, system, prompt, schema.model_json_schema(), effort], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def call(step, item_id, system, prompt, schema, effort="low", max_tokens=8000, use_cache=True):
    """Return a validated `schema` instance (or None if Claude declined), using the cache.

    use_cache=False bypasses the cache in both directions: it neither reads nor writes it.
    """
    path = CACHE_DIR / step / f"{item_id}-{_key(system, prompt, schema, effort)}.json"
    if use_cache and path.exists():
        with _lock:
            usage_totals["cached"] += 1
        data = json.loads(path.read_text())
        return None if data.get("refused") else schema.model_validate(data["output"])

    resp = _get_client().beta.messages.parse(
        model=MODEL,
        max_tokens=max_tokens,
        output_config={"effort": effort},
        # Security-vendor transcripts talk about threats and attacks; if a safety classifier
        # declines, the server re-runs the request on a fallback model instead of failing.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,
    )
    with _lock:
        usage_totals["calls"] += 1
        usage_totals["input_tokens"] += resp.usage.input_tokens
        usage_totals["output_tokens"] += resp.usage.output_tokens
    if not use_cache:   # live/demo calls never overwrite the cached run the notebooks depend on
        return None if resp.stop_reason == "refusal" else resp.parsed_output
    path.parent.mkdir(parents=True, exist_ok=True)
    if resp.stop_reason == "refusal":
        with _lock:
            usage_totals["refusals"] += 1
        path.write_text(json.dumps({"refused": True, "request_id": resp._request_id}))
        return None
    out = resp.parsed_output
    path.write_text(json.dumps({"model": resp.model, "request_id": resp._request_id,
                                "output": out.model_dump()}, indent=1))
    return out


def run_many(step, items, build, schema, effort="low", workers=8):
    """Run `call` over {item_id: payload}; `build(payload)` returns (system, prompt)."""
    results = {}
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(call, step, iid, *build(p), schema, effort): iid for iid, p in items.items()}
        for f in as_completed(futs):
            results[futs[f]] = f.result()
    return results


def cost_so_far():
    u = usage_totals
    return (u["input_tokens"] * PRICE_PER_MTOK["input"] + u["output_tokens"] * PRICE_PER_MTOK["output"]) / 1e6


def has_api_credentials():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ---- grounding check -------------------------------------------------------------------
def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def quote_grounded(quote, transcript, min_words=4):
    """True if the quote (punctuation/case-insensitive) appears verbatim in the transcript.

    Quotes that stitch two fragments with '...' count if every fragment is present.
    """
    if not quote:
        return False
    t = _norm(transcript)
    frags = [_norm(f) for f in re.split(r"\.\.\.|…", quote)]
    frags = [f for f in frags if len(f.split()) >= min_words] or [_norm(quote)]
    return all(f in t for f in frags)
