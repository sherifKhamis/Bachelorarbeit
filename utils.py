# =============================================================================
# utils.py — Shared utility functions (side-effect-free)
#
# Lightweight helpers imported by multiple modules throughout the application.
# Design principle: NO Streamlit imports, NO side effects, NO I/O.  Safe to
# import from daemon threads, background tasks, and unit tests.
#
# Exported functions:
#   • shorten_uri(item, graph=None) → str
#       Converts an rdflib term (URIRef, Literal, BNode) or plain URI string
#       into a short, human-readable label for DataFrames and graph nodes.
#
#   • estimate_token_count(text) → int
#       Estimates token count using tiktoken (o200k_base) when available,
#       otherwise falls back to a character-based heuristic (len/4).
#       Used by sparql_generator, ontology_panel, sql_panel, and
#       benchmark_metrics for token budget tracking.
#
#   • fmt_ms(ms) → str
#       Formats a millisecond value as a human-readable duration string
#       (e.g. "1.23s" or "456ms").  Used by ui_components.render_large_timer
#       and any module that needs to display phase durations inline.
#
# Performance notes:
#   • _shorten_uri_str is LRU-cached (4096 slots): in CONSTRUCT results the
#     same URIs repeat hundreds of times per query; the cache eliminates
#     redundant '#'/'/' splits and str() conversions entirely after the first hit.
#   • _get_tiktoken_encoder is cached so the encoder object is built only once
#     per process instead of on every estimate_token_count() call.
# =============================================================================

from functools import lru_cache

from rdflib import Graph, URIRef, Literal, BNode


# ---------------------------------------------------------------------------
# URI shortening
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _shorten_uri_str(uri_str: str) -> str:
    """Cached URI-string → short label conversion.

    Called by shorten_uri() after normalising the input to a plain str so that
    lru_cache can key on a hashable, immutable value.  The cache size of 4096
    comfortably covers all distinct URIs in the MRS production ontology.
    """
    if '#' in uri_str:
        return uri_str.split('#')[-1]
    if '/' in uri_str:
        last = uri_str.rsplit('/', 1)[-1]
        return last if last else uri_str
    return uri_str


def shorten_uri(item, graph: Graph = None) -> str:
    """Convert an rdflib term (or plain URI string) to a short human-readable label.

    The function handles all four cases that appear when iterating an rdflib Graph:

    - ``URIRef``  → the local name after ``#`` or the last ``/`` segment
                    e.g. ``<http://example.org/onto#Checkpoint>`` → ``"Checkpoint"``
    - ``Literal`` → the raw string value
                    e.g. ``Literal("42.5")`` → ``"42.5"``
    - ``BNode``   → ``"_:<blank_id>"`` (keeps blank nodes distinguishable)
    - ``str``     → treated as a URI string (same logic as URIRef)
                    This covers callers that already called ``str(node)`` before
                    passing the value in — both paths produce the same result.

    Args:
        item:  An rdflib term (URIRef, Literal, BNode) or a plain Python string.
        graph: Unused; kept for backwards-compatibility with older call sites.

    Returns:
        A short, human-readable string label.
    """
    # Literals hold their value directly — return it as-is.
    if isinstance(item, Literal):
        return str(item)

    # Blank nodes have no URI; prefix with "_:" to make them identifiable.
    if isinstance(item, BNode):
        return f"_:{item}"

    # Both URIRef objects and plain strings go through the cached helper.
    return _shorten_uri_str(str(item))


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _get_tiktoken_encoder():
    """Return a cached tiktoken encoder for gpt-4o (o200k_base).

    Building the encoder is expensive (~50 ms on first call); caching it means
    subsequent estimate_token_count() calls pay only the encode() cost.
    Returns None if tiktoken is not installed so the caller can fall back.
    """
    try:
        import tiktoken
        return tiktoken.encoding_for_model("gpt-4o")
    except (ImportError, Exception):
        return None


def estimate_token_count(text: str) -> int:
    """Estimate the token count for a text string.

    Uses tiktoken with the o200k_base encoding (GPT-4o compatible) when the
    library is available.  Falls back to a character-based heuristic (~4 chars
    per token) for environments where tiktoken is not installed.

    Args:
        text: The input string to estimate tokens for.

    Returns:
        Estimated number of tokens (always >= 1).
    """
    enc = _get_tiktoken_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Duration formatting
# ---------------------------------------------------------------------------

def fmt_ms(ms: float) -> str:
    """Format a millisecond value as a human-readable duration string.

    Displays values >= 1000 ms as seconds with two decimal places, and
    smaller values as whole milliseconds.  Used by
    ``ui_components.render_large_timer`` and any module that needs to
    display per-phase execution times.

    Args:
        ms: Duration in milliseconds.

    Returns:
        Human-readable string, e.g. ``"1.23s"`` or ``"456ms"``.

    Examples:
        >>> fmt_ms(1234.5)
        '1.23s'
        >>> fmt_ms(456)
        '456ms'
    """
    return f"{ms / 1000:.2f}s" if ms >= 1000 else f"{ms:.0f}ms"
