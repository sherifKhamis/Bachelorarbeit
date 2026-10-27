# =============================================================================
# app.py — Ontology vs Database AI Benchmark · Main Entry Point
#
# Streamlit application root.  Pure orchestrator — delegates all rendering
# to specialised modules:
#
#   1. ONTOLOGY BOOTSTRAP (ontology_loader.py)
#      Downloads and parses the RDF/OWL TTL file from a Unity Catalog Volume
#      into rdflib.Graph (schema introspection) + pyoxigraph.Store (fast SPARQL
#      execution).  Cached in session_state; gc.freeze() prevents GC stalls.
#
#   2. APP HEADER & KEEPALIVE (app_header.py)
#      Professional header card with ontology stats + hidden iframe for Azure
#      Application Gateway WebSocket keepalive.
#
#   3. MAIN CONTENT FRAGMENT (@st.fragment)
#      All interactive content lives inside a single fragment so that
#      st.rerun(scope="fragment") during benchmark execution does NOT
#      re-trigger the ontology load or reset page-level session state.
#      Delegates to:
#        - corpus_runner.render_corpus_section()  — batch benchmark
#        - single_benchmark.render_single_benchmark() — one-shot benchmark
#        - panel_layout.render_panel_layout()     — side-by-side panels
#        - benchmark_dashboard.render_benchmark_results() — results
#
# Threading model:
#   Both pipelines (SPARQL and SQL) create a FRESH WorkspaceClient() inside
#   each daemon thread.  Thread death closes the connection.  This prevents
#   connection-pool exhaustion across 55+ rapid-fire benchmark questions.
#
# Session state keys (page-level, persist across fragment reruns):
#   - ontology_graph:          None (legacy — Oxigraph Store handles all queries)
#   - ontology_schema_context: str (LLM system prompt context)
#   - ontology_store:          pyoxigraph.Store (fast SPARQL execution)
#   - ont_collapsed / sql_collapsed: bool (panel visibility)
#   - benchmark_runs:          list[dict] (accumulated corpus results)
#   - _corpus_queue:           list[dict] (remaining items to execute)
#   - benchmark_last_run:      dict (most recent single-question result)
# =============================================================================

import gc
import sys

import streamlit as st

from ontology_loader import load_ontology_and_schema
from app_header import render_app_header, render_azure_keepalive
from corpus_runner import render_corpus_section
from single_benchmark import render_single_benchmark
from panel_layout import render_panel_layout
from benchmark_dashboard import render_benchmark_results
from graph_rag import (
    build_ontology_index,
    OntologyGraphRAG,
)


# =============================================================================
# Page configuration & session state initialisation
# =============================================================================

st.set_page_config(page_title="Ontology vs Database LLM Benchmark", layout="wide")

for key, default in [
    ("ont_collapsed",  False),
    ("sql_collapsed",  False),
    ("benchmark_runs", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# =============================================================================
# Ontology bootstrap (one-time load, persisted in session state)
# =============================================================================

if "ontology_graph" not in st.session_state:
    _graph, _schema, _store, _stats, _prefix_map = load_ontology_and_schema()
    st.session_state["ontology_graph"]          = _graph
    st.session_state["ontology_schema_context"] = _schema
    st.session_state["ontology_store"]          = _store  # Oxigraph (fast SPARQL)
    st.session_state["ontology_stats"]          = _stats
    st.session_state["ontology_prefix_map"]     = _prefix_map
    gc.collect()
    gc.freeze()

# =============================================================================
# Graph RAG index bootstrapping (once per session, after ontology load)
#
# Both blocks use try/except so that an unavailable embedding endpoint (e.g.
# NotFound, network error) does not crash the app.  On failure the session key
# is set to None and all downstream callers fall back to the legacy context.
#
# The "empty index" guard forces a rebuild when ont_rag/sql_rag exists in
# session_state but the embedded index is empty — this happens after a
# hot-reload (file save) without a fresh browser session: Streamlit reloads
# the module code but keeps the old session_state, so the pre-fix broken
# OntologyGraphRAG instance (with empty index) would otherwise persist
# indefinitely.  Checking len(index) == 0 triggers a transparent rebuild.
# =============================================================================

_ont_rag_needs_init = (
    "ont_rag" not in st.session_state
    or (
        st.session_state.get("ont_rag") is not None
        and len(st.session_state["ont_rag"].index) == 0
    )
)
if _ont_rag_needs_init:
    if st.session_state.get("ont_rag") is not None:
        print("[app] ont_rag index was empty — forcing rebuild.", file=sys.stderr)
    try:
        _pm    = st.session_state["ontology_prefix_map"]
        _store = st.session_state["ontology_store"]
        with st.spinner("Graph RAG: Ontologie-Embeddings werden erstellt..."):
            _ont_idx = build_ontology_index(_store, _pm)
            st.session_state["ont_rag"] = OntologyGraphRAG(_store, _ont_idx, _pm)
    except Exception as _exc:
        st.warning(
            f"⚠️ Graph RAG (Ontologie) konnte nicht initialisiert werden "
            f"({type(_exc).__name__}) — Legacy-Fallback aktiv."
        )
        st.session_state["ont_rag"] = None

ontology_graph          = st.session_state["ontology_graph"]
ontology_schema_context = st.session_state["ontology_schema_context"]
ontology_store          = st.session_state["ontology_store"]
ontology_stats          = st.session_state.get("ontology_stats", {})
ont_rag                 = st.session_state.get("ont_rag")


# =============================================================================
# App header & Azure keepalive
# =============================================================================

render_app_header(ontology_stats)
render_azure_keepalive()


# =============================================================================
# Main content fragment
#
# All interactive content lives inside a single @st.fragment so that
# st.rerun(scope="fragment") during benchmark execution does NOT re-trigger
# the ontology load or reset page-level session state.
# =============================================================================

@st.fragment
def _main_content_fragment() -> None:
    """Top-level fragment composing all interactive sections."""

    # ── 1. Corpus batch benchmark (expander + incremental executor) ─────────
    render_corpus_section(
        ontology_graph, ontology_schema_context,
        store=ontology_store, ont_rag=ont_rag,
    )

    # ── 2. Accumulated results dashboard ───────────────────────────────────
    runs = st.session_state.get("benchmark_runs", [])
    if runs:
        st.subheader(f"📊 Benchmark-Ergebnisse ({len(runs)} Läufe)")
        render_benchmark_results(runs)

    # ── 3. Single-question benchmark ───────────────────────────────────────
    ont_q = st.session_state.get("ontology_input",     "").strip()
    sql_q = st.session_state.get("sql_question_input", "").strip()
    render_single_benchmark(
        ont_q, sql_q, ontology_graph, ontology_schema_context,
        store=ontology_store, ont_rag=ont_rag,
    )

    # ── 4. Two-column panel layout (Ontology | SQL) ────────────────────────
    render_panel_layout(
        ontology_graph, ontology_schema_context,
        store=ontology_store, ont_rag=ont_rag,
    )


# =============================================================================
# App entry point
# =============================================================================
_main_content_fragment()
