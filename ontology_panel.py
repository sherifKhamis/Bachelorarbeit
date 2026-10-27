# =============================================================================
# ontology_panel.py — Left panel: SPARQL ontology explorer
#
# Owns the full SPARQL pipeline from natural-language question to displayed
# results.  Used both interactively (render_ontology_panel) and headlessly
# by the benchmark runners (run_ontology_pipeline).
#
# ─── PIPELINE ────────────────────────────────────────────────────────────────
#
#   Phase 1 — LLM Generation    (sparql_generator.generate_sparql)
#              Graph-RAG: keyword → class BFS → schema context filtering
#              Timeout-protected via fresh WorkspaceClient in daemon thread
#
#   Phase 2 — SPARQL Execution  (_run_sparql_with_timeout)
#              Primary:  Oxigraph (Rust-native, 10–100× faster than rdflib)
#              Fallback: rdflib.Graph (when no store is provided)
#              Oxigraph terms converted to rdflib terms for uniform downstream
#
#   Phase 3 — Visualisation     (graph_vis.visualize_construct_graph_interactive)
#              CONSTRUCT results → interactive pyvis HTML graph
#              SELECT results    → pandas DataFrame table
#
# ─── THREADING ───────────────────────────────────────────────────────────────
#
#   _run_sparql_with_timeout():
#     Oxigraph/rdflib query runs in a daemon thread.
#     thread.join(timeout=sparql_timeout_s) enforces the wall-clock budget.
#     Frischer WorkspaceClient pro Thread (see generate_sparql in sparql_generator)
#     verhindert Connection-Pool-Exhaustion über 55+ Benchmark-Runs hinweg.
#
# ─── EXPORTS ─────────────────────────────────────────────────────────────────
#
#   • run_ontology_pipeline(question, graph, schema_context, store, model_config)
#       Headless pipeline — no st.* calls; safe for benchmark runners.
#   • render_ontology_panel(graph, schema_context, store)
#       Streamlit UI (input, SPARQL, graph/table, token breakdown).
#
# =============================================================================

import threading
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rdflib import Graph, URIRef, Literal, BNode

import pyoxigraph as ox

from graph_vis import visualize_construct_graph_interactive
from sparql_generator import (
    generate_sparql,
    is_construct_query,
    repair_sparql_with_feedback,
    strip_prefixes_for_display,
)
from ui_components import render_large_timer, render_token_breakdown
from utils import estimate_token_count, shorten_uri


# ---------------------------------------------------------------------------
# Session state keys
# Prefixed with "ontology_" to avoid collisions with sql_panel keys.
# ---------------------------------------------------------------------------
_SESSION_KEY  = "ontology_last_total_time"   # total wall-clock seconds (float)
_RESULTS_KEY  = "ontology_results"           # last pipeline result dict
_OVERHEAD_KEY = "ontology_last_overhead_ms"  # thread + parse overhead in ms
_LLM_MS_KEY   = "ontology_last_llm_ms"       # LLM generation phase in ms
_EXEC_MS_KEY  = "ontology_last_exec_ms"      # SPARQL execution phase in ms
_VIZ_MS_KEY   = "ontology_last_viz_ms"       # graph visualisation phase in ms
_RAG_MS_KEY   = "ontology_last_rag_ms"       # Graph RAG sub-phase (embedding + traversal)

# Default SPARQL execution timeout used when model_config provides no override.
_SPARQL_TIMEOUT_S = 90


# ---------------------------------------------------------------------------
# Oxigraph → rdflib term conversion
# ---------------------------------------------------------------------------

def _ox_term_to_rdflib(term):
    """Convert a pyoxigraph term to the equivalent rdflib term.

    Handles all four pyoxigraph term types that can appear in query results:

    - ``ox.NamedNode``  → ``rdflib.URIRef``
    - ``ox.BlankNode``  → ``rdflib.BNode``
    - ``ox.Literal``    → ``rdflib.Literal`` (preserves datatype and language)
    - ``None``          → ``None`` (unbound variable in a SELECT result)

    Any unrecognised term type is coerced to a plain string Literal as a
    safe fallback so downstream code never encounters a raw pyoxigraph object.

    Args:
        term: A pyoxigraph term or ``None``.

    Returns:
        The equivalent rdflib term, or ``None`` for unbound variables.
    """
    if term is None:
        return None
    if isinstance(term, ox.NamedNode):
        return URIRef(term.value)
    if isinstance(term, ox.BlankNode):
        return BNode(term.value)
    if isinstance(term, ox.Literal):
        if term.datatype is not None:
            return Literal(term.value, datatype=URIRef(term.datatype.value))
        if term.language is not None:
            return Literal(term.value, lang=term.language)
        return Literal(term.value)
    # Fallback: wrap unknown term types as a plain string Literal.
    return Literal(str(term))


# ---------------------------------------------------------------------------
# SPARQL execution with timeout
# ---------------------------------------------------------------------------

def _run_sparql_with_timeout(
    graph: Graph,
    query: str,
    timeout_s: int,
    store: "ox.Store | None" = None,
):
    """Execute a SPARQL query inside a daemon thread with a wall-clock timeout.

    Selects the execution engine based on whether ``store`` is provided:

    - **Oxigraph** (primary, when ``store`` is not None):
      10–100× faster than rdflib for aggregations and multi-hop traversals.
      CONSTRUCT results are materialised into an rdflib Graph for uniform
      downstream processing.  SELECT results convert each Oxigraph solution
      to a dict of rdflib terms via ``_ox_term_to_rdflib``.

    - **rdflib** (fallback, when ``store`` is None):
      Used in development/testing without a loaded pyoxigraph store.

    Args:
        graph:     rdflib.Graph instance (used only for the rdflib fallback path).
        query:     SPARQL query string (with PREFIX declarations).
        timeout_s: Wall-clock budget in seconds.
        store:     pyoxigraph.Store instance (Oxigraph path) or None (rdflib path).

    Returns:
        A result dict with the following keys:
        - For CONSTRUCT: ``{"type": "construct", "graph": rdflib.Graph,
                            "engine": "Oxigraph" | "rdflib"}``
        - For SELECT:    ``{"type": "select", "vars": list[str],
                            "rows": list[dict], "engine": ...}``

    Raises:
        TimeoutError: If the worker thread is still alive after ``timeout_s``.
        Exception:    Any exception raised inside the worker thread.
    """
    result_holder: list = [None]
    error_holder:  list = [None]

    def _run_oxigraph() -> None:
        """Execute the query via Oxigraph and normalise results to rdflib types."""
        try:
            ox_result = store.query(query)
            # Detect query form: CONSTRUCT results are iterable triples (no .variables).
            if hasattr(ox_result, "__iter__") and not hasattr(ox_result, "variables"):
                # CONSTRUCT → materialise into an rdflib Graph.
                g = Graph()
                for triple in list(ox_result):
                    g.add((
                        _ox_term_to_rdflib(triple.subject),
                        _ox_term_to_rdflib(triple.predicate),
                        _ox_term_to_rdflib(triple.object),
                    ))
                result_holder[0] = {"type": "construct", "graph": g, "engine": "Oxigraph"}
            else:
                # SELECT → convert each solution to a dict of rdflib terms.
                variables = [str(v.value) for v in ox_result.variables]
                rows = []
                for solution in ox_result:
                    row = {}
                    for var in ox_result.variables:
                        val = solution[var]
                        row[str(var.value)] = _ox_term_to_rdflib(val)
                    rows.append(row)
                result_holder[0] = {"type": "select", "vars": variables, "rows": rows, "engine": "Oxigraph"}
        except Exception as exc:
            error_holder[0] = exc

    def _run_rdflib() -> None:
        """Execute the query via rdflib (fallback when no Oxigraph store is available)."""
        try:
            rdf_result = graph.query(query)
            if rdf_result.type == "CONSTRUCT":
                result_holder[0] = {"type": "construct", "graph": rdf_result.graph, "engine": "rdflib"}
            else:
                vars_ = [str(v) for v in rdf_result.vars]
                rows  = [{v: row[v] for v in vars_} for row in rdf_result]
                result_holder[0] = {"type": "select", "vars": vars_, "rows": rows, "engine": "rdflib"}
        except Exception as exc:
            error_holder[0] = exc

    # Choose engine and run inside a daemon thread.
    target = _run_oxigraph if store is not None else _run_rdflib
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)

    if worker.is_alive():
        engine = "Oxigraph" if store is not None else "rdflib"
        raise TimeoutError(
            f"SPARQL-Abfrage überschritt das Zeitlimit von {timeout_s}s ({engine}). "
            "Tipp: Stelle komplexe Fragen in zwei separaten Abfragen."
        )
    if error_holder[0] is not None:
        raise error_holder[0]
    return result_holder[0]


# =============================================================================
# Pipeline orchestration (headless entry point)
# =============================================================================

def run_ontology_pipeline(
    question: str,
    ontology_graph: Graph,
    schema_context: str,
    store: "ox.Store | None" = None,
    model_config: dict | None = None,
    ont_rag=None,
) -> dict:
    """Run the full SPARQL ontology pipeline (LLM generation + execution + viz).

    This is the headless entry point used by the benchmark runners
    (corpus_runner, single_benchmark).  It calls no Streamlit functions and
    is safe to call from background threads.

    Pipeline:
        1. ``generate_sparql``       — LLM → validated SPARQL + token metrics
        2. ``_run_sparql_with_timeout`` — Oxigraph / rdflib → result dict
        3. Materialise results:
           - CONSTRUCT → rdflib Graph + triples_df (Subject/Predicate/Object)
           - SELECT    → results_df
        4. ``visualize_construct_graph_interactive``
           — CONSTRUCT results with > 0 triples → pyvis HTML graph

    Args:
        question:        Natural-language question from the user or corpus.
        ontology_graph:  rdflib.Graph instance (used only for rdflib fallback).
        schema_context:  LLM system prompt context string built by
                         ontology_loader.load_ontology_and_schema().
        store:           pyoxigraph.Store (Oxigraph path, 10–100× faster).
                         Pass None to use rdflib fallback.
        model_config:    Model config dict from ``config.MODELS``.
                         Key used: ``sparql_timeout_s``.

    Returns:
        A flat result dict.  All keys are always present (may be None/0/empty):

        Timing:
          - ``total_elapsed``            (float)  — end-to-end wall-clock seconds
          - ``sparql_gen_time``          (float)  — LLM generation seconds
          - ``sparql_exec_time``         (float)  — SPARQL execution seconds
          - ``viz_time``                 (float)  — graph build seconds (CONSTRUCT only)

        Query:
          - ``question``                 (str)
          - ``generated_query``          (str)
          - ``is_construct``             (bool)
          - ``num_triples``              (int)   — CONSTRUCT only
          - ``triples_df``               (pd.DataFrame) — CONSTRUCT: S/P/O table
          - ``results_df``               (pd.DataFrame) — SELECT results
          - ``graph_html``               (str)   — pyvis HTML (CONSTRUCT + triples > 0)
          - ``sparql_engine``            (str)   — "Oxigraph" | "rdflib"
          - ``error``                    (str | None)

        Token metrics:
          - ``schema_context_size_tokens`` (int)  — schema context tiktoken count
          - ``token_breakdown``            (dict) — full per-component breakdown
          - ``prompt_token_count``         (int)
          - ``query_token_count``          (int)
          - ``prompt_tokens``              (int)  — alias for prompt_token_count
          - ``completion_tokens``          (int)  — alias for query_token_count
    """
    total_start = time.perf_counter()
    result: dict = {"question": question, "error": None, "repair_rounds": 0}

    cfg            = model_config or {}
    sparql_timeout = cfg.get("sparql_timeout_s", _SPARQL_TIMEOUT_S)

    try:
        (
            generated_query,
            sparql_gen_time,
            api_prompt_tokens,
            api_completion_tokens,
            system_prompt_tokens,
            token_breakdown,
        ) = generate_sparql(question, schema_context, model_config=model_config, ont_rag=ont_rag)

        result["generated_query"]            = generated_query
        result["sparql_gen_time"]            = sparql_gen_time
        result["is_construct"]               = is_construct_query(generated_query)
        result["token_breakdown"]            = token_breakdown
        result["prompt_token_count"]         = token_breakdown["total_prompt_tokens"]
        result["query_token_count"]          = token_breakdown["output_tokens"]
        result["schema_context_size_tokens"] = token_breakdown["schema_context_tokens"]
        result["prompt_tokens"]              = token_breakdown["total_prompt_tokens"]
        result["completion_tokens"]          = token_breakdown["output_tokens"]

        # ── Phase 2: SPARQL execution ────────────────────────────────────
        t_exec = time.perf_counter()
        _repair_llm_total = 0.0  # Accumulated repair-LLM seconds (must be subtracted from exec span)
        exec_error: str | None = None
        query_results = None
        try:
            query_results = _run_sparql_with_timeout(
                ontology_graph, generated_query, sparql_timeout, store=store
            )
        except Exception as exec_exc:
            exec_error = str(exec_exc)

        # ── Stage 5: Feedback-driven repair on execution failure ──────────
        # Mirrors sql_panel.py Stage 5: if initial execution fails, feed the
        # Oxigraph error back to the LLM for iterative correction (max 2 rounds).
        if exec_error is not None:
            fixed_sparql, rounds, repair_elapsed, repair_tokens = (
                repair_sparql_with_feedback(
                    question, schema_context,
                    generated_query, exec_error,
                    model_config=model_config, ont_rag=ont_rag,
                )
            )
            result["repair_rounds"] = rounds
            sparql_gen_time += repair_elapsed
            _repair_llm_total += repair_elapsed
            token_breakdown.setdefault("repair_prompt_tokens", 0)
            token_breakdown.setdefault("repair_output_tokens", 0)
            token_breakdown["repair_prompt_tokens"]  += repair_tokens.get("api_prompt", 0)
            token_breakdown["repair_output_tokens"]  += repair_tokens.get("api_completion", 0)
            token_breakdown["api_prompt_tokens"]     += repair_tokens.get("api_prompt", 0)
            token_breakdown["api_completion_tokens"] += repair_tokens.get("api_completion", 0)

            if fixed_sparql is not None:
                generated_query = fixed_sparql
                result["generated_query"] = fixed_sparql
                result["is_construct"]    = is_construct_query(fixed_sparql)
                try:
                    query_results = _run_sparql_with_timeout(
                        ontology_graph, fixed_sparql, sparql_timeout, store=store
                    )
                    exec_error = None
                except Exception as re_exc:
                    exec_error = str(re_exc)

        # ── Stage 5a: Empty-result repair (SELECT with 0 rows) ────────────
        # Mirrors sql_panel.py Stage 5a: a syntactically correct query that
        # returns 0 results is likely semantically wrong. Feed a hint.
        if (
            exec_error is None
            and query_results is not None
            and result.get("repair_rounds", 0) == 0  # don't double-repair
        ):
            is_empty = False
            if query_results["type"] == "select" and len(query_results["rows"]) == 0:
                is_empty = True
            elif query_results["type"] == "construct" and len(query_results["graph"]) == 0:
                is_empty = True

            if is_empty:
                empty_hint = (
                    "The query executed without errors but returned 0 results. "
                    "The triple patterns, FILTER conditions, or URI references "
                    "are likely incorrect for this ontology. Revise the query "
                    "so it returns matching data."
                )
                fixed_sparql, rounds, repair_elapsed, repair_tokens = (
                    repair_sparql_with_feedback(
                        question, schema_context,
                        generated_query, empty_hint,
                        model_config=model_config, ont_rag=ont_rag,
                    )
                )
                result["repair_rounds"] = rounds
                sparql_gen_time += repair_elapsed
                _repair_llm_total += repair_elapsed
                token_breakdown.setdefault("repair_prompt_tokens", 0)
                token_breakdown.setdefault("repair_output_tokens", 0)
                token_breakdown["repair_prompt_tokens"]  += repair_tokens.get("api_prompt", 0)
                token_breakdown["repair_output_tokens"]  += repair_tokens.get("api_completion", 0)
                token_breakdown["api_prompt_tokens"]     += repair_tokens.get("api_prompt", 0)
                token_breakdown["api_completion_tokens"] += repair_tokens.get("api_completion", 0)

                if fixed_sparql is not None:
                    generated_query = fixed_sparql
                    result["generated_query"] = fixed_sparql
                    result["is_construct"]    = is_construct_query(fixed_sparql)
                    try:
                        repaired_results = _run_sparql_with_timeout(
                            ontology_graph, fixed_sparql, sparql_timeout, store=store
                        )
                        # Only accept if it actually returns data.
                        r_has_data = False
                        if repaired_results["type"] == "select" and len(repaired_results["rows"]) > 0:
                            r_has_data = True
                        elif repaired_results["type"] == "construct" and len(repaired_results["graph"]) > 0:
                            r_has_data = True
                        if r_has_data:
                            query_results = repaired_results
                    except Exception:
                        pass  # Keep original empty result rather than error.

        # ── Process final results ──────────────────────────────────────────
        if exec_error is not None:
            result["error"] = exec_error
        elif query_results is not None:
            if query_results["type"] == "construct":
                g = query_results["graph"]
                result["num_triples"] = len(g)
                rows = [
                    {"Subject": shorten_uri(s), "Predicate": shorten_uri(p), "Object": shorten_uri(o)}
                    for s, p, o in g
                ]
                result["triples_df"] = pd.DataFrame(rows)
            else:
                vars_ = query_results["vars"]
                rows  = [
                    {v: shorten_uri(row.get(v)) for v in vars_}
                    for row in query_results["rows"]
                ]
                result["results_df"] = pd.DataFrame(rows, columns=vars_)

            result["sparql_engine"] = query_results.get("engine", "unknown")

        # Subtract repair-LLM seconds: the t_exec→now span includes repair
        # LLM calls whose time is already counted in sparql_gen_time.
        result["sparql_exec_time"] = (time.perf_counter() - t_exec) - _repair_llm_total
        result["sparql_gen_time"]  = sparql_gen_time

        # ── Phase 3: Visualisation (CONSTRUCT only) ──────────────────────
        if result.get("is_construct") and result.get("num_triples", 0) > 0:
            t_viz = time.perf_counter()
            result["graph_html"] = visualize_construct_graph_interactive(query_results["graph"])
            result["viz_time"]   = time.perf_counter() - t_viz

    except Exception as e:
        result["error"] = str(e)

    result["total_elapsed"] = time.perf_counter() - total_start
    # n_llm_calls: 1 (generation) + repair_rounds (0–2 per stage).
    result["n_llm_calls"] = 1 + result.get("repair_rounds", 0)
    return result


# =============================================================================
# Display helpers
# (Streamlit-only — never called from headless benchmark runners)
# =============================================================================

def _display_token_breakdown(res: dict) -> None:
    """Render the unified token breakdown expander for the ontology panel.

    Delegates to ``ui_components.render_token_breakdown`` which provides
    the same two-row layout as the SQL panel:
        Row 1 — Input Tokens (gesamt) | Output Tokens
        Row 2 — Schema Context | Instruktionen | Nutzerfrage
        Footer — additivity check + schema ratio + Graph-RAG badge

    The Graph-RAG badge appears when the ``graph_rag_applied`` key is present
    in the token_breakdown dict (SPARQL-only; absent for SQL).

    Does nothing if the result dict contains no ``token_breakdown`` key.

    Args:
        res: Pipeline result dict as returned by ``run_ontology_pipeline``.
    """
    tb = res.get("token_breakdown")
    if not tb:
        return
    render_token_breakdown(tb)


def _display_ontology_results(res: dict) -> None:
    """Render SPARQL query + results (error, graph, or DataFrame).

    Handles three display modes:
    - Error: shows error message in a code block.
    - CONSTRUCT with triples: interactive pyvis graph + raw triples expander.
    - CONSTRUCT with no triples: warning message.
    - SELECT: DataFrame table.

    Timing details are intentionally omitted here — they are shown in the
    execution-time card rendered by ``render_ontology_panel``.

    Args:
        res: Pipeline result dict as returned by ``run_ontology_pipeline`` or
             read from ``st.session_state[_RESULTS_KEY]``.
    """
    # Show the generated SPARQL query FIRST (even on error/timeout).
    generated_query = res.get("generated_query", "")
    if generated_query:
        display_query = strip_prefixes_for_display(generated_query)
        engine        = res.get("sparql_engine", "")
        engine_badge  = f" · {engine}" if engine else ""
        with st.expander("Generierte SPARQL-Abfrage", expanded=True):
            if engine_badge:
                st.caption(f"Engine{engine_badge}")
            st.code(display_query, language="sparql")

    _display_token_breakdown(res)

    if res.get("error"):
        st.error("Fehler bei der Verarbeitung:")
        st.code(res["error"])
        return

    if res.get("is_construct"):
        if res.get("num_triples", 0) > 0:
            st.success(f"{res['num_triples']} Tripel gefunden · Interaktiver Graph wird angezeigt.")
            graph_html = res.get("graph_html")
            if graph_html:
                components.html(graph_html, height=770, scrolling=False)
            # Raw triples table (collapsed by default — graph is primary view).
            triples_df = res.get("triples_df")
            if triples_df is not None:
                with st.expander("Rohe Tripel (Tabellenansicht)", expanded=False):
                    st.dataframe(triples_df, use_container_width=True)
        else:
            st.warning("CONSTRUCT lieferte keine Tripel. Versuche eine allgemeinere Frage.")
    else:
        df = res.get("results_df")
        if df is not None and not df.empty:
            st.success(f"{len(df)} Ergebnisse gefunden.")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Abfrage lieferte keine Ergebnisse. Versuche eine allgemeinere Frage.")


# =============================================================================
# Panel entry point
# =============================================================================

def render_ontology_panel(
    ontology_graph: Graph,
    ontology_schema_context: str,
    store: "ox.Store | None" = None,
    ont_rag=None,
) -> None:
    """Render the ontology panel as a Streamlit fragment child.

    This is the single Streamlit-facing entry point for the ontology panel.
    It must be called from within a ``@st.fragment`` context.

    UI structure:
        1. Description text
        2. Execution-time card (if a previous result exists in session state)
        3. Question text input
        4. "AI fragen" button — triggers pipeline, stores result, reruns fragment
        5. Result display (SPARQL query + graph/table + token breakdown)

    Args:
        ontology_graph:          rdflib.Graph (used only as rdflib fallback).
        ontology_schema_context: LLM system prompt context string.
        store:                   pyoxigraph.Store (primary execution engine).
    """
    st.markdown(
        "_Stelle Fragen in natürlicher Sprache. "
        "CONSTRUCT → interaktiver Graph · SELECT → Tabelle._"
    )

    # Show the execution-time card from the previous run (ontology panel uses
    # the default blue theme from ui_components — no extra args needed).
    if _SESSION_KEY in st.session_state:
        render_large_timer(
            st.session_state[_SESSION_KEY],
            overhead_ms=st.session_state.get(_OVERHEAD_KEY, 0.0),
            llm_ms=st.session_state.get(_LLM_MS_KEY,   0.0),
            exec_ms=st.session_state.get(_EXEC_MS_KEY,  0.0),
            viz_ms=st.session_state.get(_VIZ_MS_KEY,   0.0),
            rag_ms=st.session_state.get(_RAG_MS_KEY,   0.0),
            # Default accent/bg colours are the ontology blue theme — no overrides needed.
        )

    user_input = st.text_input(
        "Was möchtest du aus der Ontologie wissen?",
        placeholder="z.B. Alle Checkpoints und ihre Verbindungen zu Hallen anzeigen",
        key="ontology_input",
    )

    if st.button("AI fragen", type="primary", key="ontology_ask_btn", use_container_width=True):
        if not user_input.strip():
            st.warning("Bitte eine Frage eingeben.")
        else:
            with st.spinner("Ontologie wird analysiert..."):
                res = run_ontology_pipeline(
                    user_input, ontology_graph, ontology_schema_context,
                    store=store, ont_rag=ont_rag,
                )
            st.session_state[_RESULTS_KEY] = res
            st.session_state[_SESSION_KEY] = res["total_elapsed"]
            _llm_ms  = res.get("sparql_gen_time",  0.0) * 1000.0
            _exec_ms = res.get("sparql_exec_time", 0.0) * 1000.0
            _viz_ms  = res.get("viz_time",         0.0) * 1000.0
            st.session_state[_LLM_MS_KEY]   = _llm_ms
            st.session_state[_EXEC_MS_KEY]  = _exec_ms
            st.session_state[_VIZ_MS_KEY]   = _viz_ms
            # Overhead = total − (LLM + SPARQL exec + viz), floored at 0.
            st.session_state[_OVERHEAD_KEY] = max(
                0.0, res["total_elapsed"] * 1000.0 - _llm_ms - _exec_ms - _viz_ms
            )
            # Graph RAG sub-phase: Carina embedding + Oxigraph traversal
            _rag_meta = (res.get("token_breakdown") or {}).get("graph_rag_meta", {})
            st.session_state[_RAG_MS_KEY] = round(
                _rag_meta.get("embedding_ms", 0.0) + _rag_meta.get("traversal_ms", 0.0), 1
            )
            st.rerun(scope="fragment")

    if _RESULTS_KEY in st.session_state:
        _display_ontology_results(st.session_state[_RESULTS_KEY])
