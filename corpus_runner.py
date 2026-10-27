# =============================================================================
# corpus_runner.py — Corpus batch benchmark lifecycle
#
# Manages the full lifecycle of a corpus benchmark run (models × questions).
# Called from app.py via render_corpus_section().
#
# Architecture:
#   1. RENDER — _render_corpus_expander()
#      Shows the corpus management expander (question CRUD, model selection,
#      pipeline selection, run/clear buttons) and builds the run queue on
#      button press.
#
#   2. INCREMENTAL EXECUTION — _run_corpus_queue()
#      Processes exactly ONE queued item per Streamlit render cycle, then
#      requests another rerun to advance. It prefers fragment-scoped reruns,
#      but falls back to a normal app rerun when Streamlit rejects fragment
#      scope in the current execution context. This avoids session timeouts
#      during long benchmark runs while providing a live progress bar.
#
#   3. PER-QUESTION TIMEOUT — _execute_with_timeout()
#      Wraps the selected pipelines (SPARQL + SQL + Genie, as chosen) in a
#      single daemon thread with a wall-clock cap
#      (model_config["per_question_timeout_s"], default 120 s).
#      If the thread exceeds the budget, all results are marked as timeout.
#
# Data flow:
#   render_corpus_section(ontology_graph, ontology_schema_context, store)
#     → _render_corpus_expander()          [UI: expander + queue builder]
#     → _run_corpus_queue(..., store=store) [one item per render cycle]
#       → _execute_with_timeout(...)        [daemon thread + timeout]
#         → run_ontology_pipeline(store=store)  [Oxigraph SPARQL] (if selected)
#         → run_sql_pipeline(...)               [Databricks SQL]  (if selected)
#         → run_genie_space(...)                [Genie]           (if selected)
#       → build_run_record(...)             [flat metrics dict]
#       → st.session_state["benchmark_runs"].append(record)
#
# The Oxigraph store is passed through the entire call chain to ensure
# fast SPARQL execution (10–100× vs rdflib) during batch benchmarks.
#
# Pipeline selection:
#   Each queue item carries a "pipelines" field (set[str]) frozen at queue-build
#   time so mid-run checkbox changes don't affect in-flight items.
#   Skipped pipelines return a minimal dict with {"skipped": True} so that
#   build_run_record produces zero-values for all their metrics.
# =============================================================================

import os
import threading
import uuid

import streamlit as st

from config import TABLES, GENIE_SPACES, get_model_config
from corpus import get_corpus, get_selected_models, get_selected_pipelines, render_corpus_manager
from ontology_panel import run_ontology_pipeline
from sql_panel import run_sql_pipeline, get_sql_schema_context
from genie_client import run_genie_space
from benchmark_metrics import build_run_record

# Default per-question wall-clock cap (seconds); can be overridden per model config.
_PER_QUESTION_TIMEOUT_S = 120

_ALL_SQL_TABLES = list(TABLES.values())

# Sentinel dict returned for pipelines that are skipped (not selected).
# build_run_record uses .get() with defaults so all metrics become 0.
_SKIPPED_RESULT = {"skipped": True, "total_elapsed": 0}


def _request_rerun() -> None:
    """Request the next rerun, preferring fragment scope when allowed.

    Streamlit raises StreamlitAPIException if scope="fragment" is requested
    during a full-app rerun. That can happen when the queue is advanced from a
    fragment during an app-triggered rerun (for example, directly after a button
    click). In that case we safely fall back to a normal rerun.
    """
    try:
        st.rerun(scope="fragment")
    except st.errors.StreamlitAPIException:
        st.rerun()


# =============================================================================
# Public entry point
# =============================================================================

def render_corpus_section(ontology_graph, ontology_schema_context, store=None, ont_rag=None) -> None:
    """Render the full corpus section.

    Delegates to the management expander (question CRUD, run/clear buttons) and
    the incremental executor that processes one queued item per render cycle.

    Args:
        ontology_graph:          Always None (legacy — Oxigraph Store handles all queries).
        ontology_schema_context: Structured schema-context string for LLM prompts.
        store:                   pyoxigraph.Store for fast SPARQL execution.
    """
    _render_corpus_expander()
    _run_corpus_queue(ontology_graph, ontology_schema_context, store=store, ont_rag=ont_rag)


# =============================================================================
# Internal helpers
# =============================================================================

def _render_corpus_expander() -> None:
    """Render the corpus management expander with run/clear buttons.

    Contains:
      - render_corpus_manager() for question CRUD + model selection + pipeline selection
      - Run button that builds the full queue (models × questions)
        Disabled when no questions, no models, OR no pipelines are selected.
      - Clear button that resets accumulated results
      - Queue builder that triggers a safe rerun helper
    """
    with st.expander("📋 Fragen-Corpus (Parameterstudie)", expanded=False):
        render_corpus_manager()
        st.divider()

        corpus            = get_corpus()
        selected_models   = get_selected_models()
        selected_pipelines = get_selected_pipelines()

        col_run, col_reps, col_clear = st.columns([3, 1, 1])
        with col_reps:
            num_repetitions = st.number_input(
                "🔁 Wdh.",
                min_value=1,
                max_value=10,
                value=st.session_state.get("benchmark_repetitions", 1),
                step=1,
                help="Wiederholungen pro Frage für gemittelte Werte.",
                key="_corpus_repetitions_input",
            )
            st.session_state["benchmark_repetitions"] = num_repetitions

        total_runs = len(corpus) * len(selected_models) * num_repetitions
        rep_label = f" × {num_repetitions} Wdh." if num_repetitions > 1 else ""

        # Compose a short pipeline summary for the button label
        _pipeline_short = {
            "ONT": "ONT",
            "SQL": "SQL",
        }
        for gk in GENIE_SPACES:
            # Abbreviate Genie keys: "Genie: MRS pure DPs" → "Genie(pure)"
            short = gk.replace("Genie Space: MRS ", "Genie(").replace("Genie Space: ", "Genie(")
            if short.startswith("Genie(") and not short.endswith(")"):
                short += ")"
            _pipeline_short[gk] = short

        pipeline_label = (
            " · " + " + ".join(_pipeline_short.get(p, p) for p in sorted(selected_pipelines))
            if selected_pipelines else " · ⚠️ keine Pipeline"
        )

        with col_run:
            run_corpus = st.button(
                f"🔬 Corpus ausführen ({total_runs} Läufe = "
                f"{len(corpus)} Fragen × "
                f"{len(selected_models)} Modell{'e' if len(selected_models) != 1 else ''}"
                f"{rep_label}{pipeline_label})",
                type="primary",
                disabled=(
                    len(corpus) == 0
                    or len(selected_models) == 0
                    or len(selected_pipelines) == 0
                ),
                use_container_width=True,
                key="run_corpus_btn",
            )
        with col_clear:
            if st.button("🗑 Ergebnisse löschen", key="clear_results_btn"):
                st.session_state["benchmark_runs"] = []
                st.session_state.pop("_corpus_queue", None)
                _request_rerun()

        # Queue builder: freeze selected_pipelines into each item at queue-build time
        if run_corpus:
            frozen_pipelines = list(selected_pipelines)  # snapshot at press time
            queue = [
                {
                    "model":     model_name,
                    "question":  q,
                    "repetition": rep,
                    "pipelines": frozen_pipelines,
                }
                for rep in range(1, num_repetitions + 1)
                for model_name in get_selected_models()
                for q in get_corpus()
            ]
            st.session_state["_corpus_queue"] = queue
            _request_rerun()


def _run_corpus_queue(ontology_graph, ontology_schema_context, store=None, ont_rag=None) -> None:
    """Execute one queued item per render cycle (incremental execution).

    Processes the first item in st.session_state["_corpus_queue"], appends the
    result to st.session_state["benchmark_runs"], pops the queue, and triggers
    st.rerun(scope="fragment") for the next iteration.

    Each queue item contains a 'repetition' field (1-based) so the aggregator
    can group and average results across repetitions.

    Args:
        ontology_graph:          Always None (legacy — Oxigraph Store handles all queries).
        ontology_schema_context: Structured schema-context string for LLM prompts.
        store:                   pyoxigraph.Store for fast SPARQL execution.
    """
    queue = st.session_state.get("_corpus_queue", [])
    if not queue:
        return

    corpus_len      = len(get_corpus())
    models_len      = len(get_selected_models())
    num_repetitions = st.session_state.get("benchmark_repetitions", 1)
    total_planned   = corpus_len * models_len * num_repetitions
    done_so_far     = total_planned - len(queue)

    next_item          = queue[0]
    model_name         = next_item["model"]
    q                  = next_item["question"]
    repetition         = next_item.get("repetition", 1)
    item_pipelines     = set(next_item.get("pipelines", ["ONT", "SQL"] + list(GENIE_SPACES.keys())))

    rep_label = f" (Wdh. {repetition}/{num_repetitions})" if num_repetitions > 1 else ""
    progress_pct = done_so_far / total_planned if total_planned > 0 else 0
    st.progress(
        progress_pct,
        text=(
            f"🔬 [{model_name}] {done_so_far + 1}/{total_planned}{rep_label} — "
            f"{q['id']}: {q['question']}"
        ),
    )

    ont_result, sql_result, genie_results = _execute_with_timeout(
        q, model_name, ontology_graph, ontology_schema_context,
        store=store, ont_rag=ont_rag,
        selected_pipelines=item_pipelines,
    )

    record = build_run_record(
        run_id=uuid.uuid4().hex[:8],
        question_id=q["id"],
        question_text=q["question"],
        ont_result=ont_result,
        sql_result=sql_result,
        schema_context_ont=ontology_schema_context,
        schema_context_sql=get_sql_schema_context(),
        model_name=model_name,
        repetition=repetition,
        genie_results=genie_results,
        is_fixture=q.get("is_fixture", False),  # fixture → Q-number category; user → auto-classify
    )

    accumulated = st.session_state.get("benchmark_runs", [])
    accumulated.append(record)
    st.session_state["benchmark_runs"] = accumulated

    queue.pop(0)
    if queue:
        st.session_state["_corpus_queue"] = queue
    else:
        st.session_state.pop("_corpus_queue", None)
    _request_rerun()


def _execute_with_timeout(
    q: dict,
    model_name: str,
    ontology_graph,
    ontology_schema_context,
    store=None,
    ont_rag=None,
    selected_pipelines: set[str] | None = None,
) -> tuple[dict, dict, dict]:
    """Run selected pipelines in a background thread with a wall-clock timeout cap.

    Returns (ont_result, sql_result, genie_results).
    Pipelines not in selected_pipelines are skipped (return _SKIPPED_RESULT).
    On timeout, active pipeline dicts carry error keys.
    genie_results is a dict[label, result_dict] for each configured Genie Space
    that was selected.
    """
    if selected_pipelines is None:
        selected_pipelines = set(["ONT", "SQL"] + list(GENIE_SPACES.keys()))

    wh_id            = os.environ.get("DATABRICKS_WAREHOUSE_ID", st.session_state.get("_sql_wh_id", ""))
    model_cfg        = get_model_config(model_name)
    question_timeout = model_cfg.get("per_question_timeout_s", _PER_QUESTION_TIMEOUT_S)

    ont_holder: list   = [None]
    sql_holder: list   = [None]
    genie_holder: list = [{}]

    run_ont   = "ONT" in selected_pipelines
    run_sql   = "SQL" in selected_pipelines
    run_genie = {label for label in GENIE_SPACES if label in selected_pipelines}

    def _run_selected() -> None:
        if run_ont:
            try:
                ont_holder[0] = run_ontology_pipeline(
                    q["question"], ontology_graph, ontology_schema_context,
                    store=store, model_config=model_cfg, ont_rag=ont_rag,
                )
            except Exception as exc:
                ont_holder[0] = {"question": q["question"], "error": str(exc), "total_elapsed": 0}

        if run_sql:
            try:
                sql_holder[0] = run_sql_pipeline(
                    q["question"], _ALL_SQL_TABLES, wh_id,
                    model_config=model_cfg,
                )
            except Exception as exc:
                sql_holder[0] = {"question": q["question"], "error": str(exc), "total_elapsed": 0}

        # Genie Spaces — only those selected
        genie_res: dict[str, dict] = {}
        for label, genie_cfg in GENIE_SPACES.items():
            if label not in run_genie:
                continue
            try:
                genie_res[label] = run_genie_space(
                    question=q["question"],
                    space_id=genie_cfg["space_id"],
                    timeout_s=genie_cfg.get("timeout_s", 120),
                )
            except Exception as exc:
                genie_res[label] = {
                    "question": q["question"],
                    "total_elapsed": 0,
                    "status": "ERROR",
                    "error": str(exc),
                    "generated_sql": None,
                    "description": None,
                    "row_count": None,
                }
        genie_holder[0] = genie_res

    worker = threading.Thread(target=_run_selected, daemon=True)
    worker.start()
    worker.join(timeout=question_timeout)

    if worker.is_alive():
        timeout_msg = (
            f"Per-Question-Timeout von {question_timeout}s überschritten. "
            "Alle Pipelines wurden abgebrochen."
        )
        ont_result   = ont_holder[0]  or (
            {"question": q["question"], "error": timeout_msg, "total_elapsed": question_timeout}
            if run_ont else {**_SKIPPED_RESULT, "question": q["question"]}
        )
        sql_result   = sql_holder[0]  or (
            {"question": q["question"], "error": timeout_msg, "total_elapsed": question_timeout}
            if run_sql else {**_SKIPPED_RESULT, "question": q["question"]}
        )
        genie_results = genie_holder[0] or {}
        for label in run_genie:
            if label not in genie_results:
                genie_results[label] = {
                    "question": q["question"], "total_elapsed": question_timeout,
                    "status": "TIMEOUT", "error": timeout_msg,
                    "generated_sql": None, "description": None, "row_count": None,
                }
    else:
        ont_result = ont_holder[0] or (
            {"question": q["question"], "error": "Unbekannter Fehler", "total_elapsed": 0}
            if run_ont else {**_SKIPPED_RESULT, "question": q["question"]}
        )
        sql_result = sql_holder[0] or (
            {"question": q["question"], "error": "Unbekannter Fehler", "total_elapsed": 0}
            if run_sql else {**_SKIPPED_RESULT, "question": q["question"]}
        )
        genie_results = genie_holder[0] or {}

    return ont_result, sql_result, genie_results
