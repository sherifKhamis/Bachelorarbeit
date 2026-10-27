# =============================================================================
# single_benchmark.py — Single-question benchmark UI and execution
#
# Provides the „Einzelfrage-Benchmark“ feature: run one question through
# selected pipelines (Ontologie, SQL, Genie Spaces) sequentially with a
# user-selected model, then display a timing comparison card.  Called from
# app.py via render_single_benchmark().
#
# Flow:
#   1. User enters questions in one or more panels
#   2. User selects a model and clicks the run button
#   3. Prompt injection guard checks all non-empty inputs
#   4. Only pipelines with input are executed (phases numbered dynamically)
#   5. Results stored in session_state and fragment reruns to display card
#
# Unlike the corpus benchmark (which processes one question per render
# cycle), the single benchmark runs all active phases synchronously within
# one render cycle — acceptable for a single question.
#
# Exported function:
#   • render_single_benchmark(ont_q, sql_q, ontology_graph,
#                             ontology_schema_context, store=None)
# =============================================================================

import os
import time
import uuid

import streamlit as st

from config import TABLES, MODELS, GENIE_SPACES, get_model_config
from ontology_panel import run_ontology_pipeline
from sql_panel import run_sql_pipeline
from genie_client import run_genie_space
from benchmark_summary import render_benchmark_summary
from prompt_guard import sanitize_user_input

_ALL_SQL_TABLES = list(TABLES.values())


def render_single_benchmark(
    ont_q: str,
    sql_q: str,
    ontology_graph,
    ontology_schema_context,
    store=None,
    ont_rag=None,
) -> None:
    """Render the single-question benchmark block.

    Shows a model selector, repetition count, and run button (disabled until
    at least one panel has input).  Executes only those pipelines that have
    input N times sequentially with spinners, aggregates results, and displays
    the benchmark summary card.

    Args:
        ont_q:                   Current question from the ontology panel input.
        sql_q:                   Current question from the SQL panel input.
        ontology_graph:          Always None (legacy — Oxigraph Store handles all queries).
        ontology_schema_context: LLM context string injected into the system prompt.
        store:                   pyoxigraph.Store for fast SPARQL execution.
    """
    # ── Collect Genie Space inputs from session state ───────────────────────
    genie_questions: dict[str, str] = {}
    for label, cfg in GENIE_SPACES.items():
        space_id = cfg["space_id"]
        genie_questions[label] = st.session_state.get(f"genie_input_{space_id}", "").strip()

    # ── Determine which pipelines are active (have input) ──────────────────
    active_pipelines: list[str] = []
    if ont_q:
        active_pipelines.append("Ontologie")
    if sql_q:
        active_pipelines.append("SQL")
    for label, q in genie_questions.items():
        if q:
            active_pipelines.append(label)

    any_ready = len(active_pipelines) >= 1

    col_model, col_reps = st.columns([3, 1])
    with col_model:
        selected_bm_model = st.selectbox(
            "Modell für Einzelfrage-Benchmark",
            options=list(MODELS.keys()),
            format_func=lambda m: f"{m}  ·  {MODELS[m].get('description', '')}",
            index=0,
            key="single_bm_model",
            help="Modell für den Einzelfrage-Benchmark.",
        )
    with col_reps:
        num_reps = st.number_input(
            "🔁 Wdh.",
            min_value=1,
            max_value=10,
            value=st.session_state.get("single_bm_repetitions", 1),
            step=1,
            help="Wiederholungen für gemittelte Latenz/Tokens.",
            key="_single_bm_reps_input",
        )
        st.session_state["single_bm_repetitions"] = num_reps

    rep_label = f" × {num_reps} Wdh." if num_reps > 1 else ""
    pipelines_label = ", ".join(active_pipelines) if active_pipelines else "—"
    run_benchmark = st.button(
        f"🔬 Einzelfrage-Benchmark mit {selected_bm_model}{rep_label}",
        type="primary",
        disabled=not any_ready,
        use_container_width=True,
        help=(
            "Mindestens 1 Panel muss befüllt sein."
            if not any_ready else
            f"Aktive Pipelines ({len(active_pipelines)}): {pipelines_label}.{rep_label}"
        ),
        key="run_benchmark_btn",
    )

    if run_benchmark:
        _execute_single_benchmark(
            ont_q, sql_q, selected_bm_model,
            ontology_graph, ontology_schema_context,
            store=store, num_repetitions=num_reps,
            ont_rag=ont_rag,
            genie_questions=genie_questions,
        )

    if "benchmark_last_run" in st.session_state:
        render_benchmark_summary(st.session_state["benchmark_last_run"])


# =============================================================================
# Internal helpers
# =============================================================================

def _execute_single_benchmark(
    ont_q: str,
    sql_q: str,
    model_name: str,
    ontology_graph,
    ontology_schema_context,
    store=None,
    num_repetitions: int = 1,
    ont_rag=None,
    genie_questions: dict[str, str] | None = None,
) -> None:
    """Run active pipelines sequentially N times, aggregate, store, then rerun.

    Only pipelines whose input is non-empty are executed.  Skipped pipelines
    store None in the run dict and are excluded from aggregation/display.

    Args:
        ont_q:                   Ontology panel question (empty = skip).
        sql_q:                   SQL panel question (empty = skip).
        model_name:              Key into ``config.MODELS``.
        ontology_graph:          Always None (legacy — Oxigraph Store handles all queries).
        ontology_schema_context: LLM context string injected into the system prompt.
        store:                   pyoxigraph.Store for fast SPARQL execution.
        num_repetitions:         Number of times to run each pipeline (1–10).
        genie_questions:         {label: question} for each Genie Space (empty str = skip).
    """
    if genie_questions is None:
        genie_questions = {}

    # ── Prompt Injection Guard: check all non-empty inputs ──────────────────
    inputs_to_check: list[str] = []
    if ont_q:
        inputs_to_check.append(ont_q)
    if sql_q:
        inputs_to_check.append(sql_q)
    for gq in genie_questions.values():
        if gq:
            inputs_to_check.append(gq)

    for text in inputs_to_check:
        _, blocked, reason = sanitize_user_input(text)
        if blocked:
            st.error(
                f"🛡️ **Prompt Injection erkannt** — Anfrage blockiert.\n\n"
                f"Grund: {reason}"
            )
            return

    wh_id     = os.environ.get("DATABRICKS_WAREHOUSE_ID", st.session_state.get("_sql_wh_id", ""))
    model_cfg = get_model_config(model_name)

    # ── Determine active pipelines ─────────────────────────────────────────
    run_ont = bool(ont_q)
    run_sql = bool(sql_q)
    active_genie: dict[str, str] = {
        label: q for label, q in genie_questions.items() if q
    }
    _total_phases = int(run_ont) + int(run_sql) + len(active_genie)

    # ── Run active pipelines N times ────────────────────────────────────────
    all_runs: list[dict] = []

    for rep in range(1, num_repetitions + 1):
        rep_tag = f" (Wdh. {rep}/{num_repetitions})" if num_repetitions > 1 else ""
        phase_idx = 0

        # ── Ontologie (conditional) ──────────────────────────────────────
        ont_result = None
        if run_ont:
            phase_idx += 1
            with st.spinner(f"🔬 [{model_name}] Phase {phase_idx}/{_total_phases} — Ontologie läuft…{rep_tag}"):
                ont_result = run_ontology_pipeline(
                    ont_q, ontology_graph, ontology_schema_context,
                    store=store, model_config=model_cfg, ont_rag=ont_rag,
                )

        # ── SQL (conditional) ───────────────────────────────────────────
        sql_result = None
        if run_sql:
            phase_idx += 1
            with st.spinner(f"🔬 [{model_name}] Phase {phase_idx}/{_total_phases} — SQL läuft…{rep_tag}"):
                sql_result = run_sql_pipeline(
                    sql_q, _ALL_SQL_TABLES, wh_id, model_config=model_cfg,
                )

        # ── Genie Spaces (only those with input) ──────────────────────────
        genie_results: dict[str, dict] = {}
        for genie_label, genie_q in active_genie.items():
            genie_cfg = GENIE_SPACES[genie_label]
            phase_idx += 1
            with st.spinner(
                f"🔬 Phase {phase_idx}/{_total_phases} — {genie_label} läuft…{rep_tag}"
            ):
                genie_results[genie_label] = run_genie_space(
                    question=genie_q,
                    space_id=genie_cfg["space_id"],
                    timeout_s=genie_cfg.get("timeout_s", 120),
                )

        all_runs.append({
            "ont_result": ont_result,
            "sql_result": sql_result,
            "genie_results": genie_results,
            "repetition": rep,
        })

    # ── Build aggregated summary first (averages over all runs) ─────────────
    _build_aggregated_summary(all_runs, model_name, run_ont=run_ont, run_sql=run_sql)
    agg = st.session_state["benchmark_last_run"]

    # ── Update panel state with averaged timing (not just last run) ──────────
    # Panel timer (render_large_timer) shows „Letzte Ausführung · Gesamtzeit“.
    # With repetitions > 1 we want it to display the average, not the last run.
    last_ont = all_runs[-1]["ont_result"]
    last_sql = all_runs[-1]["sql_result"]

    if last_ont is not None:
        st.session_state["ontology_results"]          = last_ont
        st.session_state["ontology_last_total_time"]  = agg["ont_wall_ms"] / 1000.0
        st.session_state["ontology_last_llm_ms"]      = agg["ont_llm_ms"]
        st.session_state["ontology_last_exec_ms"]     = agg["ont_exec_ms"]
        st.session_state["ontology_last_viz_ms"]      = agg["ont_viz_ms"]
        st.session_state["ontology_last_overhead_ms"] = agg["ont_overhead_ms"]
        st.session_state["ontology_last_rag_ms"]      = agg["ont_rag_ms"]

    if last_sql is not None:
        st.session_state["sql_results"]               = last_sql
        st.session_state["sql_last_total_time"]       = agg["sql_wall_ms"] / 1000.0
        st.session_state["sql_last_llm_ms"]           = agg["sql_llm_ms"]
        st.session_state["sql_last_exec_ms"]          = agg["sql_exec_ms"]
        st.session_state["sql_last_overhead_ms"]      = agg["sql_overhead_ms"]
        st.session_state["sql_last_rag_ms"]           = agg["sql_rag_ms"]

    # ── Update Genie panel result keys (results + averaged timer) ────────────
    last_genie_results = all_runs[-1].get("genie_results", {})
    for genie_label, genie_cfg in GENIE_SPACES.items():
        space_id = genie_cfg["space_id"]
        g_raw = last_genie_results.get(genie_label, {})
        agg_g = agg["genie_results"].get(genie_label, {})
        if g_raw or agg_g:
            st.session_state[f"genie_result_{space_id}"] = {
                **g_raw,
                "total_elapsed":     agg_g.get("wall_ms", 0) / 1000.0,
                "latency_breakdown": agg_g.get("latency_breakdown") or g_raw.get("latency_breakdown"),
                "status":            agg_g.get("status",        g_raw.get("status", "UNKNOWN")),
                "error":             agg_g.get("error",          g_raw.get("error")),
                "generated_sql":     agg_g.get("generated_sql",  g_raw.get("generated_sql")),
                "row_count":         agg_g.get("row_count",      g_raw.get("row_count")),
                "result_preview":    agg_g.get("result_preview", g_raw.get("result_preview")),
            }

    st.rerun(scope="fragment")


def _build_aggregated_summary(
    all_runs: list[dict],
    model_name: str,
    *,
    run_ont: bool = True,
    run_sql: bool = True,
) -> None:
    """Aggregate N run results into a single benchmark_last_run dict.

    Handles partial pipeline runs: when run_ont or run_sql is False, the
    corresponding timing/token fields are set to 0 and the ``pipelines_run``
    dict in the result records which pipelines were actually executed.

    Args:
        all_runs:   List of {"ont_result": ..., "sql_result": ..., "repetition": int}.
        model_name: Key into config.MODELS.
        run_ont:    Whether ontology pipeline was executed.
        run_sql:    Whether SQL pipeline was executed.
    """
    n = len(all_runs)
    run_id = str(uuid.uuid4())[:8]

    # ── Helper: safe dict access for potentially-None results ──────────────
    _EMPTY: dict = {}

    def _ont(run: dict) -> dict:
        return run.get("ont_result") or _EMPTY

    def _sql(run: dict) -> dict:
        return run.get("sql_result") or _EMPTY

    # ── Per-run timing summaries ────────────────────────────────────────
    timing_records: list[dict] = []
    for run in all_runs:
        ont = _ont(run)
        sql = _sql(run)

        ont_wall = ont.get("total_elapsed", 0.0) * 1000.0
        ont_llm  = ont.get("sparql_gen_time", 0.0) * 1000.0
        ont_exec = ont.get("sparql_exec_time", 0.0) * 1000.0
        ont_viz  = ont.get("viz_time", 0.0) * 1000.0
        _ont_rag_meta = (ont.get("token_breakdown") or {}).get("graph_rag_meta", {})
        ont_rag  = _ont_rag_meta.get("embedding_ms", 0.0) + _ont_rag_meta.get("traversal_ms", 0.0)
        ont_oh   = max(0.0, ont_wall - ont_llm - ont_exec - ont_viz - ont_rag)

        sql_wall = sql.get("total_elapsed", 0.0) * 1000.0
        sql_llm  = sql.get("sql_gen_time", 0.0) * 1000.0
        sql_exec = sql.get("sql_exec_time", 0.0) * 1000.0
        _sql_rag_meta = (sql.get("token_breakdown") or {}).get("graph_rag_meta", {})
        sql_rag  = _sql_rag_meta.get("embedding_ms", 0.0) + _sql_rag_meta.get("retrieval_ms", 0.0)
        sql_oh   = max(0.0, sql_wall - sql_llm - sql_exec - sql_rag)

        timing_records.append({
            "ont_wall_ms": ont_wall, "ont_llm_ms": ont_llm,
            "ont_exec_ms": ont_exec, "ont_viz_ms": ont_viz,
            "ont_overhead_ms": ont_oh,  "ont_rag_ms": round(ont_rag, 1),
            "sql_wall_ms": sql_wall, "sql_llm_ms": sql_llm,
            "sql_exec_ms": sql_exec, "sql_overhead_ms": sql_oh, "sql_rag_ms": round(sql_rag, 1),
        })

    # ── Average latency keys ────────────────────────────────────────────
    latency_keys = [
        "ont_wall_ms", "ont_llm_ms", "ont_exec_ms", "ont_viz_ms", "ont_overhead_ms", "ont_rag_ms",
        "sql_wall_ms", "sql_llm_ms", "sql_exec_ms", "sql_overhead_ms", "sql_rag_ms",
    ]
    avg_latency = {
        k: round(sum(r.get(k, 0.0) for r in timing_records) / n, 1)
        for k in latency_keys
    }

    # ── Average token keys ──────────────────────────────────────────────
    token_keys = [
        "total_prompt_tokens", "output_tokens",
        "schema_context_tokens", "instructions_tokens", "user_message_tokens",
    ]

    def _avg_tokens(results: list[dict], breakdown_key: str, prefix: str) -> dict:
        """Average token values from token_breakdown across runs."""
        out = {}
        for tk in token_keys:
            vals = [
                ((r.get(breakdown_key) or {}).get("token_breakdown") or {}).get(tk, 0)
                for r in results
            ]
            out[f"{prefix}_{tk}"] = round(sum(vals) / n) if vals else 0
        return out

    ont_token_avgs = _avg_tokens(all_runs, "ont_result", "ont")
    sql_token_avgs = _avg_tokens(all_runs, "sql_result", "sql")

    # ── Select best result (prefer no-error) for status fields ──────────────
    best_ont = (
        next(
            (r["ont_result"] for r in all_runs if r.get("ont_result") and not r["ont_result"].get("error")),
            all_runs[-1].get("ont_result"),
        )
        if run_ont else None
    )
    best_sql = (
        next(
            (r["sql_result"] for r in all_runs if r.get("sql_result") and not r["sql_result"].get("error")),
            all_runs[-1].get("sql_result"),
        )
        if run_sql else None
    )

    # Use best result's token_breakdown for the full dicts
    ont_tb = (best_ont.get("token_breakdown") or {}) if best_ont else {}
    sql_tb = (best_sql.get("token_breakdown") or {}) if best_sql else {}

    graph_rag_applied   = ont_tb.get("graph_rag_applied", False)
    graph_rag_reduction = ont_tb.get("graph_rag_reduction_pct", 0.0)

    sql_graph_rag_applied   = sql_tb.get("graph_rag_applied", False)
    sql_graph_rag_reduction = sql_tb.get("graph_rag_reduction_pct", 0.0)

    # ── Aggregate Genie Space results (only those that were run) ─────────────
    genie_agg: dict[str, dict] = {}
    for label in GENIE_SPACES.keys():
        # Only aggregate if this space was actually run (present in genie_results)
        space_runs = [r.get("genie_results", {}).get(label) for r in all_runs]
        if not any(space_runs):
            continue  # skip spaces that were not executed

        wall_vals = [
            (sr or {}).get("total_elapsed", 0.0) * 1000.0
            for sr in space_runs
        ]
        avg_wall = round(sum(wall_vals) / n, 1) if wall_vals else 0.0
        best_genie = next(
            (sr for sr in space_runs if sr and not sr.get("error")),
            space_runs[-1] or {},
        )
        # Average latency breakdown phases across repetitions
        _bd_keys = ["init_s", "start_s", "polling_s", "fetch_s"]
        avg_bd: dict[str, float | None] = {}
        for bk in _bd_keys:
            vals = [
                ((sr or {}).get("latency_breakdown") or {}).get(bk)
                for sr in space_runs
            ]
            valid = [v for v in vals if v is not None]
            avg_bd[bk] = round(sum(valid) / len(valid), 4) if valid else None

        genie_agg[label] = {
            "wall_ms":          avg_wall,
            "error":            best_genie.get("error"),
            "status":           best_genie.get("status", "UNKNOWN"),
            "exec_success":     best_genie.get("status") == "COMPLETED" and not best_genie.get("error"),
            "generated_sql":    best_genie.get("generated_sql"),
            "row_count":        best_genie.get("row_count"),
            "result_preview":   best_genie.get("result_preview"),
            "latency_breakdown": avg_bd if any(v is not None for v in avg_bd.values()) else None,
        }

    # ── Store aggregated summary ──────────────────────────────────────────
    st.session_state["benchmark_last_run"] = {
        "run_id":         run_id,
        "model_name":     model_name,

        # ── Which pipelines were active ───────────────────────────────────────
        "pipelines_run": {
            "ont": run_ont,
            "sql": run_sql,
            "genie": list(genie_agg.keys()),
        },

        # ── Timing (averaged) ─────────────────────────────────────────────
        "ont_wall_ms":     avg_latency["ont_wall_ms"],
        "ont_llm_ms":      avg_latency["ont_llm_ms"],
        "ont_exec_ms":     avg_latency["ont_exec_ms"],
        "ont_viz_ms":      avg_latency["ont_viz_ms"],
        "ont_overhead_ms": avg_latency["ont_overhead_ms"],
        "ont_rag_ms":      avg_latency["ont_rag_ms"],
        "ont_error":       best_ont.get("error") if best_ont else None,

        "sql_wall_ms":     avg_latency["sql_wall_ms"],
        "sql_llm_ms":      avg_latency["sql_llm_ms"],
        "sql_exec_ms":     avg_latency["sql_exec_ms"],
        "sql_overhead_ms": avg_latency["sql_overhead_ms"],
        "sql_rag_ms":      avg_latency["sql_rag_ms"],
        "sql_error":       best_sql.get("error") if best_sql else None,

        # ── Genie Spaces (averaged, only those that ran) ───────────────────
        "genie_results": genie_agg,

        # ── Graph-RAG ───────────────────────────────────────────────────
        "graph_rag_applied":           graph_rag_applied,
        "graph_rag_reduction_pct":     graph_rag_reduction,
        "sql_graph_rag_applied":       sql_graph_rag_applied,
        "sql_graph_rag_reduction_pct": sql_graph_rag_reduction,

        # ── Full token_breakdown dicts (for unified render_token_breakdown) ──
        "ont_token_breakdown": ont_tb,
        "sql_token_breakdown": sql_tb,

        # ── Flat token fields (averaged, backward compat with corpus runs) ──
        "ont_input_tokens":  ont_token_avgs.get("ont_total_prompt_tokens", 0),
        "ont_output_tokens": ont_token_avgs.get("ont_output_tokens", 0),
        "ont_schema_tokens": ont_token_avgs.get("ont_schema_context_tokens", 0),
        "ont_instr_tokens":  ont_token_avgs.get("ont_instructions_tokens", 0),
        "ont_user_tokens":   ont_token_avgs.get("ont_user_message_tokens", 0),
        "sql_input_tokens":  sql_token_avgs.get("sql_total_prompt_tokens", 0),
        "sql_output_tokens": sql_token_avgs.get("sql_output_tokens", 0),
        "sql_schema_tokens": sql_token_avgs.get("sql_schema_context_tokens", 0),
        "sql_instr_tokens":  sql_token_avgs.get("sql_instructions_tokens", 0),
        "sql_user_tokens":   sql_token_avgs.get("sql_user_message_tokens", 0),

        # ── Multi-run metadata ────────────────────────────────────────────
        "_num_repetitions": n,
        "_all_runs":        all_runs,
        "_timing_records":  timing_records,
    }
