# =============================================================================
# benchmark_aggregator.py — Multi-Run Aggregationslogik
#
# Aggregiert mehrere Wiederholungen desselben Benchmarks zu einer
# zusammengefassten Ansicht. Gruppierung nach (question_id, model_name).
#
# Aggregationsregeln:
#   • Latenz:         Durchschnitt über alle Läufe pro Frage
#   • Tokenverbrauch: Durchschnitt über alle Läufe
#   • Status:         Bestes Ergebnis (kein Fehler > mit Fehler)
#   • Qualität:       Bestes Ergebnis; wenn mehrere nicht-leere → alle sammeln
#
# Exportierte Funktionen:
#   • aggregate_runs(runs: list[dict]) → list[dict]
#     Gibt eine aggregierte Ergebnisliste zurück (ein Eintrag pro Frage×Modell)
#   • get_all_results_for_question(runs, question_id, model_name) → list[dict]
#     Gibt alle Einzel-Ergebnisse für eine bestimmte Frage zurück
#
# =============================================================================

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd


# =============================================================================
# Public API
# =============================================================================

def aggregate_runs(runs: list[dict]) -> list[dict]:
    """Aggregate multiple repetition runs into a single record per (question_id, model_name).

    Grouping key: (question_id, model_name).

    Aggregation rules:
      - Latency fields (*_ms):       arithmetic mean across repetitions
      - Token fields (*_token*):     arithmetic mean across repetitions
      - Status/quality fields:       "best" result selected (see _select_best)
      - All individual results:      stored in '_all_runs' for expandable display

    Args:
        runs: List of flat run-record dicts (from build_run_record + repetition field).

    Returns:
        List of aggregated dicts, one per unique (question_id, model_name) pair.
        Each dict has:
          - All averaged numeric fields
          - Best-result quality/status fields
          - '_all_runs': list of all individual records for this group
          - '_num_repetitions': count of repetitions in this group
          - '_best_ont_results': list of non-empty ONT result previews
          - '_best_sql_results': list of non-empty SQL result previews
    """
    if not runs:
        return []

    # Group by (question_id, model_name)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        key = (r.get("question_id", ""), r.get("model_name", ""))
        groups[key].append(r)

    aggregated: list[dict] = []
    for (q_id, model), group_runs in groups.items():
        agg = _aggregate_group(group_runs)
        aggregated.append(agg)

    # Sort by question_id for consistent display
    aggregated.sort(key=lambda r: (r.get("question_id", ""), r.get("model_name", "")))
    return aggregated


def get_all_results_for_question(
    runs: list[dict],
    question_id: str,
    model_name: str,
) -> list[dict]:
    """Return all individual run records for a specific (question_id, model_name) pair.

    Args:
        runs:        Full list of benchmark run records (all repetitions).
        question_id: Question ID to filter on.
        model_name:  Model name to filter on.

    Returns:
        List of matching run records, sorted by repetition number.
    """
    return sorted(
        [r for r in runs
         if r.get("question_id") == question_id
         and r.get("model_name") == model_name],
        key=lambda r: r.get("repetition", 1),
    )


def is_multi_run(runs: list[dict]) -> bool:
    """Check whether the runs list contains multiple repetitions.

    Returns True if any (question_id, model_name) pair has more than one record.
    """
    seen: set[tuple[str, str]] = set()
    for r in runs:
        key = (r.get("question_id", ""), r.get("model_name", ""))
        if key in seen:
            return True
        seen.add(key)
    return False


# =============================================================================
# Internal aggregation logic
# =============================================================================

# Fields to average (latency)
_LATENCY_FIELDS = [
    "ont_llm_ms", "ont_exec_ms", "ont_viz_ms", "ont_total_ms", "ont_overhead_ms",
    "ont_rag_ms",   # Graph RAG sub-phase: Carina embedding + Oxigraph traversal
    "sql_llm_ms", "sql_exec_ms", "sql_total_ms", "sql_overhead_ms",
    "sql_rag_ms",   # Graph RAG sub-phase: Carina embedding + live SQL retrieval
]

# Fields to average (tokens)
_TOKEN_FIELDS = [
    "ont_prompt_token_count", "ont_query_token_count",
    "ont_schema_context_size_tokens", "ont_instructions_tokens", "ont_user_message_tokens",
    "sql_prompt_token_count", "sql_query_token_count",
    "sql_schema_context_size_tokens", "sql_instructions_tokens", "sql_user_message_tokens",
    # Legacy keys
    "ont_prompt_tokens", "ont_query_tokens", "sql_prompt_tokens", "sql_query_tokens",
    "schema_ont_approx_tokens", "schema_sql_approx_tokens",
]

# Fields to take from the "best" run (non-numeric, quality-related)
_BEST_FIELDS = [
    "ont_error_type", "sql_error_type", "ont_error_msg", "sql_error_msg",
    "ont_syntax_valid", "sql_syntax_valid", "ont_exec_success", "sql_exec_success",
    "ont_row_count", "sql_row_count",
    "ont_generated_query", "sql_generated_query",
    "ont_result_preview", "sql_result_preview",
    "ont_first_attempt_success", "sql_first_attempt_success",
    "human_evaluation", "ont_human_score", "sql_human_score",
]

# Fields to copy from first record (identity, classification — same across runs)
_IDENTITY_FIELDS = [
    "question_id", "question", "model_name",
    "question_type", "join_depth", "triple_path_depth",
    "ont_graph_rag_applied", "ont_graph_rag_reduction_pct",
    "sql_graph_rag_applied", "sql_graph_rag_reduction_pct",
]


def _aggregate_group(group_runs: list[dict]) -> dict:
    """Aggregate a list of runs for the same (question_id, model_name)."""
    n = len(group_runs)
    result: dict[str, Any] = {}

    # ── Identity: copy from first record ─────────────────────────────────────
    first = group_runs[0]
    for f in _IDENTITY_FIELDS:
        result[f] = first.get(f)

    # Use the run_id of the best run (selected below)
    best = _select_best(group_runs)
    result["run_id"] = best.get("run_id", first.get("run_id"))

    # ── Latency: arithmetic mean ─────────────────────────────────────────────
    for f in _LATENCY_FIELDS:
        values = [r.get(f, 0) or 0 for r in group_runs]
        result[f] = round(sum(values) / n, 1) if n > 0 else 0

    # ── Tokens: arithmetic mean ──────────────────────────────────────────────
    for f in _TOKEN_FIELDS:
        values = [r.get(f, 0) or 0 for r in group_runs]
        result[f] = round(sum(values) / n) if n > 0 else 0

    # ── Status/Quality: best run ─────────────────────────────────────────────
    for f in _BEST_FIELDS:
        result[f] = best.get(f)

    # ── Multi-run metadata ───────────────────────────────────────────────────
    result["_num_repetitions"] = n
    result["_all_runs"] = group_runs

    # Collect all non-empty result previews for expandable display
    result["_best_ont_results"] = _collect_non_empty_results(group_runs, panel="ont")
    result["_best_sql_results"] = _collect_non_empty_results(group_runs, panel="sql")

    # ── Genie Space results: average wall_ms, pick best status ───────────────
    all_genie = [r.get("genie_results") for r in group_runs if isinstance(r.get("genie_results"), dict)]
    if all_genie:
        genie_agg: dict[str, Any] = {}
        all_labels: set[str] = set()
        for gr in all_genie:
            all_labels.update(gr.keys())
        for label in sorted(all_labels):
            wall_values = [gr[label].get("wall_ms", 0) for gr in all_genie if label in gr]
            avg_wall = round(sum(wall_values) / len(wall_values), 1) if wall_values else 0
            # Prefer a successful result
            best_g = next(
                (gr[label] for gr in all_genie if label in gr and not gr[label].get("error")),
                all_genie[-1].get(label, {}),
            )
            genie_agg[label] = {
                "wall_ms": avg_wall,
                "status": best_g.get("status", "UNKNOWN"),
                "error": best_g.get("error"),
                "exec_success": best_g.get("exec_success", False),
                "generated_sql": best_g.get("generated_sql"),
                "row_count": best_g.get("row_count"),
                "result_preview": best_g.get("result_preview"),
                "latency_breakdown": best_g.get("latency_breakdown"),
            }
        result["genie_results"] = genie_agg
    else:
        result["genie_results"] = {}

    return result


def _select_best(group_runs: list[dict]) -> dict:
    """Select the 'best' run from a group of repetitions.

    Priority (highest first):
      1. No error AND non-empty result (both panels)
      2. No error AND non-empty result (at least one panel)
      3. No error (even if empty result)
      4. Fewest errors
      5. First run (fallback)
    """
    def _score(r: dict) -> tuple[int, int, int]:
        ont_ok = r.get("ont_exec_success", False) and (r.get("ont_row_count", 0) or 0) > 0
        sql_ok = r.get("sql_exec_success", False) and (r.get("sql_row_count", 0) or 0) > 0
        no_err = r.get("ont_exec_success", False) and r.get("sql_exec_success", False)

        # Higher is better
        both_good = 3 if (ont_ok and sql_ok) else (2 if (ont_ok or sql_ok) else (1 if no_err else 0))
        row_count = (r.get("ont_row_count", 0) or 0) + (r.get("sql_row_count", 0) or 0)
        return (both_good, int(no_err), row_count)

    return max(group_runs, key=_score)


def _collect_non_empty_results(
    group_runs: list[dict],
    panel: str,
) -> list[dict]:
    """Collect all non-empty result previews from a panel across repetitions.

    Args:
        group_runs: All runs for one (question_id, model_name) group.
        panel:      'ont' or 'sql'.

    Returns:
        List of dicts with keys:
          - 'repetition': int
          - 'preview': list[dict] (row data)
          - 'generated_query': str
          - 'row_count': int
          - 'error': str or None
    """
    results: list[dict] = []
    query_key = f"{panel}_generated_query"
    preview_key = f"{panel}_result_preview"
    row_count_key = f"{panel}_row_count"
    error_key = f"{panel}_error_msg"
    success_key = f"{panel}_exec_success"

    for r in group_runs:
        row_count = r.get(row_count_key, 0) or 0
        is_success = r.get(success_key, False)
        if is_success and row_count > 0:
            results.append({
                "repetition": r.get("repetition", 1),
                "preview": r.get(preview_key),
                "generated_query": r.get(query_key, ""),
                "row_count": row_count,
                "error": None,
            })

    return results
