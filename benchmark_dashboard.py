# =============================================================================
# benchmark_dashboard.py — Multi-Run Benchmark-Dashboard
#
# Rendert die Batch-Benchmark-Ergebnisse als 5-Tab Streamlit-Dashboard.
# Alle Charts nutzen Plotly für interaktives Hover, Zoom und Export.
#
# Hinweis: render_benchmark_summary() (Einzellauf-Karte) wurde nach
# benchmark_summary.py extrahiert und ist NICHT mehr in diesem Modul.
#
# Exportierte Funktion:
#   • render_benchmark_results(runs: list[dict])
#     Vollständiges Multi-Run-Dashboard mit 5 Tabs:
#       ⏱ Latenz           — Grouped bar: total ms per question, chunked by type
#       🔢 Aufschlüsselung  — Phase breakdown bars + token KPIs & charts:
#                            • Static KPIs: Schema Context + Instruktionen (constant)
#                            • Chart: Nutzerfrage + Query Tokens (variable per Q)
#                            • Genie Spaces: API-Start · Polling · Overhead
#       ✅ Status           — HTML table with error badges, collapsible error
#                            messages, join_depth/triple_path_depth columns,
#                            and ✓ VERIFIED badge (manual verification)
#       📊 Qualität         — Side-by-side query + result comparison,
#                            collapsible error messages, ✓ Verified checkboxes,
#                            per-pipeline correctness buttons (korrekt/teilweise/falsch)
#       📋 Rohdaten        — Full DataFrame + CSV export (incl. verified status)
#
# Features:
#   • Graph-RAG aware: displays graph_rag_reduction_pct when available
#   • Manual verification: ont_verified/sql_verified persisted to benchmark_runs
#   • Error messages: collapsible <details> in Status and Qualität tabs
#   • Token chart: Schema + Instruktionen shown as KPIs (constant per run),
#     chart only shows variable tokens (Nutzerfrage + Query Tokens)
#
# Display constants:
#   • ONT_COLORS / SQL_COLORS — Model → colour for bar charts
#   • _QT_COLORS / _QT_DISPLAY — QuestionType → colour/label
#   • _ERR_STYLE / _ERR_DISPLAY — Error type → badge colour/label
#   • _CHUNK_SIZE = 10 — Max questions per chart before new chart
#   • _CHART_LAYOUT — Plotly layout defaults (transparent bg, grid style)
#
# Session state keys:
#   • _human_scores — dict of "{qid}__{model}__{pipeline}" → score label (korrekt/teilweise/falsch)
#   • _verified_status — dict of "{qid}__{model}__ont/sql" → bool
#   • benchmark_runs — list[dict] (source of truth, scores/verified persisted here)
#   • _lat_hidden_phases — set[str] of phase keys hidden in the Latency tab
#                          ("ont", "sql", "genie_{label}"); toggled via buttons
#   • _brk_hidden_phases — set[str] of phase keys hidden in the Breakdown tab
#                          ("ont", "sql", "genie_{label}"); toggled via buttons
#   • _brk_hidden_sub_phases — set[str] of sub-phase labels hidden in the Breakdown tab
#                              ("LLM", "Ausführung", "Graph RAG", "Overhead" …); toggled via buttons
# =============================================================================

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_components import ONT_ACCENT_COLOR, SQL_ACCENT_COLOR
from benchmark_aggregator import aggregate_runs, is_multi_run

# ─── Colour & Layout constants ────────────────────────────────────────────────

ONT_COLORS = {"GPT-4o": "#0288d1"}
SQL_COLORS = {"GPT-4o": "#9c27b0"}
GENIE_COLORS = {
    "Genie: MRS pure DPs": "#66bb6a",
    "Genie: MRS DPs + Markdown": "#ffa726",
}

_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=11),
    margin=dict(l=50, r=20, t=30, b=90),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    xaxis=dict(showgrid=False, tickangle=-30),
    yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
    height=420,
)

_QT_COLORS = {
    "unbeantwortbar": "#78909c",
    "faktisch": "#0277bd",
    "relational": "#2e7d32",
    "aggregativ": "#e65100",
    "multi-hop": "#ad1457",
}

_QT_DISPLAY = {
    "unbeantwortbar": "Unbeantwortbar",
    "faktisch": "Faktisch",
    "relational": "Relational",
    "aggregativ": "Aggregativ",
    "multi-hop": "Multi-Hop",
}

# Maximum questions per chart before a new chart is created
_CHUNK_SIZE = 10


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _col(frame: pd.DataFrame, col: str) -> list:
    """Safe column access — returns zeros if column missing."""
    return frame[col].fillna(0).tolist() if col in frame.columns else [0] * len(frame)


def _tok_col(frame: pd.DataFrame, new_name: str, legacy_name: str) -> list:
    """Token column access with new → legacy fallback."""
    if new_name in frame.columns:
        return frame[new_name].fillna(0).tolist()
    if legacy_name in frame.columns:
        return frame[legacy_name].fillna(0).tolist()
    return [0] * len(frame)


def _group_questions(df: pd.DataFrame, df_columns: set) -> list[tuple[str, list[str]]]:
    """Group question IDs for chunked chart rendering.

    If question_type is available, groups by type (in canonical order).
    Otherwise, falls back to chunks of _CHUNK_SIZE questions.

    Returns list of (group_label, [question_ids]).
    """
    questions = df["question_id"].unique().tolist() if "question_id" in df_columns else []
    if not questions:
        return []

    if "question_type" in df_columns:
        # Canonical order
        type_order = ["unbeantwortbar", "faktisch", "relational", "aggregativ", "multi-hop"]
        groups: list[tuple[str, list[str]]] = []
        seen_types: set[str] = set()
        for qt in type_order:
            qids = df[df["question_type"].apply(lambda x: _normalize_enum(str(x))) == qt]["question_id"].unique().tolist()
            if qids:
                groups.append((_QT_DISPLAY.get(qt, qt), qids))
                seen_types.add(qt)
        # Any remaining types not in canonical order
        remaining = df[~df["question_type"].apply(lambda x: _normalize_enum(str(x))).isin(seen_types)]["question_id"].unique().tolist()
        if remaining:
            groups.append(("Sonstige", remaining))
        return groups
    else:
        # Chunk by _CHUNK_SIZE
        chunks: list[tuple[str, list[str]]] = []
        for i in range(0, len(questions), _CHUNK_SIZE):
            chunk = questions[i:i + _CHUNK_SIZE]
            label = f"{chunk[0]}–{chunk[-1]}" if len(chunk) > 1 else chunk[0]
            chunks.append((label, chunk))
        return chunks


# =============================================================================
# Full multi-run dashboard
# =============================================================================

def render_benchmark_results(runs: list[dict]) -> None:
    """Render the full benchmark results dashboard with 6 tabs."""
    if not runs:
        st.info("Noch keine Benchmark-Läufe. Führe den Corpus-Benchmark aus.")
        return

    # ── Multi-run aggregation ─────────────────────────────────────────────────
    _is_aggregated = is_multi_run(runs)
    _raw_run_count = len(runs)
    if _is_aggregated:
        runs = aggregate_runs(runs)
        _num_reps = runs[0].get("_num_repetitions", 1) if runs else 1
    else:
        _num_reps = 1

    df = pd.DataFrame(runs)
    df_columns = set(df.columns.tolist())

    # Deduplizierung: neuester Lauf pro Frage + Modell behalten
    _dedup_cols = [c for c in ["question_id", "model_name"] if c in df_columns]
    _total_raw = len(df)
    if _dedup_cols:
        df = df.drop_duplicates(subset=_dedup_cols, keep="last").reset_index(drop=True)
        df_columns = set(df.columns.tolist())
    _n_dupes = _total_raw - len(df)

    models = sorted(df["model_name"].unique().tolist()) if "model_name" in df_columns else ["—"]
    questions = df["question_id"].unique().tolist() if "question_id" in df_columns else []
    total = len(df)

    # ── Multi-run indicator ───────────────────────────────────────────────────
    if _is_aggregated:
        st.info(
            f"🔁 **{_num_reps} Wiederholungen** — Durchschnittswerte "
            f"({_raw_run_count} Einzelläufe → {total} aggregierte Ergebnisse)"
        )

    # ── KPI tile ──────────────────────────────────────────────────────────────
    st.metric(
        "Läufe gesamt", total,
        delta=f"⚠️ {_n_dupes} Duplikat{'e' if _n_dupes != 1 else ''} entfernt" if _n_dupes > 0 else None,
        delta_color="off",
    )

    # ── Token KPIs (new terminology) ─────────────────────────────────────────
    _tok_new = ["ont_prompt_token_count", "ont_query_token_count", "sql_prompt_token_count", "sql_query_token_count"]
    _tok_legacy = ["ont_prompt_tokens", "ont_query_tokens", "sql_prompt_tokens", "sql_query_tokens"]
    _has_tokens = any(c in df_columns for c in _tok_new + _tok_legacy)

    if _has_tokens:
        tok_tiles = st.columns(len(models) * 2)
        for i, model in enumerate(models):
            mdf_tok = df[df["model_name"] == model] if "model_name" in df_columns else df
            n_runs = max(len(mdf_tok), 1)

            ont_prompt = _tok_col(mdf_tok, "ont_prompt_token_count", "ont_prompt_tokens")
            ont_query  = _tok_col(mdf_tok, "ont_query_token_count",  "ont_query_tokens")
            ont_tok = int(sum(ont_prompt) + sum(ont_query))
            ont_query_total = int(sum(ont_query))

            sql_prompt = _tok_col(mdf_tok, "sql_prompt_token_count", "sql_prompt_tokens")
            sql_query  = _tok_col(mdf_tok, "sql_query_token_count",  "sql_query_tokens")
            sql_tok = int(sum(sql_prompt) + sum(sql_query))
            sql_query_total = int(sum(sql_query))

            tok_tiles[i * 2].metric(
                label=f"🔤 {model} · SPARQL mit Ontologie", value=f"{ont_tok:,} tok",
                delta=f"Ø {ont_tok // n_runs:,} Tokens / Lauf", delta_color="off",
            )
            tok_tiles[i * 2 + 1].metric(
                label=f"🔤 {model} · SQL", value=f"{sql_tok:,} tok",
                delta=f"Ø {sql_tok // n_runs:,} Tokens / Lauf", delta_color="off",
            )

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_lat, tab_stack, tab_status, tab_qual, tab_raw = st.tabs([
        "⏱ Latenz", "🔢 Aufschlüsselung", "✅ Status",
        "📊 Qualität", "📋 Rohdaten",
    ])

    with tab_lat:
        _render_tab_latency(df, df_columns, models)

    with tab_stack:
        _render_tab_breakdown(df, df_columns, models)

    with tab_status:
        _render_tab_status(df, df_columns, models, questions)

    with tab_qual:
        _render_tab_quality(df, df_columns, models, questions)

    with tab_raw:
        _render_tab_raw(df, df_columns)


# =============================================================================
# Tab 1: Latenz (chunked — one chart per question group)
# =============================================================================

def _render_tab_latency(df: pd.DataFrame, df_columns: set, models: list) -> None:
    st.caption("Gesamtzeit pro Frage und Modell. Hover für LLM / Exec/ Overhead-Aufschlüsselung.")

    # ── Phase toggle buttons ──────────────────────────────────────────────────
    # Session state key that stores the set of phase keys currently hidden.
    _LAT_PHASE_KEY = "_lat_hidden_phases"
    if _LAT_PHASE_KEY not in st.session_state:
        st.session_state[_LAT_PHASE_KEY] = set()

    # Detect available Genie spaces from the first non-null row.
    _genie_space_labels: list[str] = []
    if "genie_results" in df_columns:
        for _gv in df["genie_results"].dropna():
            if isinstance(_gv, dict) and _gv:
                _genie_space_labels = sorted(_gv.keys())
                break

    # Build canonical phase list: (session-state key, display label).
    _phases: list[tuple[str, str]] = [
        ("ont", "SPARQL / Ontologie"),
        ("sql", "SQL"),
    ]
    for _gl in _genie_space_labels:
        _phases.append((f"genie_{_gl}", _gl.replace("Genie: ", "Genie Spaces ")))

    # One toggle button per phase.  Active (visible) → primary style; hidden → secondary.
    st.markdown("**Phasen ein-/ausblenden:**")
    _btn_cols = st.columns(len(_phases))
    _hidden: set[str] = st.session_state[_LAT_PHASE_KEY]
    for _i, (_pk, _pl) in enumerate(_phases):
        _is_hidden = _pk in _hidden
        if _btn_cols[_i].button(
            f"{'✗' if _is_hidden else '✓'} {_pl}",
            key=f"_lat_tog_{_pk}",
            type="secondary" if _is_hidden else "primary",
            use_container_width=True,
        ):
            if _is_hidden:
                st.session_state[_LAT_PHASE_KEY].discard(_pk)
            else:
                st.session_state[_LAT_PHASE_KEY].add(_pk)
            st.rerun()

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    groups = _group_questions(df, df_columns)
    if not groups:
        # Fallback: single chart with all data
        groups = [("Alle Fragen", df["question_id"].unique().tolist())]

    for group_label, qids in groups:
        st.markdown(f"**{group_label}**")
        group_df = df[df["question_id"].isin(qids)]

        fig = go.Figure()
        for model in models:
            mdf = group_df[group_df["model_name"] == model]
            if len(mdf) == 0:
                continue
            c_ont = ONT_COLORS.get(model, "#4fc3f7")
            c_sql = SQL_COLORS.get(model, "#ce93d8")

            # ONT bar — skip when phase is hidden
            if "ont" not in _hidden:
                ont_cd = list(zip(
                    _col(mdf, "ont_llm_ms"), _col(mdf, "ont_exec_ms"),
                    _col(mdf, "ont_viz_ms"),  _col(mdf, "ont_rag_ms"),
                    _col(mdf, "ont_overhead_ms"),
                ))
                fig.add_trace(go.Bar(
                    name=f"SPARQL mit Ontologie · {model}", x=mdf["question_id"].tolist(),
                    y=_col(mdf, "ont_total_ms"), marker_color=c_ont, customdata=ont_cd,
                    hovertemplate=(
                        "<b>%{x}</b> · SPARQL mit Ontologie · " + model + "<br>"
                        "Gesamt: <b>%{y:.0f} ms</b><br>"
                        "LLM %{customdata[0]:.0f} ms · Exec %{customdata[1]:.0f} ms · "
                        "Viz %{customdata[2]:.0f} ms · Graph RAG %{customdata[3]:.0f} ms · "
                        "Overhead %{customdata[4]:.0f} ms<extra></extra>"
                    ),
                ))

            # SQL bar — skip when phase is hidden
            if "sql" not in _hidden:
                sql_cd = list(zip(
                    _col(mdf, "sql_llm_ms"), _col(mdf, "sql_exec_ms"),
                    _col(mdf, "sql_rag_ms"), _col(mdf, "sql_overhead_ms"),
                ))
                fig.add_trace(go.Bar(
                    name=f"SQL · {model}", x=mdf["question_id"].tolist(),
                    y=_col(mdf, "sql_total_ms"), marker_color=c_sql, customdata=sql_cd,
                    hovertemplate=(
                        "<b>%{x}</b> · SQL · " + model + "<br>"
                        "Gesamt: <b>%{y:.0f} ms</b><br>"
                        "LLM %{customdata[0]:.0f} ms · Exec %{customdata[1]:.0f} ms · "
                        "Graph RAG %{customdata[2]:.0f} ms · Overhead %{customdata[3]:.0f} ms<extra></extra>"
                    ),
                ))

            # ── Genie Space bars ──────────────────────────────────────────
            if "genie_results" in df_columns:
                _genie_labels = []
                if len(mdf) > 0:
                    first_g = mdf["genie_results"].iloc[0]
                    if isinstance(first_g, dict):
                        _genie_labels = sorted(first_g.keys())
                for g_idx, g_label in enumerate(_genie_labels):
                    if f"genie_{g_label}" in _hidden:
                        continue
                    g_color = list(GENIE_COLORS.values())[g_idx % len(GENIE_COLORS)]
                    g_short = g_label.replace("Genie: ", "Genie Spaces ")
                    genie_rows_lat = mdf["genie_results"].tolist()
                    g_wall_vals = [
                        (row.get(g_label, {}).get("wall_ms", 0) if isinstance(row, dict) else 0)
                        for row in genie_rows_lat
                    ]
                    # Build hover customdata: [API-Start ms, Polling ms, Fetch ms, Overhead ms]
                    g_cd_lat = []
                    for row in genie_rows_lat:
                        if isinstance(row, dict) and g_label in row:
                            bd = row[g_label].get("latency_breakdown") or {}
                            api_ms = bd.get("start_s", 0.0) * 1000.0
                            poll_ms = bd.get("polling_s", 0.0) * 1000.0
                            fetch_ms = bd.get("fetch_s", 0.0) * 1000.0
                            wall = row[g_label].get("wall_ms", 0)
                            oh_ms = max(0.0, wall - api_ms - poll_ms - fetch_ms)
                        else:
                            api_ms, poll_ms, fetch_ms, oh_ms = 0.0, 0.0, 0.0, 0.0
                        g_cd_lat.append([api_ms, poll_ms, fetch_ms, oh_ms])
                    _has_lat_bd = any(cd[0] + cd[1] > 0 for cd in g_cd_lat)
                    fig.add_trace(go.Bar(
                        name=f"{g_short} · {model}",
                        x=mdf["question_id"].tolist(),
                        y=g_wall_vals,
                        marker_color=g_color,
                        customdata=g_cd_lat,
                        hovertemplate=(
                            "<b>%{x}</b> · " + g_short + " · " + model + "<br>"
                            "Gesamt: <b>%{y:.0f} ms</b><br>"
                            + ("API-Start %{customdata[0]:.0f} ms · "
                               "Polling %{customdata[1]:.0f} ms · "
                               "Fetch %{customdata[2]:.0f} ms · "
                               "Overhead %{customdata[3]:.0f} ms"
                               if _has_lat_bd else "") +
                            "<extra></extra>"
                        ),
                    ))

        fig.update_layout(**_CHART_LAYOUT, barmode="group", yaxis_title="ms")
        st.plotly_chart(fig, use_container_width=True, key=f"lat_{group_label}")


# =============================================================================
# Tab 2: Aufschlüsselung (chunked — one chart per question group)
# =============================================================================

def _render_tab_breakdown(df: pd.DataFrame, df_columns: set, models: list) -> None:
    st.caption(
        "Zeitaufschlüsselung pro Phase: LLM-Generierung · Ausführung · Visualisierung · "
        "Graph RAG (Carina-Embedding + Traversal/Retrieval) · Overhead. "
        "Genie Spaces: API-Start · Polling · Fetch · Overhead. "
        "Overhead = Restwert nach Abzug aller benannten Phasen."
    )

    # ── Phase toggle buttons ──────────────────────────────────────────────────
    _BRK_PHASE_KEY = "_brk_hidden_phases"
    if _BRK_PHASE_KEY not in st.session_state:
        st.session_state[_BRK_PHASE_KEY] = set()

    # Detect available Genie spaces from the first non-null row.
    _genie_space_labels_brk_tog: list[str] = []
    if "genie_results" in df_columns:
        for _gv in df["genie_results"].dropna():
            if isinstance(_gv, dict) and _gv:
                _genie_space_labels_brk_tog = sorted(_gv.keys())
                break

    _brk_phases: list[tuple[str, str]] = [
        ("ont", "SPARQL / Ontologie"),
        ("sql", "SQL"),
    ]
    for _gl in _genie_space_labels_brk_tog:
        _brk_phases.append((f"genie_{_gl}", _gl.replace("Genie: ", "Genie Spaces ")))

    st.markdown("**Phasen ein-/ausblenden:**")
    _btn_cols_brk = st.columns(len(_brk_phases))
    _brk_hidden: set[str] = st.session_state[_BRK_PHASE_KEY]
    for _i, (_pk, _pl) in enumerate(_brk_phases):
        _is_hidden = _pk in _brk_hidden
        if _btn_cols_brk[_i].button(
            f"{'✗' if _is_hidden else '✓'} {_pl}",
            key=f"_brk_tog_{_pk}",
            type="secondary" if _is_hidden else "primary",
            use_container_width=True,
        ):
            if _is_hidden:
                st.session_state[_BRK_PHASE_KEY].discard(_pk)
            else:
                st.session_state[_BRK_PHASE_KEY].add(_pk)
            st.rerun()

    st.divider()

    # ── Sub-phase (detail) toggle buttons ────────────────────────────────────────
    _BRK_SUB_KEY = "_brk_hidden_sub_phases"
    if _BRK_SUB_KEY not in st.session_state:
        st.session_state[_BRK_SUB_KEY] = set()

    _sub_phases: list[str] = [
        "LLM", "Ausführung", "Visualisierung", "Graph RAG",
        "Overhead", "API-Start", "Polling", "Fetch",
    ]
    st.markdown("**Detail-Phasen ein-/ausblenden:**")
    _sub_btn_cols = st.columns(len(_sub_phases))
    _brk_sub_hidden: set[str] = st.session_state[_BRK_SUB_KEY]
    for _si, _sp in enumerate(_sub_phases):
        _sp_hidden = _sp in _brk_sub_hidden
        if _sub_btn_cols[_si].button(
            f"{'✗' if _sp_hidden else '✓'} {_sp}",
            key=f"_brk_sub_tog_{_si}",
            type="secondary" if _sp_hidden else "primary",
            use_container_width=True,
        ):
            if _sp_hidden:
                st.session_state[_BRK_SUB_KEY].discard(_sp)
            else:
                st.session_state[_BRK_SUB_KEY].add(_sp)
            st.rerun()

    st.divider()

    groups = _group_questions(df, df_columns)
    if not groups:
        groups = [("Alle Fragen", df["question_id"].unique().tolist())]

    # Pre-compute residual overhead (total overhead minus Graph RAG sub-phase).
    # Capped at 0 so negative values from rounding never appear.
    df = df.copy()
    df["_ont_residual_oh_ms"] = (
        df["ont_overhead_ms"].fillna(0) - df["ont_rag_ms"].fillna(0)
    ).clip(lower=0)
    df["_sql_residual_oh_ms"] = (
        df["sql_overhead_ms"].fillna(0) - df["sql_rag_ms"].fillna(0)
    ).clip(lower=0)

    _PHASES = {
        "ONT": [
            ("ont_llm_ms",          "LLM",            1.0),
            ("ont_exec_ms",         "Ausführung",     0.65),
            ("ont_viz_ms",          "Visualisierung", 0.35),
            ("ont_rag_ms",          "Graph RAG",      0.50),  # Carina embedding + Oxigraph traversal
            ("_ont_residual_oh_ms", "Overhead",       0.15),  # residual after RAG
        ],
        "SQL": [
            ("sql_llm_ms",          "LLM",        1.0),
            ("sql_exec_ms",         "Ausführung", 0.65),
            ("sql_rag_ms",          "Graph RAG",  0.50),  # Carina embedding + live SQL retrieval
            ("_sql_residual_oh_ms", "Overhead",   0.15),  # residual after RAG
        ],
    }

    # Hover tooltip for the Overhead bar segment.
    # %{customdata} holds the raw segment value — %{y} would show the cumulative
    # top position (base + segment), not the segment value alone.
    _OVERHEAD_HOVER = "<b>%{x}</b><br>Overhead: <b>%{customdata:.0f} ms</b><extra></extra>"

    for group_label, qids in groups:
        st.markdown(f"**{group_label}**")
        group_df = df[df["question_id"].isin(qids)]

        fig2 = go.Figure()
        for model in models:
            mdf = group_df[group_df["model_name"] == model]
            if len(mdf) == 0:
                continue
            suffix = f" · {model}" if len(models) > 1 else ""
            x_ont = mdf["question_id"].tolist()
            x_sql = mdf["question_id"].tolist()
            model_idx = models.index(model)

            # ONT phases: stacked (offsetgroup groups them together)
            if "ont" not in _brk_hidden:
                ont_base = [0.0] * len(mdf)
                for col, label, opacity in _PHASES["ONT"]:
                    if label in _brk_sub_hidden:
                        continue
                    y_vals = _col(mdf, col)
                    fig2.add_trace(go.Bar(
                        name=f"SPARQL mit Ontologie {label}{suffix}", x=x_ont, y=y_vals,
                        base=ont_base, offsetgroup=f"ont_{model_idx}",
                        marker_color=ONT_COLORS.get(model, "#4fc3f7"), opacity=opacity,
                        legendgroup=f"SPARQL mit Ontologie_{model}",
                        customdata=y_vals,
                        hovertemplate=_OVERHEAD_HOVER if label == "Overhead" else f"<b>%{{x}}</b><br>{label}: %{{customdata:.0f}} ms<extra></extra>",
                    ))
                    ont_base = [b + v for b, v in zip(ont_base, y_vals)]

            # SQL phases: stacked (separate offsetgroup)
            if "sql" not in _brk_hidden:
                sql_base = [0.0] * len(mdf)
                for col, label, opacity in _PHASES["SQL"]:
                    if label in _brk_sub_hidden:
                        continue
                    y_vals = _col(mdf, col)
                    fig2.add_trace(go.Bar(
                        name=f"SQL {label}{suffix}", x=x_sql, y=y_vals,
                        base=sql_base, offsetgroup=f"sql_{model_idx}",
                        marker_color=SQL_COLORS.get(model, "#ce93d8"), opacity=opacity,
                        legendgroup=f"SQL_{model}",
                        customdata=y_vals,
                        hovertemplate=_OVERHEAD_HOVER if label == "Overhead" else f"<b>%{{x}}</b><br>{label}: %{{customdata:.0f}} ms<extra></extra>",
                    ))
                    sql_base = [b + v for b, v in zip(sql_base, y_vals)]

            # ── Genie Space bars (stacked SDK / API / Poll / Fetch phases) ──
            if "genie_results" in df_columns:
                _genie_labels_brk = []
                if len(mdf) > 0:
                    first_g = mdf["genie_results"].iloc[0]
                    if isinstance(first_g, dict):
                        _genie_labels_brk = sorted(first_g.keys())
                # Genie breakdown: API-Start + Polling + Fetch + Overhead
                _genie_phases = [
                    ("start_s",   "API-Start", 1.00),
                    ("polling_s", "Polling",   0.65),
                    ("fetch_s",   "Fetch",     0.45),
                ]
                for g_idx, g_label in enumerate(_genie_labels_brk):
                    if f"genie_{g_label}" in _brk_hidden:
                        continue
                    g_color = list(GENIE_COLORS.values())[g_idx % len(GENIE_COLORS)]
                    g_short = g_label.replace("Genie: ", "Genie Spaces ")
                    genie_rows = mdf["genie_results"].tolist()
                    has_bd = any(
                        isinstance(r, dict)
                        and g_label in r
                        and isinstance(r[g_label].get("latency_breakdown"), dict)
                        for r in genie_rows
                    )
                    if has_bd:
                        g_base = [0.0] * len(mdf)
                        # Named phases: API-Start, Polling
                        for phase_key, phase_label, opacity in _genie_phases:
                            if phase_label in _brk_sub_hidden:
                                continue
                            phase_vals = [
                                (r[g_label]["latency_breakdown"].get(phase_key, 0.0) * 1000.0
                                 if isinstance(r, dict) and g_label in r
                                 and isinstance(r[g_label].get("latency_breakdown"), dict)
                                 else 0.0)
                                for r in genie_rows
                            ]
                            fig2.add_trace(go.Bar(
                                name=f"{g_short} {phase_label}{suffix}",
                                x=mdf["question_id"].tolist(),
                                y=phase_vals,
                                base=g_base,
                                offsetgroup=f"genie_{g_idx}_{model_idx}",
                                marker_color=g_color,
                                opacity=opacity,
                                legendgroup=f"Genie_{g_label}_{model}",
                                customdata=phase_vals,
                                hovertemplate=(
                                    f"<b>%{{x}}</b><br>{g_short} {phase_label}: "
                                    "%{customdata:.0f} ms<extra></extra>"
                                ),
                            ))
                            g_base = [b + v for b, v in zip(g_base, phase_vals)]
                        # Overhead = wall_ms - (API-Start + Polling + Fetch)
                        overhead_vals = []
                        for r in genie_rows:
                            if isinstance(r, dict) and g_label in r:
                                wall = r[g_label].get("wall_ms", 0)
                                bd = r[g_label].get("latency_breakdown") or {}
                                named = (bd.get("start_s", 0.0) + bd.get("polling_s", 0.0) + bd.get("fetch_s", 0.0)) * 1000.0
                                overhead_vals.append(max(0.0, wall - named))
                            else:
                                overhead_vals.append(0.0)
                        if "Overhead" not in _brk_sub_hidden:
                            fig2.add_trace(go.Bar(
                                name=f"{g_short} Overhead{suffix}",
                                x=mdf["question_id"].tolist(),
                                y=overhead_vals,
                                base=g_base,
                                offsetgroup=f"genie_{g_idx}_{model_idx}",
                                marker_color=g_color,
                                opacity=0.15,
                                legendgroup=f"Genie_{g_label}_{model}",
                                customdata=overhead_vals,
                                hovertemplate=(
                                    f"<b>%{{x}}</b><br>{g_short} Overhead: "
                                    "%{customdata:.0f} ms<extra></extra>"
                                ),
                            ))
                    else:
                        # Fallback: single wall_ms bar when no breakdown stored yet
                        g_wall_vals = [
                            (r.get(g_label, {}).get("wall_ms", 0) if isinstance(r, dict) else 0)
                            for r in genie_rows
                        ]
                        fig2.add_trace(go.Bar(
                            name=f"{g_short} Gesamt{suffix}",
                            x=mdf["question_id"].tolist(),
                            y=g_wall_vals,
                            offsetgroup=f"genie_{g_idx}_{model_idx}",
                            marker_color=g_color,
                            opacity=1.0,
                            legendgroup=f"Genie_{g_label}_{model}",
                            customdata=g_wall_vals,
                            hovertemplate=(
                                f"<b>%{{x}}</b><br>{g_short}: "
                                "%{customdata:.0f} ms<extra></extra>"
                            ),
                        ))

        fig2.update_layout(**{**_CHART_LAYOUT, "height": 360}, barmode="group", yaxis_title="ms")
        st.plotly_chart(fig2, use_container_width=True, key=f"brk_{group_label}")

    # ── Per-repetition detail expander (only for multi-run data) ───────────
    if "_all_runs" in df_columns and "_num_repetitions" in df_columns:
        _n_reps = df["_num_repetitions"].iloc[0] if len(df) > 0 else 1
        if _n_reps > 1:
            with st.expander(f"📊 Einzellauf-Details ({_n_reps} Wiederholungen)", expanded=False):
                st.caption(
                    "Zeigt die Latenzwerte jeder einzelnen Wiederholung. "
                    "Die Diagramme oben zeigen den Durchschnitt."
                )
                for _, row in df.iterrows():
                    q_id = row.get("question_id", "—")
                    model = row.get("model_name", "—")
                    all_runs_data = row.get("_all_runs", [])
                    if not all_runs_data or not isinstance(all_runs_data, list):
                        continue

                    st.markdown(f"**{q_id}** · {model}")

                    # Build a detail table from individual runs
                    detail_rows = []
                    for run_item in all_runs_data:
                        rep_nr = run_item.get("repetition", "?")
                        # Handle both corpus-style (flat dict) and single-bm-style (nested ont_result/sql_result)
                        if "ont_result" in run_item:
                            ont_r = run_item["ont_result"]
                            sql_r = run_item["sql_result"]
                            o_wall = ont_r.get("total_elapsed", 0.0) * 1000.0
                            o_llm  = ont_r.get("sparql_gen_time", 0.0) * 1000.0
                            o_exec = ont_r.get("sparql_exec_time", 0.0) * 1000.0
                            o_viz  = ont_r.get("viz_time", 0.0) * 1000.0
                            o_oh   = max(0.0, o_wall - o_llm - o_exec - o_viz)
                            s_wall = sql_r.get("total_elapsed", 0.0) * 1000.0
                            s_llm  = sql_r.get("sql_gen_time", 0.0) * 1000.0
                            s_exec = sql_r.get("sql_exec_time", 0.0) * 1000.0
                            s_oh   = max(0.0, s_wall - s_llm - s_exec)
                            o_err  = "✗" if ont_r.get("error") else "✓"
                            s_err  = "✗" if sql_r.get("error") else "✓"
                        else:
                            # Corpus aggregator stores flat records
                            o_wall = run_item.get("ont_total_ms", 0)
                            o_llm  = run_item.get("ont_llm_ms", 0)
                            o_exec = run_item.get("ont_exec_ms", 0)
                            o_oh   = run_item.get("ont_overhead_ms", 0)
                            s_wall = run_item.get("sql_total_ms", 0)
                            s_llm  = run_item.get("sql_llm_ms", 0)
                            s_exec = run_item.get("sql_exec_ms", 0)
                            s_oh   = run_item.get("sql_overhead_ms", 0)
                            o_err  = "✗" if run_item.get("ont_error_msg") else "✓"
                            s_err  = "✗" if run_item.get("sql_error_msg") else "✓"

                        detail_rows.append({
                            "Wdh.": rep_nr,
                            "ONT Gesamt": f"{o_wall:.0f} ms",
                            "ONT LLM": f"{o_llm:.0f} ms",
                            "ONT Exec": f"{o_exec:.0f} ms",
                            "ONT Overhead": f"{o_oh:.0f} ms",
                            "ONT": o_err,
                            "SQL Gesamt": f"{s_wall:.0f} ms",
                            "SQL LLM": f"{s_llm:.0f} ms",
                            "SQL Exec": f"{s_exec:.0f} ms",
                            "SQL Overhead": f"{s_oh:.0f} ms",
                            "SQL": s_err,
                        })

                    if detail_rows:
                        st.dataframe(
                            pd.DataFrame(detail_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

    # ── Token-Aufschlüsselung (grouped bar chart) ──────────────────────────
    # Shows the input prompt decomposed into 3 summable segments:
    #   Schema Context + Instructions + User Message = Total Input
    # Plus the output tokens as a separate transparent segment.
    _tok_new = ["ont_prompt_token_count", "ont_query_token_count", "sql_prompt_token_count", "sql_query_token_count"]
    _tok_legacy = ["ont_prompt_tokens", "ont_query_tokens", "sql_prompt_tokens", "sql_query_tokens"]
    _breakdown_cols = ["ont_schema_context_size_tokens", "ont_instructions_tokens", "ont_user_message_tokens"]

    _has_breakdown = any(c in df_columns for c in _breakdown_cols)

    if any(c in df_columns for c in _tok_new + _tok_legacy):
        # ── Static KPIs for Schema Context & Instruktionen (constant per run) ─
        _schema_cols = ["ont_schema_context_size_tokens", "sql_schema_context_size_tokens",
                        "schema_ont_approx_tokens", "schema_sql_approx_tokens"]
        _instr_cols = ["ont_instructions_tokens", "sql_instructions_tokens"]
        _has_any_schema = any(c in df_columns for c in _schema_cols)
        _has_any_instr = any(c in df_columns for c in _instr_cols)

        if _has_any_schema or _has_any_instr:
            st.caption(
                "**Prompt-Anteile** — Ø Schema-Kontext & Instruktionen pro Frage (konstante Prompt-Bestandteile, "
                "unabhängig von der Nutzerfrage):"
            )
            kpi_cols = st.columns(len(models) * 2)
            for i, model in enumerate(models):
                mdf_s = df[df["model_name"] == model] if "model_name" in df_columns else df

                # Schema KPIs
                ont_sc = _tok_col(mdf_s, "ont_schema_context_size_tokens", "schema_ont_approx_tokens")
                sql_sc = _tok_col(mdf_s, "sql_schema_context_size_tokens", "schema_sql_approx_tokens")
                ont_sc_avg = int(sum(ont_sc) / max(len(ont_sc), 1))
                sql_sc_avg = int(sum(sql_sc) / max(len(sql_sc), 1))

                # Schema ratio
                ont_prompt = _tok_col(mdf_s, "ont_prompt_token_count", "ont_prompt_tokens")
                ont_prompt_avg = int(sum(ont_prompt) / max(len(ont_prompt), 1))
                ont_ratio = f" ({ont_sc_avg / ont_prompt_avg * 100:.0f}%)" if ont_prompt_avg > 0 else ""
                sql_prompt = _tok_col(mdf_s, "sql_prompt_token_count", "sql_prompt_tokens")
                sql_prompt_avg = int(sum(sql_prompt) / max(len(sql_prompt), 1))
                sql_ratio = f" ({sql_sc_avg / sql_prompt_avg * 100:.0f}%)" if sql_prompt_avg > 0 else ""

                # Instruktionen KPIs
                ont_instr = _col(mdf_s, "ont_instructions_tokens")
                sql_instr = _col(mdf_s, "sql_instructions_tokens")
                ont_instr_avg = int(sum(ont_instr) / max(len(ont_instr), 1))
                sql_instr_avg = int(sum(sql_instr) / max(len(sql_instr), 1))

                # Graph-RAG reduction (ONT)
                rag_col = _col(mdf_s, "ont_graph_rag_reduction_pct")
                rag_applied_col = _col(mdf_s, "ont_graph_rag_applied")
                rag_count = sum(1 for v in rag_applied_col if v)
                rag_avg = sum(v for v, a in zip(rag_col, rag_applied_col) if a) / max(rag_count, 1) if rag_count > 0 else 0.0
                _rag_sign = "−" if rag_avg >= 0 else "+"
                rag_hint = f" · Graph-RAG Ø {_rag_sign}{abs(rag_avg):.0f}% bei {rag_count}/{len(mdf_s)} Fragen" if rag_count > 0 else ""

                # Graph-RAG reduction (SQL)
                sql_rag_col = _col(mdf_s, "sql_graph_rag_reduction_pct")
                sql_rag_applied_col = _col(mdf_s, "sql_graph_rag_applied")
                sql_rag_count = sum(1 for v in sql_rag_applied_col if v)
                sql_rag_avg = sum(v for v, a in zip(sql_rag_col, sql_rag_applied_col) if a) / max(sql_rag_count, 1) if sql_rag_count > 0 else 0.0
                _sql_rag_sign = "−" if sql_rag_avg >= 0 else "+"
                sql_rag_hint = f" · Graph-RAG Ø {_sql_rag_sign}{abs(sql_rag_avg):.0f}% bei {sql_rag_count}/{len(mdf_s)} Fragen" if sql_rag_count > 0 else ""

                kpi_cols[i * 2].metric(
                    label=f"📐 {model} · SPARQL mit Ontologie",
                    value=f"Ø {ont_sc_avg:,} tok{ont_ratio}",
                    delta=f"+ {ont_instr_avg:,} Instruktionen",
                    delta_color="off",
                    help=(
                        f"**Schema-Kontext** ({ont_sc_avg:,} tok): Die vollständige OWL-Ontologie-Beschreibung "
                        f"(Klassen, Properties, Domains, Ranges) — definiert das Vokabular für die SPARQL-Generierung. "
                        f"Wird bei aktivem Graph-RAG auf relevante Teilgraphen reduziert.\n\n"
                        f"**Instruktionen** ({ont_instr_avg:,} tok): Feste Regeln, Beispiel-Queries (Few-Shot), "
                        f"Anti-Patterns und das Sandwich-Defense-Suffix gegen Prompt-Injection.\n\n"
                        f"**Anteil am Gesamt-Prompt**: {ont_ratio.strip()} des gesamten Input-Tokens sind Schema-Kontext.\n\n"
                        f"ℹ️ Je größer der Schema-Kontext, desto mehr 'Wissen' hat das LLM über die Ontologie — "
                        f"aber auch höhere Latenz und Kosten. Graph-RAG reduziert dies gezielt."
                        f"{rag_hint}"
                    ),
                )
                kpi_cols[i * 2 + 1].metric(
                    label=f"📐 {model} · SQL",
                    value=f"Ø {sql_sc_avg:,} tok{sql_ratio}",
                    delta=f"+ {sql_instr_avg:,} Instruktionen",
                    delta_color="off",
                    help=(
                        f"**Schema-Kontext** ({sql_sc_avg:,} tok): Die vollständige DDL-Beschreibung aller 3 Tabellen "
                        f"(Assembly, Paintshop, Bodyshop) inkl. Spaltentypen, verschachtelter STRUCT-Definitionen "
                        f"(faults_worksteps ARRAY<STRUCT<...>>) und Spark-SQL-Idiome (LATERAL VIEW, SIZE, CTE).\n\n"
                        f"**Instruktionen** ({sql_instr_avg:,} tok): Feste Regeln für korrekte Spark-SQL-Generierung, "
                        f"Beispiel-Queries, Anti-Patterns (z.B. SIZE(-1), LATERAL VIEW nach WHERE) und "
                        f"Sandwich-Defense-Suffix.\n\n"
                        f"**Anteil am Gesamt-Prompt**: {sql_ratio.strip()} des gesamten Input-Tokens sind Schema-Kontext.\n\n"
                        f"ℹ️ Der SQL-Schema-Kontext ist kleiner als der Ontologie-Kontext, weil DDL kompakter ist "
                        f"als OWL-Tripel. Graph-RAG filtert auf die relevantesten Spalten herunter."
                        f"{sql_rag_hint}"
                    ),
                )
            # Show Graph-RAG info line if active (ONT + SQL)
            if any("ont_graph_rag_applied" in df_columns for _ in [1]):
                rag_all = _col(df, "ont_graph_rag_reduction_pct")
                rag_applied_all = _col(df, "ont_graph_rag_applied")
                n_applied = sum(1 for v in rag_applied_all if v)

                sql_rag_all = _col(df, "sql_graph_rag_reduction_pct")
                sql_rag_applied_all = _col(df, "sql_graph_rag_applied")
                n_sql_applied = sum(1 for v in sql_rag_applied_all if v)

                if n_applied > 0 or n_sql_applied > 0:
                    avg_reduction = sum(v for v, a in zip(rag_all, rag_applied_all) if a) / max(n_applied, 1) if n_applied > 0 else 0.0
                    sql_avg_reduction = sum(v for v, a in zip(sql_rag_all, sql_rag_applied_all) if a) / max(n_sql_applied, 1) if n_sql_applied > 0 else 0.0
                    parts = []
                    if n_applied > 0:
                        parts.append(f"SPARQL: **{avg_reduction:.1f}%** ({n_applied}/{len(df)} Fragen)")
                    if n_sql_applied > 0:
                        parts.append(f"SQL: **{sql_avg_reduction:.1f}%** ({n_sql_applied}/{len(df)} Fragen)")
                    st.caption(
                        f"🔬 **Graph-RAG aktiv** — Ø Reduktion: {' · '.join(parts)}",
                        help=(
                            "Graph-RAG extrahiert aus der Nutzerfrage relevante Klassen "
                            "(z.B. 'Fahrzeugtyp' → mrs:ModelType, mrs:VIN) und filtert den "
                            "Schema-Kontext auf nur diese Seed-Klassen herunter.\n\n"
                            "Ergebnis: 50-85% weniger Schema-Tokens im Prompt.\n\n"
                            "Die TRAVERSAL PATHS im System-Prompt decken alle Property-Chains "
                            "bereits explizit ab — die PROPERTIES-Sektion muss nur zeigen, "
                            "welche Klassen/Datentypen direkt relevant sind.\n\n"
                            "Fallbacks (voller Kontext):\n"
                            "• Multi-Hop-Fragen (z.B. 'über alle Bereiche')\n"
                            "• Keine Keywords erkannt\n"
                            "• Seeds + Nachbarn decken ≥80% aller Klassen ab"
                        ),
                    )

        # ── Token chart: only Nutzerfrage + Query Tokens (variable per question)
        st.caption(
            "**Token-Verbrauch pro Frage:** "
            "Nutzerfrage (Input, variabel pro Frage) + Query Tokens (Output, die vom LLM generierte "
            "SPARQL- bzw. SQL-Query). Schema-Kontext und Instruktionen sind hier NICHT enthalten — "
            "sie sind konstant und oben separat ausgewiesen."
        )

        for group_label, qids in groups:
            st.markdown(f"**Tokens · {group_label}**")
            group_df = df[df["question_id"].isin(qids)]

            fig_tok = go.Figure()
            for model in models:
                mdf = group_df[group_df["model_name"] == model]
                if len(mdf) == 0:
                    continue
                suffix = f" · {model}" if len(models) > 1 else ""
                x_ont = mdf["question_id"].tolist()
                x_sql = mdf["question_id"].tolist()
                c_ont = ONT_COLORS.get(model, "#4fc3f7")
                c_sql = SQL_COLORS.get(model, "#ce93d8")

                if _has_breakdown:
                    model_idx = models.index(model)
                    # ── ONT: Nutzerfrage + Query Tokens (stacked) ──────────────
                    ont_user = _col(mdf, "ont_user_message_tokens")
                    ont_query = _tok_col(mdf, "ont_query_token_count", "ont_query_tokens")
                    fig_tok.add_trace(go.Bar(
                        name=f"SPARQL mit Ontologie Nutzerfrage{suffix}", x=x_ont,
                        y=ont_user, base=[0.0] * len(mdf),
                        offsetgroup=f"ont_{model_idx}",
                        marker_color=c_ont, opacity=1.0, legendgroup=f"ont_{model}",
                        hovertemplate="<b>%{x}</b><br>Nutzerfrage: %{y:,} tok<extra></extra>",
                    ))
                    fig_tok.add_trace(go.Bar(
                        name=f"SPARQL mit Ontologie Query Tokens{suffix}", x=x_ont,
                        y=ont_query, base=ont_user,
                        offsetgroup=f"ont_{model_idx}",
                        marker_color=c_ont, opacity=0.4, legendgroup=f"ont_{model}",
                        hovertemplate="<b>%{x}</b><br>Query Tokens: %{y:,}<extra></extra>",
                    ))
                    # ── SQL: Nutzerfrage + Query Tokens (stacked) ──────────────
                    sql_user = _col(mdf, "sql_user_message_tokens")
                    sql_query = _tok_col(mdf, "sql_query_token_count", "sql_query_tokens")
                    fig_tok.add_trace(go.Bar(
                        name=f"SQL Nutzerfrage{suffix}", x=x_sql,
                        y=sql_user, base=[0.0] * len(mdf),
                        offsetgroup=f"sql_{model_idx}",
                        marker_color=c_sql, opacity=1.0, legendgroup=f"sql_{model}",
                        hovertemplate="<b>%{x}</b><br>Nutzerfrage: %{y:,} tok<extra></extra>",
                    ))
                    fig_tok.add_trace(go.Bar(
                        name=f"SQL Query Tokens{suffix}", x=x_sql,
                        y=sql_query, base=sql_user,
                        offsetgroup=f"sql_{model_idx}",
                        marker_color=c_sql, opacity=0.4, legendgroup=f"sql_{model}",
                        hovertemplate="<b>%{x}</b><br>Query Tokens: %{y:,}<extra></extra>",
                    ))
                else:
                    model_idx = models.index(model)
                    # ── Fallback: monolithic prompt bar (legacy data) ─────────
                    ont_input = _tok_col(mdf, "ont_prompt_token_count", "ont_prompt_tokens")
                    ont_output = _tok_col(mdf, "ont_query_token_count", "ont_query_tokens")
                    fig_tok.add_trace(go.Bar(
                        name=f"SPARQL mit Ontologie Input{suffix}", x=x_ont,
                        y=ont_input, base=[0.0] * len(mdf),
                        offsetgroup=f"ont_{model_idx}",
                        marker_color=c_ont, opacity=1.0, legendgroup=f"ont_{model}",
                        hovertemplate="<b>%{x}</b><br>Input Tokens: %{y:,}<extra></extra>",
                    ))
                    fig_tok.add_trace(go.Bar(
                        name=f"SPARQL mit Ontologie Query Tokens{suffix}", x=x_ont,
                        y=ont_output, base=ont_input,
                        offsetgroup=f"ont_{model_idx}",
                        marker_color=c_ont, opacity=0.4, legendgroup=f"ont_{model}",
                        hovertemplate="<b>%{x}</b><br>Query Tokens: %{y:,}<extra></extra>",
                    ))
                    sql_input = _tok_col(mdf, "sql_prompt_token_count", "sql_prompt_tokens")
                    sql_output = _tok_col(mdf, "sql_query_token_count", "sql_query_tokens")
                    fig_tok.add_trace(go.Bar(
                        name=f"SQL Input{suffix}", x=x_sql,
                        y=sql_input, base=[0.0] * len(mdf),
                        offsetgroup=f"sql_{model_idx}",
                        marker_color=c_sql, opacity=1.0, legendgroup=f"sql_{model}",
                        hovertemplate="<b>%{x}</b><br>Input Tokens: %{y:,}<extra></extra>",
                    ))
                    fig_tok.add_trace(go.Bar(
                        name=f"SQL Query Tokens{suffix}", x=x_sql,
                        y=sql_output, base=sql_input,
                        offsetgroup=f"sql_{model_idx}",
                        marker_color=c_sql, opacity=0.4, legendgroup=f"sql_{model}",
                        hovertemplate="<b>%{x}</b><br>Query Tokens: %{y:,}<extra></extra>",
                    ))

            # ── Graph-RAG Savings Annotations on ONT bars ─────────────────
            if "ont_graph_rag_reduction_pct" in df_columns:
                for model in models:
                    mdf_ann = group_df[group_df["model_name"] == model]
                    if len(mdf_ann) == 0:
                        continue
                    rag_pcts = _col(mdf_ann, "ont_graph_rag_reduction_pct")
                    rag_applied = _col(mdf_ann, "ont_graph_rag_applied")
                    x_ids = mdf_ann["question_id"].tolist()
                    for xi, (qid, pct, applied) in enumerate(zip(x_ids, rag_pcts, rag_applied)):
                        if applied and pct > 0:
                            # Get the total ONT bar height for annotation position
                            if _has_breakdown:
                                bar_top = (_col(mdf_ann, "ont_user_message_tokens")[xi]
                                           + _tok_col(mdf_ann, "ont_query_token_count", "ont_query_tokens")[xi])
                            else:
                                bar_top = (_tok_col(mdf_ann, "ont_prompt_token_count", "ont_prompt_tokens")[xi]
                                           + _tok_col(mdf_ann, "ont_query_token_count", "ont_query_tokens")[xi])
                            fig_tok.add_annotation(
                                x=qid, y=bar_top,
                                text=f"<b>−{pct:.0f}%</b>",
                                showarrow=False, xshift=-18, yshift=8,
                                font=dict(size=9, color="#69db7c"),
                                hovertext="Graph-RAG Ersparnis",
                                hoverlabel=dict(bgcolor="#2d2d2d", font_color="#69db7c"),
                            )

            # ── Graph-RAG Savings Annotations on SQL bars ─────────────────
            if "sql_graph_rag_reduction_pct" in df_columns:
                for model in models:
                    mdf_ann = group_df[group_df["model_name"] == model]
                    if len(mdf_ann) == 0:
                        continue
                    sql_rag_pcts = _col(mdf_ann, "sql_graph_rag_reduction_pct")
                    sql_rag_applied = _col(mdf_ann, "sql_graph_rag_applied")
                    x_ids = mdf_ann["question_id"].tolist()
                    for xi, (qid, pct, applied) in enumerate(zip(x_ids, sql_rag_pcts, sql_rag_applied)):
                        if applied and pct > 0:
                            if _has_breakdown:
                                bar_top = (_col(mdf_ann, "sql_user_message_tokens")[xi]
                                           + _tok_col(mdf_ann, "sql_query_token_count", "sql_query_tokens")[xi])
                            else:
                                bar_top = (_tok_col(mdf_ann, "sql_prompt_token_count", "sql_prompt_tokens")[xi]
                                           + _tok_col(mdf_ann, "sql_query_token_count", "sql_query_tokens")[xi])
                            fig_tok.add_annotation(
                                x=qid, y=bar_top,
                                text=f"<b>−{pct:.0f}%</b>",
                                showarrow=False, xshift=18, yshift=8,
                                font=dict(size=9, color="#69db7c"),
                                hovertext="SQL Graph-RAG Ersparnis",
                                hoverlabel=dict(bgcolor="#2d2d2d", font_color="#69db7c"),
                            )

            fig_tok.update_layout(
                **{**_CHART_LAYOUT, "height": 380, "margin": dict(l=50, r=20, t=10, b=90)},
                barmode="group", yaxis_title="Tokens",
            )
            st.plotly_chart(fig_tok, use_container_width=True, key=f"tok_{group_label}")




# =============================================================================
# Tab 3: Status (styled HTML table)
# =============================================================================

_ERR_STYLE = {
    "none": ("#6b7280", "#ffffff"), "timeout": ("#d97706", "#ffffff"),
    "syntax_error": ("#dc2626", "#ffffff"), "empty_result": ("#2563eb", "#ffffff"),
    "other": ("#6b7280", "#ffffff"),
}
_ERR_DISPLAY = {
    "none": "RUN", "timeout": "Timeout", "syntax_error": "Syntax Error",
    "empty_result": "Empty Result", "other": "Other",
}


def _normalize_enum(val: str) -> str:
    """Strip Enum class prefix if present (e.g. 'ErrorType.none' → 'none')."""
    if "." in val:
        return val.split(".", 1)[1]
    return val


def _render_tab_status(df: pd.DataFrame, df_columns: set, models: list, questions: list) -> None:
    legend_items = "".join(
        f"<span style='background:{bg};color:{fg};"
        f"border-radius:6px;padding:4px 10px;font-size:11px;font-weight:600;"
        f"margin-right:8px;display:inline-block;'>{_ERR_DISPLAY[k]}</span>"
        for k, (bg, fg) in _ERR_STYLE.items()
    )
    legend_items += (
        "<span style='background:#16a34a;color:#ffffff;"
        "border-radius:6px;padding:4px 10px;font-size:11px;font-weight:600;"
        "margin-right:8px;display:inline-block;'>✓ VERIFIED</span>"
    )
    # legend rendered inside the status table card below

    _has_jd = "join_depth" in df_columns
    _has_tp = "triple_path_depth" in df_columns

    header = (
        "<th style='padding:12px 16px;text-align:left;font-size:13px;font-weight:600;"
        "color:#1f2937;border-bottom:2px solid #B8D4F4;'>Frage</th>"
    )
    if _has_tp:
        header += (
            "<th style='padding:12px 10px;text-align:center;font-size:11px;font-weight:600;"
            "color:#0057A8;border-bottom:2px solid #B8D4F4;white-space:nowrap;'>"
            "Path<br>Depth</th>"
        )
    if _has_jd:
        header += (
            "<th style='padding:12px 10px;text-align:center;font-size:11px;font-weight:600;"
            "color:#7e22ce;border-bottom:2px solid #B8D4F4;white-space:nowrap;'>"
            "Join<br>Depth</th>"
        )
    # Detect Genie space labels from first row
    _genie_labels: list[str] = []
    if "genie_results" in df_columns and len(df) > 0:
        _first_genie = df["genie_results"].iloc[0]
        if isinstance(_first_genie, dict):
            _genie_labels = sorted(_first_genie.keys())
    _genie_colors = list(GENIE_COLORS.values())

    for model in models:
        for panel, color in [("SPARQL mit Ontologie", "#0057A8"), ("SQL", "#7e22ce")]:
            header += (
                f"<th style='padding:12px 14px;text-align:center;font-size:13px;"
                f"font-weight:700;color:{color};border-bottom:2px solid #B8D4F4;'>"
                f"{panel}<div style='font-size:10px;opacity:0.5;font-weight:400;"
                f"margin-top:2px;'>{model}</div></th>"
            )
        # Genie Space header columns
        for g_idx, g_label in enumerate(_genie_labels):
            g_color = _genie_colors[g_idx % len(_genie_colors)]
            g_short = g_label.replace("Genie: ", "Genie Spaces ")
            header += (
                f"<th style='padding:12px 14px;text-align:center;font-size:13px;"
                f"font-weight:700;color:{g_color};border-bottom:2px solid #B8D4F4;'>"
                f"{g_short}<div style='font-size:10px;opacity:0.5;font-weight:400;"
                f"margin-top:2px;'>{model}</div></th>"
            )

    rows_html = ""
    for idx, qid in enumerate(questions):
        qdf = df[df["question_id"] == qid]
        q_text = qdf["question"].iloc[0] if "question" in df_columns and len(qdf) else qid
        row_bg = "#F2F7FC" if idx % 2 == 0 else "#ffffff"
        row = (
            f"<tr style='background:{row_bg};'>"
            f"<td style='padding:10px 16px;border-bottom:1px solid #f3f4f6;'>"
            f"<div style='font-size:13px;font-weight:700;color:#111827;margin-bottom:3px;'>{qid}</div>"
            f"<div style='font-size:11px;color:#6b7280;line-height:1.4;word-break:break-word;'>{q_text}</div></td>"
        )
        if _has_tp:
            tp_val = int(qdf["triple_path_depth"].iloc[0]) if len(qdf) and pd.notna(qdf["triple_path_depth"].iloc[0]) else "—"
            row += (
                f"<td style='text-align:center;padding:10px 10px;font-size:13px;"
                f"font-weight:600;color:#0057A8;border-bottom:1px solid #f3f4f6;'>"
                f"{tp_val}</td>"
            )
        if _has_jd:
            jd_val = int(qdf["join_depth"].iloc[0]) if len(qdf) and pd.notna(qdf["join_depth"].iloc[0]) else "—"
            row += (
                f"<td style='text-align:center;padding:10px 10px;font-size:13px;"
                f"font-weight:600;color:#7e22ce;border-bottom:1px solid #f3f4f6;'>"
                f"{jd_val}</td>"
            )
        for model in models:
            mqdf = qdf[qdf["model_name"] == model] if "model_name" in df_columns else qdf
            for err_col in ("ont_error_type", "sql_error_type"):
                if len(mqdf) == 0:
                    row += ("<td style='text-align:center;padding:10px 12px;"
                            "border-bottom:1px solid #f3f4f6;'>"
                            "<span style='opacity:0.3;'>—</span></td>")
                    continue
                err = _normalize_enum(str(mqdf[err_col].iloc[0])) if err_col in mqdf.columns else "none"
                bg, fg = _ERR_STYLE.get(err, ("#6b7280", "#ffffff"))
                label = _ERR_DISPLAY.get(err, err.replace("_", " ").title())
                # Get the actual error message
                msg_col = err_col.replace("_type", "_msg")  # ont_error_type -> ont_error_msg
                err_msg = ""
                if err != "none" and msg_col in mqdf.columns:
                    raw_msg = mqdf[msg_col].iloc[0]
                    if pd.notna(raw_msg) and str(raw_msg).strip():
                        err_msg = str(raw_msg).strip()
                err_msg_html = (
                    f"<details style='margin-top:5px;max-width:180px;text-align:left;"
                    f"display:inline-block;'>"
                    f"<summary style='font-size:10px;color:#991b1b;cursor:pointer;"
                    f"list-style:none;'>▶ Details</summary>"
                    f"<div style='font-size:10px;color:#991b1b;margin-top:3px;"
                    f"line-height:1.3;word-break:break-word;'>{err_msg}</div>"
                    f"</details>"
                ) if err_msg else ""
                # Check verified status
                panel_key = "ont" if "ont_" in err_col else "sql"
                v_key = f"{qid}__{model}__{panel_key}"
                verified = _get_verified_status().get(v_key, False)
                verified_html = (
                    "<span style='background:#16a34a;color:#ffffff;"
                    "border-radius:6px;padding:3px 8px;font-size:10px;font-weight:600;"
                    "display:inline-block;letter-spacing:0.3px;'>"
                    "✓ VERIFIED</span>"
                ) if verified else ""
                # Per-run status indicators (visible only when num_repetitions > 1)
                per_run_html = ""
                if "_all_runs" in df_columns and len(mqdf) > 0:
                    _runs_data = mqdf["_all_runs"].iloc[0]
                    if isinstance(_runs_data, list) and len(_runs_data) > 1:
                        per_badges = []
                        for run_item in _runs_data:
                            rep = run_item.get("repetition", "?")
                            r_err = _normalize_enum(str(run_item.get(err_col, "none") or "none"))
                            if r_err == "none":
                                per_badges.append(
                                    f"<span style='font-size:10px;color:#69db7c;"
                                    f"white-space:nowrap;'>Wdh.{rep} ✓</span>"
                                )
                            else:
                                r_label = _ERR_DISPLAY.get(r_err, r_err)
                                per_badges.append(
                                    f"<span style='font-size:10px;color:#ff6b6b;"
                                    f"white-space:nowrap;'>Wdh.{rep} ✗ {r_label}</span>"
                                )
                        per_run_html = (
                            "<div style='margin-top:6px;display:flex;flex-direction:column;"
                            "align-items:center;gap:2px;border-top:1px solid rgba(128,128,128,0.15);"
                            "padding-top:5px;'>" + "".join(per_badges) + "</div>"
                        )
                row += (
                    f"<td style='text-align:center;padding:10px 12px;"
                    f"border-bottom:1px solid #f3f4f6;'>"
                    f"<div style='display:flex;flex-direction:column;align-items:center;"
                    f"justify-content:center;gap:6px;'>"
                    f"<span style='background:{bg};color:{fg};"
                    f"border-radius:6px;padding:5px 12px;font-size:12px;font-weight:600;"
                    f"display:inline-block;white-space:nowrap;letter-spacing:0.3px;'>"
                    f"{label}</span>{verified_html}{err_msg_html}{per_run_html}</div></td>"
                )
        # Genie Space cells
            if _genie_labels:
                genie_data = mqdf["genie_results"].iloc[0] if "genie_results" in mqdf.columns and len(mqdf) > 0 else None
                for g_idx, g_label in enumerate(_genie_labels):
                    if not isinstance(genie_data, dict) or g_label not in genie_data:
                        row += ("<td style='text-align:center;padding:10px 12px;"
                                "border-bottom:1px solid #f3f4f6;'>"
                                "<span style='opacity:0.3;'>—</span></td>")
                        continue
                    g_m = genie_data[g_label]
                    g_success = g_m.get("exec_success", False)
                    g_status = g_m.get("status", "UNKNOWN")
                    g_wall = g_m.get("wall_ms", 0)
                    g_error = g_m.get("error")
                    g_color = _genie_colors[g_idx % len(_genie_colors)]
                    if g_success:
                        badge_bg, badge_fg = "#dcfce7", "#166534"
                        badge_label = f"OK | {g_status}"
                    else:
                        badge_bg, badge_fg = "#fee2e2", "#991b1b"
                        badge_label = f"FAIL | {g_status}"
                    g_err_html = ""
                    if g_error:
                        g_err_short = str(g_error)[:80]
                        g_err_html = (
                            f"<details style='margin-top:5px;max-width:180px;text-align:left;"
                            f"display:inline-block;'>"
                            f"<summary style='font-size:10px;color:#991b1b;cursor:pointer;"
                            f"list-style:none;'>▶ Details</summary>"
                            f"<div style='font-size:10px;color:#991b1b;margin-top:3px;"
                            f"line-height:1.3;word-break:break-word;'>{g_err_short}</div>"
                            f"</details>"
                        )
                    row += (
                        f"<td style='text-align:center;padding:10px 12px;"
                        f"border-bottom:1px solid #f3f4f6;'>"
                        f"<div style='display:flex;flex-direction:column;align-items:center;"
                        f"justify-content:center;gap:4px;'>"
                        f"<span style='background:{badge_bg};color:{badge_fg};"
                        f"border-radius:6px;padding:5px 10px;font-size:11px;font-weight:600;"
                        f"display:inline-block;white-space:nowrap;'>{badge_label}</span>"
                        f"{g_err_html}</div></td>"
                    )
        row += "</tr>"
        rows_html += row

    st.html(f"""
<div style="
    background: linear-gradient(135deg, #003A7D 0%, #0057A8 50%, #003A7D 100%);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 8px 0 16px 0;
    box-shadow: 0 8px 32px rgba(0,58,125,0.35);
">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
    <div style="font-size:24px;">✅</div>
    <div style="font-size:15px;font-weight:700;color:#ffffff;letter-spacing:-0.3px;">Status-Übersicht</div>
  </div>
  <div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:14px;">{legend_items}</div>
  <div style="background:rgba(255,255,255,0.96);border-radius:10px;overflow-x:auto;border:1px solid rgba(255,255,255,0.3);">
    <table style="width:100%;border-collapse:collapse;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#1f2937;">
      <thead style="background:#EEF4FB;"><tr>{header}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>""")





# =============================================================================
# Tab 4: Qualität
# =============================================================================

# Per-pipeline correctness evaluation options
_PIPELINE_SCORE_OPTIONS = ["—", "korrekt", "teilweise", "falsch"]
_PIPELINE_SCORE_COLORS = {
    "—":          ("#2a2a2a", "#adb5bd"),
    "korrekt":    ("#1b4332", "#69db7c"),
    "teilweise":  ("#3d2e00", "#ffa726"),
    "falsch":     ("#4a0000", "#dc2626"),
}
_PIPELINE_SCORE_ICONS = {
    "—":         "—",
    "korrekt":   "✓ Korrekt",
    "teilweise": "~ Teilweise",
    "falsch":    "✗ Falsch",
}

# Max questions rendered per page in the Qualität tab.
# Keeps widget count manageable and prevents partial rendering / scroll issues.
_QUAL_PAGE_SIZE = 4


def _get_human_scores() -> dict:
    """Return human scores dict.  Keys: '{qid}__{model}__{pipeline}' → score label.

    Pipeline identifiers: 'ont', 'sql', or sanitised Genie label (e.g. 'Genie_MRS_pure_DPs').
    Score labels: 'korrekt', 'teilweise', 'falsch', or '—' (unrated).
    """
    if "_human_scores" not in st.session_state:
        st.session_state["_human_scores"] = {}
    return st.session_state["_human_scores"]


def _get_verified_status() -> dict:
    """Get verified status dict. Keys: '{qid}__{model}__ont' / '{qid}__{model}__sql'.

    On first access, hydrates from benchmark_runs if verified flags exist there.
    """
    if "_verified_status" not in st.session_state:
        initial: dict = {}
        for run in st.session_state.get("benchmark_runs", []):
            qid = run.get("question_id", "")
            model = run.get("model_name", "")
            if run.get("ont_verified"):
                initial[f"{qid}__{model}__ont"] = True
            if run.get("sql_verified"):
                initial[f"{qid}__{model}__sql"] = True
        st.session_state["_verified_status"] = initial
    return st.session_state["_verified_status"]


@st.fragment
def _render_tab_quality(df: pd.DataFrame, df_columns: set, models: list, questions: list) -> None:
    # CSS fix: ensure tab content is not clipped / allows full scroll
    st.html(
        "<style>"
        "[data-testid='stTabs'] [data-testid='stVerticalBlockBorderWrapper'] { overflow: visible !important; }"
        "[data-testid='stTabs'] > div:nth-child(2) { overflow: visible !important; min-height: 0 !important; }"
        "section.main .block-container { overflow: visible !important; max-height: none !important; }"
        "</style>"
    )

    has_queries = "ont_generated_query" in df_columns and "sql_generated_query" in df_columns
    has_previews = "ont_result_preview" in df_columns and "sql_result_preview" in df_columns

    if not has_queries and not has_previews:
        st.info("Keine Qualitätsdaten verfügbar.")
        return

    scores = _get_human_scores()

    # Build dynamic Genie labels from the data
    _genie_labels_qual: list[str] = []
    if "genie_results" in df_columns and len(df) > 0:
        _first_gr = df["genie_results"].iloc[0]
        if isinstance(_first_gr, dict):
            _genie_labels_qual = sorted(_first_gr.keys())

    # Determine active pipelines for KPI display
    _active_pipelines: list[tuple[str, str]] = []  # (key_suffix, display_label)
    _active_pipelines.append(("ont", "ONT"))
    _active_pipelines.append(("sql", "SQL"))
    for g_lbl in _genie_labels_qual:
        _safe = g_lbl.replace(" ", "_").replace(":", "").replace("/", "_")
        _active_pipelines.append((_safe, g_lbl.replace("Genie: ", "G:")))

    # Count per-pipeline scores
    total_pairs = len(questions) * len(models) * len(_active_pipelines)
    scored_count = sum(1 for v in scores.values() if v != "—")
    korrekt_count = sum(1 for v in scores.values() if v == "korrekt")
    teilweise_count = sum(1 for v in scores.values() if v == "teilweise")
    falsch_count = sum(1 for v in scores.values() if v == "falsch")

    # ── Per-Pipeline Ergebnisübersicht ────────────────────────────────────────
    _n_q = len(questions) * len(models)  # max scores per pipeline
    pipe_cols = st.columns(len(_active_pipelines) + 1)  # +1 for totals
    # First column: totals
    with pipe_cols[0]:
        st.markdown(
            f"<div style='background:#1a1a2e;border-radius:10px;padding:12px 16px;text-align:center;'>"
            f"<div style='font-size:11px;color:#adb5bd;font-weight:600;'>GESAMT</div>"
            f"<div style='font-size:22px;font-weight:700;color:#ffffff;margin:4px 0;'>{scored_count}<span style='font-size:13px;color:#666;'>/{total_pairs}</span></div>"
            f"<div style='font-size:11px;'>"
            f"<span style='color:#69db7c;'>✓{korrekt_count}</span> &nbsp;"
            f"<span style='color:#ffa726;'>~{teilweise_count}</span> &nbsp;"
            f"<span style='color:#dc2626;'>✗{falsch_count}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    # Per-pipeline columns
    for col, (p_key, p_label) in zip(pipe_cols[1:], _active_pipelines):
        p_scores = [v for k, v in scores.items() if k.endswith(f"__{p_key}") and v != "—"]
        p_ok = sum(1 for v in p_scores if v == "korrekt")
        p_part = sum(1 for v in p_scores if v == "teilweise")
        p_bad = sum(1 for v in p_scores if v == "falsch")
        p_total = len(p_scores)
        with col:
            st.markdown(
                f"<div style='background:#1a1a2e;border-radius:10px;padding:12px 16px;text-align:center;'>"
                f"<div style='font-size:11px;color:#adb5bd;font-weight:600;'>{p_label}</div>"
                f"<div style='font-size:22px;font-weight:700;color:#ffffff;margin:4px 0;'>{p_total}<span style='font-size:13px;color:#666;'>/{_n_q}</span></div>"
                f"<div style='font-size:11px;'>"
                f"<span style='color:#69db7c;'>✓{p_ok}</span> &nbsp;"
                f"<span style='color:#ffa726;'>~{p_part}</span> &nbsp;"
                f"<span style='color:#dc2626;'>✗{p_bad}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    st.caption(
        "Bewerte pro Pipeline ob das Ergebnis korrekt, teilweise korrekt oder falsch ist."
    )

    # ── Pagination ──────────────────────────────────────────────────────────
    import math as _math
    _n_questions = len(questions)
    _n_pages = max(1, _math.ceil(_n_questions / _QUAL_PAGE_SIZE))
    _qual_page = st.session_state.get("_qual_page", 0)
    _qual_page = min(_qual_page, _n_pages - 1)  # clamp in case list shrank

    _pg_start = _qual_page * _QUAL_PAGE_SIZE
    _pg_end   = min(_pg_start + _QUAL_PAGE_SIZE, _n_questions)
    page_questions = questions[_pg_start:_pg_end]

    for idx, qid in enumerate(page_questions):
        qdf = df[df["question_id"] == qid]
        if len(qdf) == 0:
            continue

        q_text = qdf["question"].iloc[0] if "question" in df_columns else qid

        st.html(
            f"<div style='background:linear-gradient(135deg,#003A7D 0%,#0057A8 100%);"
            f"border-radius:12px;padding:10px 20px;"
            f"margin:{'24px' if idx > 0 else '8px'} 0 12px 0;"
            f"box-shadow:0 4px 16px rgba(0,58,125,0.2);'>"
            f"<div style='font-size:13px;font-weight:700;color:#B8D4F4;"
            f"font-family:monospace;letter-spacing:0.3px;'>{qid}</div>"
            f"<div style='font-size:12px;color:rgba(255,255,255,0.65);"
            f"margin-top:3px;line-height:1.4;'>{q_text}</div>"
            f"</div>"
        )

        for model in models:
            mqdf = qdf[qdf["model_name"] == model] if "model_name" in df_columns else qdf
            if len(mqdf) == 0:
                continue

            score_key = f"{qid}__{model}"

            if len(models) > 1:
                st.markdown(
                    f"<div style='font-size:11px;opacity:0.5;margin:4px 0 8px 0;'>"
                    f"Modell: {model}</div>",
                    unsafe_allow_html=True,
                )

            col_ont, col_sql = st.columns(2)

            with col_ont:
                ont_rows = int(mqdf["ont_row_count"].iloc[0]) if "ont_row_count" in mqdf.columns else 0
                ont_err = _normalize_enum(str(mqdf["ont_error_type"].iloc[0])) if "ont_error_type" in mqdf.columns else "none"
                st.html(
                    f"<div style='font-size:12px;font-weight:700;color:#0057A8;"
                    f"border-left:3px solid #0057A8;padding:5px 10px;"
                    f"background:#EEF4FB;border-radius:0 6px 6px 0;margin-bottom:8px;'>"
                    f"SPARQL · Ontologie &nbsp;·&nbsp; "
                    f"<span style='font-weight:500;color:#3b71c8;'>{ont_rows} Zeilen</span></div>"
                )
                if ont_err != "none":
                    bg, fg = _ERR_STYLE.get(ont_err, ("#6b7280", "#ffffff"))
                    st.markdown(
                        f"<span style='background:{bg};color:{fg};border:1px solid {fg}55;"
                        f"border-radius:6px;padding:3px 8px;font-size:11px;font-weight:600;'>"
                        f"{_ERR_DISPLAY.get(ont_err, ont_err)}</span>",
                        unsafe_allow_html=True,
                    )
                    ont_err_msg = str(mqdf["ont_error_msg"].iloc[0]) if "ont_error_msg" in mqdf.columns and pd.notna(mqdf["ont_error_msg"].iloc[0]) and str(mqdf["ont_error_msg"].iloc[0]).strip() else ""
                    if ont_err_msg:
                        with st.expander("⚠️ Fehlermeldung anzeigen", expanded=False):
                            st.code(ont_err_msg, language=None)
                # Status override selectbox
                _status_options = list(_ERR_DISPLAY.keys())
                _status_labels = list(_ERR_DISPLAY.values())
                _ont_status_idx = _status_options.index(ont_err) if ont_err in _status_options else 0
                new_ont_status = st.selectbox(
                    "Status",
                    options=_status_options,
                    format_func=lambda x: _ERR_DISPLAY.get(x, x),
                    index=_ont_status_idx,
                    key=f"status_ont_{qid}__{model}",
                    label_visibility="collapsed",
                )
                if new_ont_status != ont_err:
                    _write_error_type_to_runs(qid, model, "ont", new_ont_status)
                    st.rerun(scope="fragment")

                if has_queries:
                    ont_query = mqdf["ont_generated_query"].iloc[0] if "ont_generated_query" in mqdf.columns else ""
                    if ont_query:
                        with st.popover("SPARQL Query", use_container_width=True):
                            st.code(ont_query, language="sparql")
                if has_previews:
                    preview = mqdf["ont_result_preview"].iloc[0] if "ont_result_preview" in mqdf.columns else None
                    if isinstance(preview, list) and len(preview) > 0:
                        _h = min(35 * len(preview) + 38, 500)
                        st.dataframe(pd.DataFrame(preview), use_container_width=True, height=_h)
                    elif ont_err == "none" and ont_rows == 0:
                        st.caption("Keine Ergebnisse zurückgegeben.")
                # Verify toggle
                v_key_ont = f"{qid}__{model}__ont"
                verified_states = _get_verified_status()
                ont_verified = verified_states.get(v_key_ont, False)
                if st.checkbox("✓ Verified", value=ont_verified, key=f"verify_{v_key_ont}"):
                    if not ont_verified:
                        verified_states[v_key_ont] = True
                        st.session_state["_verified_status"] = verified_states
                        _write_verified_to_runs(qid, model, "ont", True)
                        st.rerun(scope="fragment")
                else:
                    if ont_verified:
                        verified_states[v_key_ont] = False
                        st.session_state["_verified_status"] = verified_states
                        _write_verified_to_runs(qid, model, "ont", False)
                        st.rerun(scope="fragment")
                # Per-pipeline correctness evaluation
                _render_pipeline_score_buttons(scores, score_key, "ont", qid, model)

            with col_sql:
                sql_rows = int(mqdf["sql_row_count"].iloc[0]) if "sql_row_count" in mqdf.columns else 0
                sql_err = _normalize_enum(str(mqdf["sql_error_type"].iloc[0])) if "sql_error_type" in mqdf.columns else "none"
                st.html(
                    f"<div style='font-size:12px;font-weight:700;color:#7e22ce;"
                    f"border-left:3px solid #7e22ce;padding:5px 10px;"
                    f"background:#f5f0ff;border-radius:0 6px 6px 0;margin-bottom:8px;'>"
                    f"SQL &nbsp;·&nbsp; "
                    f"<span style='font-weight:500;color:#9c6fd6;'>{sql_rows} Zeilen</span></div>"
                )
                if sql_err != "none":
                    bg, fg = _ERR_STYLE.get(sql_err, ("#6b7280", "#ffffff"))
                    st.markdown(
                        f"<span style='background:{bg};color:{fg};border:1px solid {fg}55;"
                        f"border-radius:6px;padding:3px 8px;font-size:11px;font-weight:600;'>"
                        f"{_ERR_DISPLAY.get(sql_err, sql_err)}</span>",
                        unsafe_allow_html=True,
                    )
                    sql_err_msg = str(mqdf["sql_error_msg"].iloc[0]) if "sql_error_msg" in mqdf.columns and pd.notna(mqdf["sql_error_msg"].iloc[0]) and str(mqdf["sql_error_msg"].iloc[0]).strip() else ""
                    if sql_err_msg:
                        with st.expander("⚠️ Fehlermeldung anzeigen", expanded=False):
                            st.code(sql_err_msg, language=None)
                # Status override selectbox
                _status_options_sql = list(_ERR_DISPLAY.keys())
                _sql_status_idx = _status_options_sql.index(sql_err) if sql_err in _status_options_sql else 0
                new_sql_status = st.selectbox(
                    "Status",
                    options=_status_options_sql,
                    format_func=lambda x: _ERR_DISPLAY.get(x, x),
                    index=_sql_status_idx,
                    key=f"status_sql_{qid}__{model}",
                    label_visibility="collapsed",
                )
                if new_sql_status != sql_err:
                    _write_error_type_to_runs(qid, model, "sql", new_sql_status)
                    st.rerun(scope="fragment")

                if has_queries:
                    sql_query = mqdf["sql_generated_query"].iloc[0] if "sql_generated_query" in mqdf.columns else ""
                    if sql_query:
                        with st.popover("SQL Query", use_container_width=True):
                            st.code(sql_query, language="sql")
                if has_previews:
                    preview = mqdf["sql_result_preview"].iloc[0] if "sql_result_preview" in mqdf.columns else None
                    if isinstance(preview, list) and len(preview) > 0:
                        _h = min(35 * len(preview) + 38, 500)
                        st.dataframe(pd.DataFrame(preview), use_container_width=True, height=_h)
                    elif sql_err == "none" and sql_rows == 0:
                        st.caption("Keine Ergebnisse zurückgegeben.")
                # Verify toggle
                v_key_sql = f"{qid}__{model}__sql"
                verified_states = _get_verified_status()
                sql_verified = verified_states.get(v_key_sql, False)
                if st.checkbox("✓ Verified", value=sql_verified, key=f"verify_{v_key_sql}"):
                    if not sql_verified:
                        verified_states[v_key_sql] = True
                        st.session_state["_verified_status"] = verified_states
                        _write_verified_to_runs(qid, model, "sql", True)
                        st.rerun(scope="fragment")
                else:
                    if sql_verified:
                        verified_states[v_key_sql] = False
                        st.session_state["_verified_status"] = verified_states
                        _write_verified_to_runs(qid, model, "sql", False)
                        st.rerun(scope="fragment")
                # Per-pipeline correctness evaluation
                _render_pipeline_score_buttons(scores, score_key, "sql", qid, model)

            # ── Genie Space previews (SQL + result) for this question×model ────────
            if "genie_results" in df_columns and len(mqdf) > 0:
                _genie_row = mqdf["genie_results"].iloc[0]
                if isinstance(_genie_row, dict) and _genie_row:
                    _render_genie_qual_section(_genie_row, qid, model, scores, score_key)

            if "_all_runs" in df_columns and len(mqdf) > 0:
                _runs_data = mqdf["_all_runs"].iloc[0]
                if isinstance(_runs_data, list) and len(_runs_data) > 1:
                    with st.expander(f"📋 Alle {len(_runs_data)} Wiederholungen", expanded=False):
                        for run_item in _runs_data:
                            rep_nr = run_item.get("repetition", "?")
                            st.markdown(f"**Wdh. {rep_nr}**")
                            r_col_ont, r_col_sql = st.columns(2)
                            with r_col_ont:
                                r_ont_q    = run_item.get("ont_generated_query", "") or ""
                                r_ont_prev = run_item.get("ont_result_preview", None)
                                r_ont_err  = run_item.get("ont_error_msg", "") or ""
                                r_ont_rows = run_item.get("ont_row_count", 0) or 0
                                st.html(
                                    f"<div style='font-size:11px;font-weight:700;color:#0057A8;"
                                    f"border-left:3px solid #0057A8;padding:3px 8px;"
                                    f"background:#EEF4FB;border-radius:0 4px 4px 0;margin-bottom:6px;'>"
                                    f"SPARQL &nbsp;·&nbsp; "
                                    f"<span style='font-weight:400;color:#3b71c8;'>{r_ont_rows} Zeilen</span></div>"
                                )
                                if r_ont_err:
                                    with st.expander("⚠️ Fehlermeldung", expanded=False):
                                        st.code(r_ont_err, language=None)
                                if r_ont_q:
                                    with st.popover("SPARQL Query", use_container_width=True):
                                        st.code(r_ont_q, language="sparql")
                                if isinstance(r_ont_prev, list) and len(r_ont_prev) > 0:
                                    _h = min(35 * len(r_ont_prev) + 38, 400)
                                    st.dataframe(pd.DataFrame(r_ont_prev), use_container_width=True, height=_h)
                                elif not r_ont_err and r_ont_rows == 0:
                                    st.caption("Keine Ergebnisse.")
                            with r_col_sql:
                                r_sql_q    = run_item.get("sql_generated_query", "") or ""
                                r_sql_prev = run_item.get("sql_result_preview", None)
                                r_sql_err  = run_item.get("sql_error_msg", "") or ""
                                r_sql_rows = run_item.get("sql_row_count", 0) or 0
                                st.html(
                                    f"<div style='font-size:11px;font-weight:700;color:#7e22ce;"
                                    f"border-left:3px solid #7e22ce;padding:3px 8px;"
                                    f"background:#f5f0ff;border-radius:0 4px 4px 0;margin-bottom:6px;'>"
                                    f"SQL &nbsp;·&nbsp; "
                                    f"<span style='font-weight:400;color:#9c6fd6;'>{r_sql_rows} Zeilen</span></div>"
                                )
                                if r_sql_err:
                                    with st.expander("⚠️ Fehlermeldung", expanded=False):
                                        st.code(r_sql_err, language=None)
                                if r_sql_q:
                                    with st.popover("SQL Query", use_container_width=True):
                                        st.code(r_sql_q, language="sql")
                                if isinstance(r_sql_prev, list) and len(r_sql_prev) > 0:
                                    _h = min(35 * len(r_sql_prev) + 38, 400)
                                    st.dataframe(pd.DataFrame(r_sql_prev), use_container_width=True, height=_h)
                                elif not r_sql_err and r_sql_rows == 0:
                                    st.caption("Keine Ergebnisse.")
                            # ── Genie Space results per repetition ───────────────
                            r_genie_data = run_item.get("genie_results")
                            if isinstance(r_genie_data, dict) and r_genie_data:
                                _g_items = list(r_genie_data.items())
                                _g_rep_cols = st.columns(len(_g_items))
                                _g_clrs = ["#66bb6a", "#ffa726"]
                                for g_idx, (g_label, g_data) in enumerate(_g_items):
                                    g_color = _g_clrs[g_idx % len(_g_clrs)]
                                    g_short = g_label.replace("Genie: ", "Genie Spaces ")
                                    g_sql = g_data.get("generated_sql", "") or ""
                                    g_err = g_data.get("error", "") or ""
                                    g_rows = g_data.get("row_count", 0) or 0
                                    g_prev = g_data.get("result_preview")
                                    with _g_rep_cols[g_idx]:
                                        st.html(
                                            f"<div style='font-size:11px;font-weight:700;color:{g_color};"
                                            f"border-left:3px solid {g_color};padding:3px 8px;"
                                            f"background:rgba(0,0,0,0.04);border-radius:0 4px 4px 0;margin-bottom:6px;'>"
                                            f"{g_short} &nbsp;·&nbsp; "
                                            f"<span style='font-weight:400;opacity:0.75;'>{g_rows} Zeilen</span></div>"
                                        )
                                        if g_err:
                                            with st.expander("⚠️ Fehlermeldung", expanded=False):
                                                st.code(str(g_err), language=None)
                                        if g_sql:
                                            with st.popover("Genie SQL", use_container_width=True):
                                                st.code(g_sql, language="sql")
                                        if isinstance(g_prev, list) and len(g_prev) > 0:
                                            _h = min(35 * len(g_prev) + 38, 400)
                                            st.dataframe(pd.DataFrame(g_prev), use_container_width=True, height=_h)
                                        elif not g_err and g_rows == 0:
                                            st.caption("Keine Ergebnisse.")
                            if run_item is not _runs_data[-1]:
                                st.divider()


        if idx < len(page_questions) - 1:
            st.divider()

    if _n_pages > 1:
        _pcol_prev, _pcol_info, _pcol_next = st.columns([1, 4, 1])
        with _pcol_prev:
            if st.button("← Zurück", key="_qual_page_prev", disabled=(_qual_page == 0)):
                st.session_state["_qual_page"] = _qual_page - 1
                st.rerun(scope="fragment")
        with _pcol_info:
            st.caption(
                f"Seite {_qual_page + 1} / {_n_pages}  ·  "
                f"Fragen {_pg_start + 1}–{_pg_end} von {_n_questions}"
            )
        with _pcol_next:
            if st.button("Weiter →", key="_qual_page_next", disabled=(_qual_page >= _n_pages - 1)):
                st.session_state["_qual_page"] = _qual_page + 1
                st.rerun(scope="fragment")

    if scored_count > 0:
        st.divider()
        st.caption("📋 Bewertungs-Zusammenfassung")
        score_rows = []
        for key, val in sorted(scores.items()):
            if val == "—":
                continue
            parts = key.split("__")
            row: dict = {
                "Frage":     parts[0] if len(parts) > 0 else key,
                "Modell":    parts[1] if len(parts) > 1 else "—",
                "Pipeline":  parts[2] if len(parts) > 2 else "—",
                "Bewertung": val,
            }
            score_rows.append(row)
        if score_rows:
            score_df = pd.DataFrame(score_rows)
            st.dataframe(score_df, use_container_width=True, height=200)
            st.download_button(
                label="📥 Bewertungen als CSV exportieren",
                data=score_df.to_csv(index=False).encode("utf-8"),
                file_name="human_scores.csv",
                mime="text/csv",
            )


def _render_genie_qual_section(
    genie_data: dict, question_id: str = "", model: str = "",
    scores: dict | None = None, score_key: str = "",
) -> None:
    """Render Genie Space SQL + result preview columns in the Qualität tab.

    Called once per question×model pair inside _render_tab_quality, below the
    SPARQL-mit-Ontologie / SQL column pair.  Shows one column per configured
    Genie Space with:
      - Coloured section header (matching GENIE_COLORS)
      - Generated SQL as st.popover (same pattern as SPARQL/SQL columns)
      - Result preview DataFrame (height=150, same as ONT/SQL previews)
      - Error badge when status != COMPLETED
      - Per-pipeline correctness evaluation buttons
      - Row count in header

    Args:
        genie_data: Dict[label, metrics_dict] from ``mqdf['genie_results'].iloc[0]``.
                    Each entry has: wall_ms, status, error, exec_success,
                    generated_sql, row_count, result_preview.
    """
    if not genie_data:
        return

    items = list(genie_data.items())
    genie_cols = st.columns(len(items))
    _g_colors = ["#66bb6a", "#ffa726"]

    for idx, (g_label, g_data) in enumerate(items):
        color       = _g_colors[idx % len(_g_colors)]
        short_label = g_label.replace("Genie: ", "Genie Spaces ")
        status      = g_data.get("status", "UNKNOWN")
        error       = g_data.get("error")
        sql         = g_data.get("generated_sql")
        row_count   = g_data.get("row_count", 0) or 0
        preview     = g_data.get("result_preview")
        is_ok       = g_data.get("exec_success", False)

        with genie_cols[idx]:
            # Coloured section header — same style as col_ont / col_sql headers
            st.html(
                f"<div style='font-size:12px;font-weight:700;color:{color};"
                f"border-left:3px solid {color};padding:5px 10px;"
                f"background:rgba(0,0,0,0.04);border-radius:0 6px 6px 0;margin-bottom:8px;'>"
                f"{short_label} &nbsp;·&nbsp; "
                f"<span style='font-weight:500;opacity:0.75;'>{row_count} Zeilen</span></div>"
            )

            if error:
                st.markdown(
                    f"<span style='background:#fee2e2;color:#991b1b;"
                    f"border-radius:6px;padding:3px 8px;font-size:11px;"
                    f"font-weight:600;'>✗ {status}</span>",
                    unsafe_allow_html=True,
                )
                with st.expander("⚠️ Fehlermeldung", expanded=False):
                    st.code(str(error), language=None)

            if sql:
                with st.popover("Genie SQL", use_container_width=True):
                    st.code(sql, language="sql")

            if isinstance(preview, list) and len(preview) > 0:
                _h = min(35 * len(preview) + 38, 500)
                st.dataframe(
                    pd.DataFrame(preview),
                    use_container_width=True,
                    height=_h,
                    hide_index=True,
                )
            elif is_ok and not preview:
                st.caption("Keine Ergebnis-Vorschau verfügbar.")

            # Verified checkbox — same pattern as ONT / SQL columns
            if question_id and model:
                _safe = g_label.replace(" ", "_").replace(":", "").replace("/", "_")
                v_key = f"{question_id}__{model}__genie__{_safe}"
                verified_states = _get_verified_status()
                g_verified = verified_states.get(v_key, False)
                if st.checkbox("✓ Verified", value=g_verified, key=f"verify_{v_key}"):
                    if not g_verified:
                        verified_states[v_key] = True
                        st.session_state["_verified_status"] = verified_states
                        _write_verified_to_runs(question_id, model, f"genie_{_safe}", True)
                        st.rerun(scope="fragment")
                else:
                    if g_verified:
                        verified_states[v_key] = False
                        st.session_state["_verified_status"] = verified_states
                        _write_verified_to_runs(question_id, model, f"genie_{_safe}", False)
                        st.rerun(scope="fragment")
                # Per-pipeline correctness evaluation for this Genie Space
                if scores is not None and score_key:
                    _render_pipeline_score_buttons(scores, score_key, _safe, question_id, model)


def _render_pipeline_score_buttons(
    scores: dict, score_key: str, pipeline: str, qid: str, model: str
) -> None:
    """Render korrekt / teilweise / falsch buttons for one pipeline.

    Args:
        scores: The shared _human_scores dict.
        score_key: '{qid}__{model}' base key.
        pipeline: 'ont', 'sql', or sanitised Genie label.
        qid: Question ID.
        model: Model name.
    """
    full_key = f"{score_key}__{pipeline}"
    current = scores.get(full_key, "—")
    # Skip the "—" option from buttons — only show actionable choices
    options = [o for o in _PIPELINE_SCORE_OPTIONS if o != "—"]
    btn_cols = st.columns(len(options) + 1)  # +1 for reset
    for i, option in enumerate(options):
        is_selected = current == option
        btn_type = "primary" if is_selected else "secondary"
        with btn_cols[i]:
            if st.button(
                _PIPELINE_SCORE_ICONS[option],
                key=f"ps_{full_key}_{i}",
                type=btn_type,
                use_container_width=True,
            ):
                scores[full_key] = option
                st.session_state["_human_scores"] = scores
                _write_score_to_runs(qid, model, pipeline, option)
                st.rerun(scope="fragment")
    # Reset button
    with btn_cols[-1]:
        if current != "—":
            if st.button("↺", key=f"ps_{full_key}_reset", use_container_width=True):
                scores[full_key] = "—"
                st.session_state["_human_scores"] = scores
                _write_score_to_runs(qid, model, pipeline, "—")
                st.rerun(scope="fragment")


def _write_score_to_runs(question_id: str, model: str, pipeline: str, score: str) -> None:
    """Write a per-pipeline human score into the benchmark_runs session state.

    Args:
        pipeline: 'ont', 'sql', or sanitised Genie label (e.g. 'Genie_MRS_pure_DPs').
        score: 'korrekt', 'teilweise', 'falsch', or '—' (reset).
    """
    runs = st.session_state.get("benchmark_runs", [])
    key = f"{pipeline}_human_score"
    val = score if score != "—" else None
    for run in runs:
        if run.get("question_id") == question_id and run.get("model_name") == model:
            run[key] = val
            # Maintain a combined human_evaluation summary field
            _scores = {}
            for k2, v2 in run.items():
                if k2.endswith("_human_score") and v2 is not None:
                    _scores[k2.replace("_human_score", "")] = v2
            run["human_evaluation"] = "; ".join(f"{p}={s}" for p, s in sorted(_scores.items())) or None
            break
    st.session_state["benchmark_runs"] = runs


def _write_verified_to_runs(question_id: str, model: str, panel: str, verified: bool) -> None:
    """Persist verified status into the benchmark_runs session state.

    Args:
        panel: 'ont' or 'sql'
        verified: True/False
    """
    runs = st.session_state.get("benchmark_runs", [])
    key = f"{panel}_verified"
    for run in runs:
        if run.get("question_id") == question_id and run.get("model_name") == model:
            run[key] = verified
            break
    st.session_state["benchmark_runs"] = runs



def _write_error_type_to_runs(question_id: str, model: str, panel: str, error_type: str) -> None:
    """Persist a manual error_type override into the benchmark_runs session state.

    Args:
        panel: 'ont' or 'sql'
        error_type: one of the _ERR_DISPLAY keys (none, timeout, syntax_error, empty_result, other)
    """
    runs = st.session_state.get("benchmark_runs", [])
    key = f"{panel}_error_type"
    for run in runs:
        if run.get("question_id") == question_id and run.get("model_name") == model:
            run[key] = error_type
            break
    st.session_state["benchmark_runs"] = runs


# =============================================================================
# Tab 5: Rohdaten
# =============================================================================

def _flatten_genie_columns(df: pd.DataFrame, df_columns: set) -> pd.DataFrame:
    """Flatten genie_results dict column into per-space metric columns."""
    if "genie_results" not in df_columns:
        return df
    # Discover Genie labels from first non-empty row
    _labels: list[str] = []
    for val in df["genie_results"]:
        if isinstance(val, dict) and val:
            _labels = sorted(val.keys())
            break
    if not _labels:
        return df
    for g_label in _labels:
        short = g_label.replace("Genie: ", "G:").replace(" ", "_")
        df[f"{short}_total_ms"] = df["genie_results"].apply(
            lambda r, lbl=g_label: round(r.get(lbl, {}).get("wall_ms", 0)) if isinstance(r, dict) else 0
        )
        df[f"{short}_api_start_ms"] = df["genie_results"].apply(
            lambda r, lbl=g_label: round(
                ((r.get(lbl, {}).get("latency_breakdown") or {}).get("start_s", 0)) * 1000
            ) if isinstance(r, dict) else 0
        )
        df[f"{short}_polling_ms"] = df["genie_results"].apply(
            lambda r, lbl=g_label: round(
                ((r.get(lbl, {}).get("latency_breakdown") or {}).get("polling_s", 0)) * 1000
            ) if isinstance(r, dict) else 0
        )
        df[f"{short}_fetch_ms"] = df["genie_results"].apply(
            lambda r, lbl=g_label: round(
                ((r.get(lbl, {}).get("latency_breakdown") or {}).get("fetch_s", 0)) * 1000
            ) if isinstance(r, dict) else 0
        )
        df[f"{short}_overhead_ms"] = df["genie_results"].apply(
            lambda r, lbl=g_label: (
                lambda wall, bd: round(max(0, wall - (bd.get("start_s", 0) + bd.get("polling_s", 0) + bd.get("fetch_s", 0)) * 1000))
            )(
                r.get(lbl, {}).get("wall_ms", 0),
                r.get(lbl, {}).get("latency_breakdown") or {},
            ) if isinstance(r, dict) else 0
        )
        df[f"{short}_status"] = df["genie_results"].apply(
            lambda r, lbl=g_label: r.get(lbl, {}).get("status", "") if isinstance(r, dict) else ""
        )
    return df


def _render_tab_raw(df: pd.DataFrame, df_columns: set) -> None:
    # Flatten Genie results into separate columns
    df = _flatten_genie_columns(df.copy(), df_columns)

    # ── Compute total token columns (prompt + completion) per pipeline ────────
    _ont_prompt_col = "ont_prompt_token_count" if "ont_prompt_token_count" in df.columns else "ont_prompt_tokens"
    _ont_query_col  = "ont_query_token_count"  if "ont_query_token_count"  in df.columns else "ont_query_tokens"
    _sql_prompt_col = "sql_prompt_token_count" if "sql_prompt_token_count" in df.columns else "sql_prompt_tokens"
    _sql_query_col  = "sql_query_token_count"  if "sql_query_token_count"  in df.columns else "sql_query_tokens"

    if _ont_prompt_col in df.columns and _ont_query_col in df.columns:
        df["ont_total_tokens"] = df[_ont_prompt_col].fillna(0).astype(int) + df[_ont_query_col].fillna(0).astype(int)
    if _sql_prompt_col in df.columns and _sql_query_col in df.columns:
        df["sql_total_tokens"] = df[_sql_prompt_col].fillna(0).astype(int) + df[_sql_query_col].fillna(0).astype(int)

    df_columns = set(df.columns.tolist())

    display_cols = [
        "model_name", "question_id", "question_type",
        "join_depth", "triple_path_depth",
        "ont_llm_ms", "ont_exec_ms", "ont_rag_ms", "ont_overhead_ms", "ont_total_ms",
        "sql_llm_ms", "sql_exec_ms", "sql_rag_ms", "sql_overhead_ms", "sql_total_ms",
    ]
    # Dynamically add Genie columns (pattern: G:*_total_ms etc.)
    genie_col_patterns = ["_total_ms", "_api_start_ms", "_polling_ms", "_fetch_ms", "_overhead_ms", "_status"]
    genie_flat_cols = [c for c in df.columns if c.startswith("G:") and any(c.endswith(p) for p in genie_col_patterns)]
    display_cols += sorted(genie_flat_cols)
    display_cols += [
        "ont_prompt_token_count", "ont_query_token_count", "ont_total_tokens",
        "ont_schema_context_size_tokens", "ont_instructions_tokens", "ont_user_message_tokens",
        "ont_graph_rag_applied", "ont_graph_rag_reduction_pct",
        "sql_prompt_token_count", "sql_query_token_count", "sql_total_tokens",
        "sql_schema_context_size_tokens", "sql_instructions_tokens", "sql_user_message_tokens",
        "sql_graph_rag_applied", "sql_graph_rag_reduction_pct",
        "human_evaluation", "ont_human_score", "sql_human_score",
        "ont_verified", "sql_verified",
    ]
    # Dynamically include any *_human_score columns (Genie pipelines)
    _extra_score_cols = [c for c in df.columns if c.endswith("_human_score") and c not in display_cols]
    display_cols += sorted(_extra_score_cols)
    display_cols = [c for c in display_cols if c in df_columns]
    st.dataframe(df[display_cols], use_container_width=True, height=280)

    export_df = df.drop(columns=[c for c in ["ont_row_count", "sql_row_count", "genie_results"] if c in df.columns])
    st.download_button(
        label="📥 Vollständige Ergebnisse als CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="benchmark_results.csv",
        mime="text/csv",
    )

    # ── Individual runs (only for multi-run data) ────────────────────────────
    if "_all_runs" in df_columns and "_num_repetitions" in df_columns:
        _n_reps = int(df["_num_repetitions"].iloc[0]) if len(df) > 0 else 1
        if _n_reps > 1:
            with st.expander(f"📋 Einzellauf-Rohdaten ({_n_reps} Wiederholungen)", expanded=False):
                st.caption(
                    "Jede Wiederholung als separate Zeile. "
                    "Latenzwerte sind Rohwerte pro Lauf — nicht gemittelt."
                )
                indiv_rows: list[dict] = []
                for _, agg_row in df.iterrows():
                    q_id   = agg_row.get("question_id", "—")
                    model  = agg_row.get("model_name", "—")
                    q_type = agg_row.get("question_type", "—")
                    all_runs_data = agg_row.get("_all_runs", [])
                    if not isinstance(all_runs_data, list):
                        continue
                    for run_item in all_runs_data:
                        rep_nr = run_item.get("repetition", "?")
                        if "ont_result" in run_item:
                            ont_r = run_item["ont_result"]
                            sql_r = run_item["sql_result"]
                            o_wall = ont_r.get("total_elapsed", 0.0) * 1000.0
                            o_llm  = ont_r.get("sparql_gen_time", 0.0) * 1000.0
                            o_exec = ont_r.get("sparql_exec_time", 0.0) * 1000.0
                            o_viz  = ont_r.get("viz_time", 0.0) * 1000.0
                            o_oh   = max(0.0, o_wall - o_llm - o_exec - o_viz)
                            _o_rag_meta = (ont_r.get("token_breakdown") or {}).get("graph_rag_meta", {})
                            o_rag  = _o_rag_meta.get("embedding_ms", 0.0) + _o_rag_meta.get("traversal_ms", 0.0)
                            s_wall = sql_r.get("total_elapsed", 0.0) * 1000.0
                            s_llm  = sql_r.get("sql_gen_time", 0.0) * 1000.0
                            s_exec = sql_r.get("sql_exec_time", 0.0) * 1000.0
                            s_oh   = max(0.0, s_wall - s_llm - s_exec)
                            _s_rag_meta = (sql_r.get("token_breakdown") or {}).get("graph_rag_meta", {})
                            s_rag  = _s_rag_meta.get("embedding_ms", 0.0) + _s_rag_meta.get("retrieval_ms", 0.0)
                            o_err  = str(ont_r.get("error", "") or "")
                            s_err  = str(sql_r.get("error", "") or "")
                            # Token usage per run (raw result path)
                            _o_tb = ont_r.get("token_breakdown") or {}
                            o_prompt_tok = _o_tb.get("total_prompt_tokens") or ont_r.get("prompt_token_count") or ont_r.get("prompt_tokens", 0)
                            o_query_tok  = _o_tb.get("output_tokens") or ont_r.get("query_token_count") or ont_r.get("completion_tokens", 0)
                            _s_tb = sql_r.get("token_breakdown") or {}
                            s_prompt_tok = _s_tb.get("total_prompt_tokens") or sql_r.get("prompt_token_count") or sql_r.get("prompt_tokens", 0)
                            s_query_tok  = _s_tb.get("output_tokens") or sql_r.get("query_token_count") or sql_r.get("completion_tokens", 0)
                        else:
                            o_wall = run_item.get("ont_total_ms", 0)
                            o_llm  = run_item.get("ont_llm_ms", 0)
                            o_exec = run_item.get("ont_exec_ms", 0)
                            o_oh   = run_item.get("ont_overhead_ms", 0)
                            o_rag  = run_item.get("ont_rag_ms", 0)
                            s_wall = run_item.get("sql_total_ms", 0)
                            s_llm  = run_item.get("sql_llm_ms", 0)
                            s_exec = run_item.get("sql_exec_ms", 0)
                            s_oh   = run_item.get("sql_overhead_ms", 0)
                            s_rag  = run_item.get("sql_rag_ms", 0)
                            o_err  = str(run_item.get("ont_error_msg", "") or "")
                            s_err  = str(run_item.get("sql_error_msg", "") or "")
                            # Token usage from flattened record
                            o_prompt_tok = run_item.get("ont_prompt_token_count", 0) or run_item.get("ont_prompt_tokens", 0)
                            o_query_tok  = run_item.get("ont_query_token_count", 0) or run_item.get("ont_query_tokens", 0)
                            s_prompt_tok = run_item.get("sql_prompt_token_count", 0) or run_item.get("sql_prompt_tokens", 0)
                            s_query_tok  = run_item.get("sql_query_token_count", 0) or run_item.get("sql_query_tokens", 0)
                        row_data = {
                            "question_id":    q_id,
                            "model_name":     model,
                            "question_type":  q_type,
                            "Wdh.":           rep_nr,
                            "ont_total_ms":   round(o_wall),
                            "ont_llm_ms":     round(o_llm),
                            "ont_exec_ms":    round(o_exec),
                            "ont_rag_ms":     round(o_rag),
                            "ont_overhead_ms": round(o_oh),
                            "ont_prompt_tokens":  o_prompt_tok,
                            "ont_completion_tokens": o_query_tok,
                            "ont_total_tokens":  (o_prompt_tok or 0) + (o_query_tok or 0),
                            "sql_total_ms":   round(s_wall),
                            "sql_llm_ms":     round(s_llm),
                            "sql_exec_ms":    round(s_exec),
                            "sql_rag_ms":     round(s_rag),
                            "sql_overhead_ms": round(s_oh),
                            "sql_prompt_tokens":  s_prompt_tok,
                            "sql_completion_tokens": s_query_tok,
                            "sql_total_tokens":  (s_prompt_tok or 0) + (s_query_tok or 0),
                            "ont_error":      o_err,
                            "sql_error":      s_err,
                        }
                        # Genie Space columns per run
                        genie_data = run_item.get("genie_results", {})
                        if isinstance(genie_data, dict):
                            for g_label, g_res in genie_data.items():
                                short = g_label.replace("Genie: ", "G:").replace(" ", "_")
                                g_wall = (g_res.get("total_elapsed", 0.0) * 1000.0) if isinstance(g_res, dict) else 0
                                g_bd = (g_res.get("latency_breakdown") or {}) if isinstance(g_res, dict) else {}
                                g_api = g_bd.get("start_s", 0.0) * 1000.0
                                g_poll = g_bd.get("polling_s", 0.0) * 1000.0
                                g_fetch = g_bd.get("fetch_s", 0.0) * 1000.0
                                g_oh = max(0.0, g_wall - g_api - g_poll - g_fetch)
                                row_data[f"{short}_total_ms"] = round(g_wall)
                                row_data[f"{short}_api_start_ms"] = round(g_api)
                                row_data[f"{short}_polling_ms"] = round(g_poll)
                                row_data[f"{short}_fetch_ms"] = round(g_fetch)
                                row_data[f"{short}_overhead_ms"] = round(g_oh)
                                row_data[f"{short}_status"] = g_res.get("status", "") if isinstance(g_res, dict) else ""
                        indiv_rows.append(row_data)
                if indiv_rows:
                    indiv_df = pd.DataFrame(indiv_rows)
                    st.dataframe(indiv_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        label="📥 Einzellauf-Rohdaten als CSV",
                        data=indiv_df.to_csv(index=False).encode("utf-8"),
                        file_name="benchmark_individual_runs.csv",
                        mime="text/csv",
                        key="dl_individual_runs",
                    )
