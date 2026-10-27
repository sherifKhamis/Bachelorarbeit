# =============================================================================
# benchmark_summary.py — Einzellauf-Benchmark-Karte (Single-Run Summary)
#
# Rendert eine kompakte Vergleichskarte nach jeder Einzelfrage-Benchmark-
# Ausführung.  Zeigt SPARQL-mit-Ontologie vs. SQL Timing-Breakdown,
# Status-Badges, Graph-RAG-Indikator und Token-Aufschlüsselung.
#
# ─── ARCHITEKTUR ─────────────────────────────────────────────────────────────
#
#   Separiert aus benchmark_dashboard.py (1188 Zeilen) für bessere
#   Wartbarkeit.  Diese Funktion wird NUR von single_benchmark.py
#   importiert und ist logisch unabhängig vom Multi-Run-Dashboard.
#
#   Eingabe: Ein flaches Dict (benchmark_last_run aus session_state)
#   Ausgabe: Streamlit-HTML-Karte mit Timing + optionalem Token-Breakdown
#
#   Supports partial pipeline runs: when pipelines_run.ont=False or
#   pipelines_run.sql=False, the corresponding columns are hidden.
#
# ─── EXPORTS ─────────────────────────────────────────────────────────────────
#
#   • render_benchmark_summary(bm: dict) — Rendert die Summary-Karte
# =============================================================================

from __future__ import annotations

import pandas as pd
import streamlit as st



def render_benchmark_summary(bm: dict) -> None:
    """Rendert eine kompakte Einzellauf-Benchmark-Karte."""
    _render_latency_card(bm)
    _render_repetition_table(bm)
    _render_runs_result_overview(bm)


# =============================================================================
# Interne Render-Funktionen
# =============================================================================

def _ms(val: float) -> str:
    """Formatiert Millisekunden als lesbare Zeichenkette."""
    return f"{val:,.0f} ms"


def _err_badge(error) -> str:
    """Erzeugt ein HTML-Status-Badge (✓ ok oder ✗ error)."""
    if error:
        return "<span style='color:#ff6b6b;font-size:11px;'>✗ error</span>"
    return "<span style='color:#69db7c;font-size:11px;'>✓ ok</span>"


def _render_latency_card(bm: dict) -> None:
    """Rendert die HTML-Latenzkarte mit Phasen-Breakdown (MB Corporate Design).

    Conditionally shows/hides columns based on pipelines_run dict.
    """
    model_label = bm.get("model_name", "—")
    num_reps = bm.get("_num_repetitions", 1)
    reps_badge = (
        f" &nbsp;·&nbsp; <code style='color:#B8D4F4;font-weight:700;"
        f"font-family:monospace;'>Ø {num_reps} Wdh.</code>"
        if num_reps > 1 else ""
    )

    # ── Determine which pipelines ran ──────────────────────────────────────
    pipelines_run = bm.get("pipelines_run", {"ont": True, "sql": True, "genie": []})
    show_ont = pipelines_run.get("ont", True)
    show_sql = pipelines_run.get("sql", True)

    # Graph-RAG badges
    _ont_pct = bm.get('graph_rag_reduction_pct', 0)
    if bm.get("graph_rag_applied"):
        if _ont_pct >= 0:
            rag_badge = f"<span style='color:#69db7c;font-size:11px;'>✓ −{_ont_pct:.0f}%</span>"
        else:
            rag_badge = f"<span style='color:#ffa94d;font-size:11px;'>✓ +{abs(_ont_pct):.0f}%</span>"
    else:
        rag_badge = "<span style='color:rgba(255,255,255,0.4);font-size:11px;'>○ Vollkontext</span>"

    _sql_pct = bm.get('sql_graph_rag_reduction_pct', 0)
    if bm.get("sql_graph_rag_applied"):
        if _sql_pct >= 0:
            sql_rag_badge = f"<span style='color:#69db7c;font-size:11px;'>✓ −{_sql_pct:.0f}%</span>"
        else:
            sql_rag_badge = f"<span style='color:#ffa94d;font-size:11px;'>✓ +{abs(_sql_pct):.0f}%</span>"
    else:
        sql_rag_badge = "<span style='color:rgba(255,255,255,0.4);font-size:11px;'>○ Vollkontext</span>"

    # ── Build Genie columns dynamically (only those that ran) ──────────────
    genie_data = bm.get("genie_results", {})
    genie_colors = ["#66bb6a", "#ffa726"]
    genie_header_cells = ""
    genie_total_cells = ""
    genie_status_cells = ""
    genie_empty_cells = ""
    for i, (g_label, g_data) in enumerate(genie_data.items()):
        color = genie_colors[i % len(genie_colors)]
        short_label = g_label.replace("Genie: ", "")
        genie_header_cells += (
            f'<td style="padding:4px 8px 8px;text-align:center;color:{color};'
            f'font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;">'
            f'{short_label}</td>'
        )
        wall = g_data.get("wall_ms", 0)
        _bd = g_data.get("latency_breakdown") or {}
        _named_ms = ((_bd.get("start_s") or 0) + (_bd.get("polling_s") or 0)) * 1000
        _ohne_sub = (
            f'<div style="font-size:10px;font-weight:normal;margin-top:2px;'
            f'color:rgba(255,255,255,0.5);font-family:sans-serif;">'
            f'{_ms(_named_ms)} ohne Overhead</div>'
            if 0 < _named_ms < wall else ""
        )
        genie_total_cells += (
            f'<td style="padding:8px 8px 4px;text-align:center;font-weight:700;color:{color};'
            f'font-size:16px;font-family:monospace;">'
            f'{_ms(wall)}{_ohne_sub}</td>'
        )
        genie_status_cells += (
            f'<td style="padding:4px 8px;text-align:center;">{_err_badge(g_data.get("error"))}</td>'
        )
        genie_empty_cells += '<td style="padding:4px 8px;text-align:center;opacity:0.35;">—</td>'

    # ── Phase breakdown rows for Genie (API-Start / Polling / Overhead) ────
    _has_bd = bool(genie_data) and any(
        g_data.get("latency_breakdown") for _, g_data in genie_data.items()
    )
    _genie_phase_rows = ""

    if _has_bd:
        # Empty cells for ONT + SQL columns (only for those shown)
        _e2_parts = []
        if show_ont:
            _e2_parts.append('<td style="padding:4px 8px;text-align:center;opacity:0.35;">—</td>')
        if show_sql:
            _e2_parts.append('<td style="padding:4px 8px;text-align:center;opacity:0.35;">—</td>')
        _e2 = "".join(_e2_parts)

        def _bd_cell(val_s, c):
            return (
                f'<td style="padding:4px 8px;text-align:center;color:{c};">'
                f'{_ms((val_s or 0) * 1000)}</td>'
            )

        for key, label in [
            ("start_s",   "API-Start"),
            ("polling_s", "Polling"),
        ]:
            cells = "".join(
                _bd_cell(
                    (g_data.get("latency_breakdown") or {}).get(key),
                    genie_colors[i % len(genie_colors)],
                )
                for i, (_, g_data) in enumerate(genie_data.items())
            )
            _genie_phase_rows += (
                f'<tr><td style="padding:4px 8px 4px 12px;font-size:11px;'
                f'color:rgba(255,255,255,0.4);">{label}</td>'
                f"{_e2}{cells}</tr>\n    "
            )

        # Overhead row
        oh_cells = ""
        for i, (_, g_data) in enumerate(genie_data.items()):
            color = genie_colors[i % len(genie_colors)]
            wall = g_data.get("wall_ms", 0)
            bd = g_data.get("latency_breakdown") or {}
            named_ms = ((bd.get("start_s") or 0) + (bd.get("polling_s") or 0)) * 1000.0
            oh_ms = max(0.0, wall - named_ms)
            oh_cells += (
                f'<td style="padding:4px 8px;text-align:center;color:{color};opacity:0.5;">'
                f'{_ms(oh_ms)}</td>'
            )
        _genie_phase_rows += (
            '<tr><td style="padding:4px 8px 4px 12px;font-size:11px;color:rgba(255,255,255,0.3);">'
            'Overhead'
            '<span title="Overhead = Gesamtzeit \u2212 (API-Start + Polling)&#10;&#10;'
            '\u2022 SDK-Initialisierung (WorkspaceClient)&#10;'
            '\u2022 Ergebnis-Abruf (query-result GET)&#10;'
            '\u2022 JSON-Parsing &amp; Serialisierung&#10;'
            '\u2022 Thread-Synchronisation"'
            ' style="cursor:help;font-size:9px;opacity:0.7;margin-left:3px;">&#9432;</span>'
            '</td>'
            f"{_e2}{oh_cells}</tr>\n    "
        )

    # Shared label style
    _lbl = "font-size:11px;color:rgba(255,255,255,0.45);text-transform:uppercase;letter-spacing:0.5px;"
    # Pre-compute conditional RAG cells
    _ont_rag_cell = '<span style="opacity:0.3;">—</span>' if bm.get('ont_rag_ms', 0) == 0 else _ms(bm['ont_rag_ms'])
    _sql_rag_cell = '<span style="opacity:0.3;">—</span>' if bm.get('sql_rag_ms', 0) == 0 else _ms(bm['sql_rag_ms'])

    # ── Build ONT column cells (conditionally) ─────────────────────────────
    ont_header = (
        '<td style="padding:4px 8px 8px;text-align:center;color:#B8D4F4;font-weight:700;'
        'font-size:11px;text-transform:uppercase;letter-spacing:0.5px;">SPARQL · Ontologie</td>'
        if show_ont else ""
    )
    sql_header = (
        '<td style="padding:4px 8px 8px;text-align:center;color:#ce93d8;font-weight:700;'
        'font-size:11px;text-transform:uppercase;letter-spacing:0.5px;">SQL</td>'
        if show_sql else ""
    )

    def _ont_cell(content: str, extra_style: str = "") -> str:
        return f'<td style="padding:4px 8px;text-align:center;{extra_style}">{content}</td>' if show_ont else ""

    def _sql_cell(content: str, extra_style: str = "") -> str:
        return f'<td style="padding:4px 8px;text-align:center;{extra_style}">{content}</td>' if show_sql else ""

    # ── Build the main HTML table rows ─────────────────────────────────────
    ont_total_html = ""
    if show_ont:
        ont_total_html = (
            f'<td style="padding:8px 8px 4px;text-align:center;font-weight:700;color:#B8D4F4;'
            f'font-size:16px;font-family:monospace;">'
            f'{_ms(bm["ont_wall_ms"])}'
            f'<div style="font-size:10px;font-weight:normal;margin-top:2px;color:rgba(255,255,255,0.5);'
            f'font-family:sans-serif;">{_ms(bm["ont_wall_ms"] - bm.get("ont_overhead_ms", 0))} ohne Overhead</div>'
            f'</td>'
        )
    sql_total_html = ""
    if show_sql:
        sql_total_html = (
            f'<td style="padding:8px 8px 4px;text-align:center;font-weight:700;color:#ce93d8;'
            f'font-size:16px;font-family:monospace;">'
            f'{_ms(bm["sql_wall_ms"])}'
            f'<div style="font-size:10px;font-weight:normal;margin-top:2px;color:rgba(255,255,255,0.5);'
            f'font-family:sans-serif;">{_ms(bm["sql_wall_ms"] - bm.get("sql_overhead_ms", 0))} ohne Overhead</div>'
            f'</td>'
        )

    st.html(f"""
<div style="
    background: linear-gradient(135deg, #003A7D 0%, #0057A8 50%, #003A7D 100%);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 16px;
    padding: 20px 28px;
    margin: 8px 0 16px 0;
    box-shadow: 0 8px 32px rgba(0,58,125,0.35);
    color: #ffffff;
">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
    <div style="font-size:28px;">🔬</div>
    <div>
      <div style="font-size:15px;font-weight:700;color:#ffffff;letter-spacing:-0.3px;">
        Benchmark &nbsp;·&nbsp; Run <code style="color:#B8D4F4;font-family:monospace;">{bm['run_id']}</code>{reps_badge}
      </div>
      <div style="font-size:11px;color:rgba(255,255,255,0.5);margin-top:3px;letter-spacing:0.3px;">
        Modell: <code style="color:#B8D4F4;font-family:monospace;">{model_label}</code>
      </div>
    </div>
  </div>
  <div style="background:rgba(255,255,255,0.08);border-radius:10px;padding:12px 16px;border:1px solid rgba(255,255,255,0.15);">
    <table style="width:100%;border-collapse:collapse;color:#ffffff;font-size:13px;">
      <tr>
        <td style="width:22%;padding:4px 8px 8px 0;"></td>
        {ont_header}
        {sql_header}
        {genie_header_cells}
      </tr>
      <tr>
        <td style="padding:4px 8px 4px 0;{_lbl}">LLM-Generierung</td>
        {_ont_cell(_ms(bm['ont_llm_ms']))}
        {_sql_cell(_ms(bm['sql_llm_ms']))}
        {genie_empty_cells}
      </tr>
      <tr>
        <td style="padding:4px 8px 4px 0;{_lbl}">Query-Ausführung</td>
        {_ont_cell(_ms(bm['ont_exec_ms']))}
        {_sql_cell(_ms(bm['sql_exec_ms']))}
        {genie_empty_cells}
      </tr>
      <tr>
        <td style="padding:4px 8px 4px 0;{_lbl}">Graph-Visualisierung</td>
        {_ont_cell(_ms(bm['ont_viz_ms']))}
        {_sql_cell('<span style="opacity:0.35;">—</span>')}
        {genie_empty_cells}
      </tr>
      <tr>
        <td style="padding:4px 8px 4px 0;{_lbl}">
          Graph-RAG-Latenz
          <span title="Carina-Embedding-Lookup + Oxigraph/ColumnGraph-Traversal&#10;(Teilmenge des Overheads — nur wenn Graph RAG aktiv)"
                style="cursor:help;font-size:9px;opacity:0.7;margin-left:3px;">&#9432;</span>
        </td>
        {_ont_cell(_ont_rag_cell)}
        {_sql_cell(_sql_rag_cell)}
        {genie_empty_cells}
      </tr>
      <tr>
        <td style="padding:4px 8px 4px 0;font-size:11px;color:rgba(255,255,255,0.3);text-transform:uppercase;letter-spacing:0.5px;">
          Overhead
          <span title="Overhead = Gesamtzeit − (LLM + Ausführung + Visualisierung + Graph RAG)&#10;&#10;Enthält: Thread-Setup + HTTP-Auth + sonstige Initialisierung"
                style="cursor:help;font-size:9px;opacity:0.7;margin-left:3px;">&#9432;</span>
        </td>
        {_ont_cell(_ms(bm.get('ont_overhead_ms', 0)), "opacity:0.5;")}
        {_sql_cell(_ms(bm.get('sql_overhead_ms', 0)), "opacity:0.5;")}
        {genie_empty_cells}
      </tr>
      {_genie_phase_rows}
      <tr style="border-top:1px solid rgba(255,255,255,0.15);">
        <td style="padding:8px 8px 4px 0;{_lbl}">Gesamtzeit</td>
        {ont_total_html}
        {sql_total_html}
        {genie_total_cells}
      </tr>
      <tr>
        <td style="padding:4px 8px 4px 0;{_lbl}">Status</td>
        {_ont_cell(_err_badge(bm.get('ont_error')))}
        {_sql_cell(_err_badge(bm.get('sql_error')))}
        {genie_status_cells}
      </tr>
      <tr>
        <td style="padding:4px 8px 4px 0;{_lbl}">Graph-RAG</td>
        {_ont_cell(rag_badge)}
        {_sql_cell(sql_rag_badge)}
        {genie_empty_cells}
      </tr>
    </table>
  </div>
</div>
""")


def _render_repetition_table(bm: dict) -> None:
    """Rendert eine ausklappbare Übersichtstabelle pro Wiederholung."""
    num_reps = bm.get("_num_repetitions", 1)
    if num_reps <= 1:
        return

    timing_records = bm.get("_timing_records", [])
    all_runs = bm.get("_all_runs", [])

    if not timing_records and not all_runs:
        return

    pipelines_run = bm.get("pipelines_run", {"ont": True, "sql": True, "genie": []})
    show_ont = pipelines_run.get("ont", True)
    show_sql = pipelines_run.get("sql", True)

    with st.expander(f"📊 Einzellauf-Details ({num_reps} Wiederholungen)", expanded=False):
        st.caption(
            "Zeigt die Latenzwerte jeder einzelnen Wiederholung. "
            "Die Hauptkarte oben zeigt den **Durchschnitt**."
        )

        detail_rows = []
        for i in range(num_reps):
            row: dict = {"Wdh.": i + 1}

            if i < len(timing_records):
                tr = timing_records[i]
                if show_ont:
                    row["ONT Gesamt"] = f"{tr.get('ont_wall_ms', 0):.0f} ms"
                    row["ONT LLM"] = f"{tr.get('ont_llm_ms', 0):.0f} ms"
                    row["ONT Exec"] = f"{tr.get('ont_exec_ms', 0):.0f} ms"
                    row["ONT Graph RAG"] = f"{tr.get('ont_rag_ms', 0):.0f} ms"
                    row["ONT Overhead"] = f"{tr.get('ont_overhead_ms', 0):.0f} ms"
                if show_sql:
                    row["SQL Gesamt"] = f"{tr.get('sql_wall_ms', 0):.0f} ms"
                    row["SQL LLM"] = f"{tr.get('sql_llm_ms', 0):.0f} ms"
                    row["SQL Exec"] = f"{tr.get('sql_exec_ms', 0):.0f} ms"
                    row["SQL Graph RAG"] = f"{tr.get('sql_rag_ms', 0):.0f} ms"
                    row["SQL Overhead"] = f"{tr.get('sql_overhead_ms', 0):.0f} ms"

            if i < len(all_runs):
                run_item = all_runs[i]
                if show_ont:
                    ont_r = run_item.get("ont_result") or {}
                    row["ONT"] = "✗" if ont_r.get("error") else "✓"
                if show_sql:
                    sql_r = run_item.get("sql_result") or {}
                    row["SQL"] = "✗" if sql_r.get("error") else "✓"

            detail_rows.append(row)

        if detail_rows:
            st.dataframe(
                pd.DataFrame(detail_rows),
                use_container_width=True,
                hide_index=True,
            )


def _render_runs_result_overview(bm: dict) -> None:
    """Rendert eine ausklappbare Läufe-Ergebnis Übersicht pro Wiederholung.

    Zeigt für jeden einzelnen Lauf die generierten Queries und Ergebnis-
    Vorschauen (analog zur Qualitätsübersicht im Corpus-Benchmark).
    Nur sichtbar wenn num_repetitions > 1.
    """
    num_reps = bm.get("_num_repetitions", 1)
    if num_reps <= 1:
        return

    all_runs = bm.get("_all_runs", [])
    if not all_runs or len(all_runs) <= 1:
        return

    pipelines_run = bm.get("pipelines_run", {"ont": True, "sql": True, "genie": []})
    show_ont = pipelines_run.get("ont", True)
    show_sql = pipelines_run.get("sql", True)

    with st.expander(f"📋 Läufe-Ergebnis Übersicht ({len(all_runs)} Wiederholungen)", expanded=False):
        st.caption(
            "Vergleiche die generierten Queries und Ergebnisse jeder einzelnen "
            "Wiederholung. Die Hauptkarte oben zeigt den **Durchschnitt**."
        )

        for run_item in all_runs:
            rep_nr = run_item.get("repetition", "?")
            ont_r = run_item.get("ont_result") or {}
            sql_r = run_item.get("sql_result") or {}
            genie_results = run_item.get("genie_results", {})

            st.markdown(f"**Wdh. {rep_nr}**")

            # ── ONT + SQL side-by-side (only show those that ran) ─────────────
            main_cols = []
            if show_ont:
                main_cols.append("ont")
            if show_sql:
                main_cols.append("sql")

            if main_cols:
                cols = st.columns(len(main_cols))
                col_idx = 0

                if show_ont:
                    with cols[col_idx]:
                        ont_err = ont_r.get("error", "") or ""
                        ont_query = ont_r.get("generated_query", "") or ""
                        ont_rows = 0
                        ont_preview_df = None
                        if "results_df" in ont_r and ont_r["results_df"] is not None:
                            try:
                                ont_preview_df = ont_r["results_df"]
                                ont_rows = len(ont_preview_df)
                            except Exception:
                                pass
                        elif "triples_df" in ont_r and ont_r["triples_df"] is not None:
                            try:
                                ont_preview_df = ont_r["triples_df"]
                                ont_rows = ont_r.get("num_triples", len(ont_preview_df))
                            except Exception:
                                pass

                        st.markdown(
                            f"<div style='font-size:11px;font-weight:700;color:#0369a1;"
                            f"margin-bottom:4px;'>SPARQL mit Ontologie · {ont_rows} Zeilen</div>",
                            unsafe_allow_html=True,
                        )
                        if ont_err:
                            with st.expander("⚠️ Fehlermeldung", expanded=False):
                                st.code(ont_err, language=None)
                        if ont_query:
                            with st.popover("SPARQL Query", use_container_width=True):
                                st.code(ont_query, language="sparql")
                        if ont_preview_df is not None and len(ont_preview_df) > 0:
                            _h = min(35 * len(ont_preview_df) + 38, 400)
                            st.dataframe(ont_preview_df, use_container_width=True, height=_h)
                        elif not ont_err and ont_rows == 0:
                            st.caption("Keine Ergebnisse.")
                    col_idx += 1

                if show_sql:
                    with cols[col_idx]:
                        sql_err = sql_r.get("error", "") or ""
                        sql_query = sql_r.get("generated_sql", "") or ""
                        sql_rows = 0
                        sql_preview_df = None
                        if "results_df" in sql_r and sql_r["results_df"] is not None:
                            try:
                                sql_preview_df = sql_r["results_df"]
                                sql_rows = len(sql_preview_df)
                            except Exception:
                                pass

                        st.markdown(
                            f"<div style='font-size:11px;font-weight:700;color:#7e22ce;"
                            f"margin-bottom:4px;'>SQL · {sql_rows} Zeilen</div>",
                            unsafe_allow_html=True,
                        )
                        if sql_err:
                            with st.expander("⚠️ Fehlermeldung", expanded=False):
                                st.code(sql_err, language=None)
                        if sql_query:
                            with st.popover("SQL Query", use_container_width=True):
                                st.code(sql_query, language="sql")
                        if sql_preview_df is not None and len(sql_preview_df) > 0:
                            _h = min(35 * len(sql_preview_df) + 38, 400)
                            st.dataframe(sql_preview_df, use_container_width=True, height=_h)
                        elif not sql_err and sql_rows == 0:
                            st.caption("Keine Ergebnisse.")

            # ── Genie Spaces (if present) ─────────────────────────────────────
            if genie_results:
                genie_items = list(genie_results.items())
                genie_cols = st.columns(len(genie_items))
                _g_colors = ["#66bb6a", "#ffa726"]

                for g_idx, (g_label, g_data) in enumerate(genie_items):
                    color = _g_colors[g_idx % len(_g_colors)]
                    short_label = g_label.replace("Genie: ", "Genie Spaces ")
                    g_err = g_data.get("error", "") or ""
                    g_sql = g_data.get("generated_sql", "") or ""
                    g_row_count = g_data.get("row_count", 0) or 0
                    g_preview = g_data.get("result_preview")

                    with genie_cols[g_idx]:
                        st.markdown(
                            f"<div style='font-size:11px;font-weight:700;color:{color};"
                            f"margin-bottom:4px;'>{short_label} · {g_row_count} Zeilen</div>",
                            unsafe_allow_html=True,
                        )
                        if g_err:
                            with st.expander("⚠️ Fehlermeldung", expanded=False):
                                st.code(g_err, language=None)
                        if g_sql:
                            with st.popover("SQL Query", use_container_width=True):
                                st.code(g_sql, language="sql")
                        if g_preview:
                            try:
                                _pdf = pd.DataFrame(g_preview)
                                _h = min(35 * len(_pdf) + 38, 400)
                                st.dataframe(_pdf, use_container_width=True, height=_h)
                            except Exception:
                                st.caption(f"{g_row_count} Zeilen (Vorschau nicht darstellbar).")
                        elif not g_err and g_row_count == 0:
                            st.caption("Keine Ergebnisse.")

            st.divider()
