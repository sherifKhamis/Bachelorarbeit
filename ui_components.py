# =============================================================================
# ui_components.py — Shared Streamlit UI components
#
# Reusable visual building blocks used by more than one panel.
# All functions in this module call st.* directly and therefore MUST NOT be
# imported from daemon threads or background tasks.
#
# Design constraint:
#   This module intentionally has no application-level imports (no config,
#   no sql_panel, no ontology_panel).  It depends only on the standard library
#   and on utils.py (which is side-effect-free and thread-safe).
#
# Exported constants (accent / background colours per panel):
#   SQL_ACCENT_COLOR, SQL_BG_FROM, SQL_BG_TO  — dark-purple theme (SQL panel)
#   ONT_ACCENT_COLOR, ONT_BG_FROM, ONT_BG_TO  — dark-blue  theme (Ontology panel)
#
# Exported functions:
#   • render_large_timer(elapsed, overhead_ms, llm_ms, exec_ms, viz_ms,
#                        accent_color, bg_from, bg_to)
#       Renders a large, styled execution-time card showing total elapsed time
#       with a per-phase breakdown and an "ohne Overhead" net time.
#       Used by both ontology_panel.py and sql_panel.py.
#
#   • render_genie_timer(elapsed, breakdown, accent_color, bg_from, bg_to)
#       Renders a Genie Space latency card in the same style as render_large_timer,
#       with Genie-specific phase labels (API-Start · Polling · Overhead).
#       Used by genie_panel.py.
#
#   • render_token_breakdown(token_breakdown, expander_label, expanded,
#                             use_expander)
#       Unified token breakdown widget for SPARQL and SQL panels.
#       Shows: Input Tokens (gesamt) | Output Tokens (top row), then
#       Schema Context | Instruktionen | Nutzerfrage (detail row), plus
#       an additivity check caption, schema-ratio, and Graph-RAG badge.
#       Set use_expander=False to render inline (e.g. in benchmark summary).
#       Used by ontology_panel.py, sql_panel.py, and benchmark_dashboard.py.
# =============================================================================

import streamlit as st

from utils import fmt_ms


# ---------------------------------------------------------------------------
# Per-panel theme constants
#
# Pass these to render_large_timer() to keep each panel's visual identity
# consistent without hard-coding colour strings at call sites.
# ---------------------------------------------------------------------------

# SQL panel — Mercedes-Benz BKK Light: Corporate Blue auf weißem Hintergrund
SQL_ACCENT_COLOR = "#003A7D"    # MB Corporate Deep Blue
SQL_BG_FROM      = "#FFFFFF"    # Weiß
SQL_BG_TO        = "#E8F2FB"    # MB BKK Hellblau

# Ontology panel — MB BKK Mid Blue
ONT_ACCENT_COLOR = "#0057A8"    # MB Action Blue
ONT_BG_FROM      = "#EEF4FB"    # MB BKK sehr helles Blau
ONT_BG_TO        = "#DCE9F7"    # MB BKK blassblau

# Genie Space panel — MB BKK weiß/hellblau
GENIE_BG_FROM = "#FFFFFF"       # Weiß
GENIE_BG_TO   = "#F2F7FC"       # MB BKK sehr helles Blau-Grau


# ---------------------------------------------------------------------------
# Execution-time display card
# ---------------------------------------------------------------------------

def render_large_timer(
    elapsed: float,
    overhead_ms: float = 0.0,
    llm_ms: float = 0.0,
    exec_ms: float = 0.0,
    viz_ms: float = 0.0,
    rag_ms: float = 0.0,
    accent_color: str = ONT_ACCENT_COLOR,
    bg_from: str = ONT_BG_FROM,
    bg_to: str = ONT_BG_TO,
) -> None:
    """Render a large, styled execution-time card with phase breakdown.

    Displays the total elapsed time prominently with a colour-coded phase
    breakdown (LLM · Exec · [Viz] · Graph RAG · Overhead) and, when overhead
    is non-zero, an "ohne Overhead" net time separated by a thin horizontal
    divider.

    When ``rag_ms`` is provided, the Overhead phase is split into two:
      - "Graph RAG Xms"  — embedding + SPARQL traversal / SQL live retrieval
      - "Overhead Yms"   — residual (thread setup, HTTP auth, etc.)
    The standard "Overhead" label is omitted when rag_ms covers all of it.

    The function is shared by both panels.  Pass the module-level theme
    constants for the panel you are rendering to maintain consistent styling:

        # SQL panel (dark-purple):
        render_large_timer(
            elapsed, ...,
            accent_color=SQL_ACCENT_COLOR,
            bg_from=SQL_BG_FROM,
            bg_to=SQL_BG_TO,
        )

        # Ontology panel (dark-blue, default colours):
        render_large_timer(elapsed, ...)

    Args:
        elapsed:      Total wall-clock seconds (LLM + exec + [viz] + overhead).
        overhead_ms:  Total thread/HTTP/RAG overhead in milliseconds.
                      Subtracted from ``elapsed`` when computing the
                      "ohne Overhead" net time below the divider.
        llm_ms:       LLM generation phase duration in milliseconds.
        exec_ms:      Query execution duration in milliseconds (Databricks SQL
                      warehouse or Oxigraph SPARQL runtime).
        viz_ms:       Visualisation duration in milliseconds (CONSTRUCT →
                      pyvis graph build).  Pass ``0.0`` to omit from breakdown.
        rag_ms:       Graph RAG sub-phase duration in milliseconds
                      (Carina embedding + SPARQL traversal or SQL retrieval).
                      When > 0, shown as a named phase; the residual
                      (overhead_ms − rag_ms) is shown as "Overhead".
        accent_color: CSS colour string for the large time number and the
                      breakdown label text.
        bg_from:      CSS colour for the left / top gradient stop.
        bg_to:        CSS colour for the right / bottom gradient stop.
    """
    # ── Format total elapsed time ──────────────────────────────────────────
    mins  = int(elapsed // 60)
    secs  = elapsed % 60
    label = f"{mins}m {secs:.2f}s" if mins > 0 else f"{secs:.2f}s"

    # ── Phase breakdown row (non-zero phases joined by " · ") ──────────────
    parts: list[str] = []
    if llm_ms > 0:
        parts.append(f"LLM {fmt_ms(llm_ms)}")
    if exec_ms > 0:
        parts.append(f"Exec {fmt_ms(exec_ms)}")
    if viz_ms > 0:
        parts.append(f"Viz {fmt_ms(viz_ms)}")
    # Show Graph RAG as an explicit named phase when timing is available.
    if rag_ms > 0:
        parts.append(f"Graph RAG {fmt_ms(rag_ms)}")
    # Residual overhead = total overhead minus the Graph RAG portion.
    residual_overhead_ms = max(0.0, overhead_ms - rag_ms) if rag_ms > 0 else overhead_ms
    if residual_overhead_ms > 0:
        parts.append(f"Overhead {fmt_ms(residual_overhead_ms)}")
    elif overhead_ms > 0 and rag_ms == 0:
        parts.append(f"Overhead {fmt_ms(overhead_ms)}")

    breakdown_html = ""
    if parts:
        breakdown_html = (
            f"<div style=\"font-size:13px; opacity:0.55; margin-top:10px;"
            f"font-family:'Courier New',monospace; color:{accent_color}; letter-spacing:0.3px;\">"
            + " · ".join(parts) + "</div>"
        )

    # ── "Ohne Overhead" net time (always shown for consistency) ─────────────
    wo_label = ""
    if overhead_ms >= 0:
        wo       = elapsed - overhead_ms / 1000.0
        wo_mins  = int(wo // 60)
        wo_secs  = wo % 60
        wo_label = f"{wo_mins}m {wo_secs:.2f}s" if wo_mins > 0 else f"{wo_secs:.2f}s"

    ohne_section = ""
    if wo_label:
        ohne_section = (
            '<p style="margin:10px 0 0 0; padding-top:8px;'
            ' border-top:1px solid rgba(0,74,151,0.15);">'
            '<span style="font-size:22px; font-weight:700;'
            " font-family:'Courier New',monospace; color:#1A1A1A;\">"
            f'{wo_label}</span>'
            '<span style="font-size:12px; font-weight:500; color:rgba(0,0,0,0.5);'
            ' margin-left:8px; letter-spacing:1px; text-transform:uppercase;">'
            'ohne Overhead</span></p>'
        )

    st.markdown(
        f"""<div style="
    background: linear-gradient(135deg, {bg_from} 0%, {bg_to} 100%);
    color: #1A1A1A; border-radius: 16px; padding: 14px 24px;
    text-align: center; margin: 4px 0 16px 0;
    border: 1px solid rgba(0,74,151,0.2);
    box-shadow: 0 4px 15px rgba(0,74,151,0.1);
">
    <div style="font-size:11px; color:rgba(0,0,0,0.5); letter-spacing:3px;
                text-transform:uppercase; margin-bottom:4px;">
        Letzte Ausführung · Gesamtzeit
    </div>
    <div style="font-size:52px; font-weight:700;
                font-family:'Courier New',monospace;
                line-height:1.1; color:{accent_color};">
        {label}
    </div>
    {breakdown_html}
    {ohne_section}
</div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Unified token breakdown widget
# ---------------------------------------------------------------------------

def render_token_breakdown(
    token_breakdown: dict,
    expander_label: str = "🔢 Token-Aufschlüsselung",
    expanded: bool = False,
    use_expander: bool = True,
) -> None:
    """Render a unified, detailed token breakdown for SPARQL and SQL panels.

    Displays prompt composition in a consistent two-row layout:

        Row 1 — totals:
            Input Tokens (gesamt)  |  Output Tokens

        Row 2 — input composition:
            Schema Context  |  Instruktionen  |  Nutzerfrage

        Footer:
            ✓ additivity check · Schema-Anteil: X%
            [Graph-RAG badge — SPARQL only, when key present in dict]

    Both panels share the same ``token_breakdown`` dict structure
    (``schema_context_tokens``, ``instructions_tokens``, ``user_message_tokens``,
    ``total_prompt_tokens``, ``output_tokens``).  The Graph-RAG fields
    (``graph_rag_applied``, ``graph_rag_reduction_pct``) are present only
    for the SPARQL panel and are silently ignored for SQL.

    Args:
        token_breakdown: Dict as returned by ``generate_sparql()`` /
                         ``generate_sql_query()`` under the ``token_breakdown``
                         key.  If empty or None, the widget is not rendered.
        expander_label:  Label for the ``st.expander`` header.
                         Ignored when ``use_expander=False``.
        expanded:        Whether the expander starts open.
                         Ignored when ``use_expander=False``.
        use_expander:    ``True``  → wrap content in ``st.expander`` (default,
                                   used by interactive panel display functions).
                         ``False`` → render content inline without a wrapper
                                   (used inside columns in benchmark_dashboard).
    """
    if not token_breakdown:
        return

    schema_tc = token_breakdown.get("schema_context_tokens", 0)
    instr_tc  = token_breakdown.get("instructions_tokens", 0)
    user_tc   = token_breakdown.get("user_message_tokens", 0)
    total_tc  = token_breakdown.get("total_prompt_tokens", 0)
    output_tc = token_breakdown.get("output_tokens", 0)

    def _body() -> None:
        # ── Row 1: Input + Output totals ──────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Input Tokens (gesamt)",
                f"{total_tc:,}",
                help="Summe: Schema Context + Instruktionen + Nutzerfrage",
            )
        with col2:
            st.metric(
                "Output Tokens",
                f"{output_tc:,}",
                help="Die generierte Query — lokale tiktoken-Schätzung (cl100k_base). Bei Mehr-Kandidaten-Pipelines: Summe aller Kandidaten.",
            )

        # ── Row 2: Input composition ───────────────────────────────────────
        st.caption("**Aufschlüsselung Input:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Schema Context",
                f"{schema_tc:,}",
                help=(
                    "ONT: Klassen · Properties · Instanzproben aus der Ontologie.\n"
                    "SQL: Regelbasiert gefilterter Tabellen-Subset (3–7 von 7 MRS-Tabellen "
                    "je nach Frage-Typ) + JOIN-Pfade + Terminologie + "
                    "Pipeline-Docs (CORE ± CASCADE)."
                ),
            )
        with c2:
            st.metric(
                "Instruktionen",
                f"{instr_tc:,}",
                help=(
                    "ONT: Prompt-Regeln + SPARQL-Beispiele + Security Suffix.\n"
                    "SQL: RULES (7 Regeln inkl. SELECT*-Warnung) + "
                    "k=2 Few-Shot-Beispiele (8er-Pool, Carina-Retrieval) + "
                    "Strategie-Prompt + Security Suffix."
                ),
            )
        with c3:
            st.metric(
                "Nutzerfrage",
                f"{user_tc:,}",
                help="Die Frage des Nutzers inkl. Wrapper-Text",
            )

        # ── Footer: additivity check + schema ratio ────────────────────────
        check_sum = schema_tc + instr_tc + user_tc
        if total_tc > 0:
            schema_ratio = schema_tc / total_tc * 100
            st.caption(
                f"✓ Summe: {schema_tc:,} + {instr_tc:,} + {user_tc:,} = "
                f"**{check_sum:,}** Tokens  ·  "
                f"Schema-Anteil: **{schema_ratio:.1f}%**"
            )

        # ── Graph-RAG badge (SPARQL + SQL — both pipelines set this key) ──
        if token_breakdown.get("graph_rag_applied"):
            reduction = token_breakdown.get("graph_rag_reduction_pct", 0.0)
            if reduction >= 0:
                st.caption(
                    f"🔬 **Graph-RAG aktiv** — Schema-Kontext um **{reduction:.1f}%** reduziert."
                )
            else:
                st.caption(
                    f"🔬 **Graph-RAG aktiv** — Kontext um **{abs(reduction):.1f}%** erweitert "
                    f"(enthält live Filter-Werte + Terminologie)."
                )
        elif "graph_rag_applied" in token_breakdown:
            st.caption(
                "🔬 Graph-RAG: Vollkontext gesendet (Multi-Hop oder keine Keyword-Treffer)."
            )

    # ── Optionally wrap in expander ────────────────────────────────────────
    if use_expander:
        with st.expander(expander_label, expanded=expanded):
            _body()
    else:
        _body()


# ---------------------------------------------------------------------------
# Genie Space latency card
# ---------------------------------------------------------------------------

def render_genie_timer(
    elapsed: float,
    breakdown: dict | None = None,
    accent_color: str = "#003A7D",
    bg_from: str = GENIE_BG_FROM,
    bg_to: str = GENIE_BG_TO,
) -> None:
    """Render a Genie Space latency card in the same style as render_large_timer.

    Shows total wall-clock time with a phase breakdown (SDK · API · Poll · Fetch).
    No 'ohne Overhead' section — Genie has different latency semantics.

    Args:
        elapsed:      Total wall-clock seconds.
        breakdown:    Dict with keys 'init_s', 'start_s', 'polling_s', 'fetch_s'
                      as returned by genie_client._run(). Pass None to omit phases.
        accent_color: CSS colour string for the large time number and breakdown text.
        bg_from:      CSS colour for the left / top gradient stop.
        bg_to:        CSS colour for the right / bottom gradient stop.
    """
    mins  = int(elapsed // 60)
    secs  = elapsed % 60
    label = f"{mins}m {secs:.2f}s" if mins > 0 else f"{secs:.2f}s"

    breakdown_html = ""
    if breakdown:
        parts: list[str] = []
        if breakdown.get("init_s") is not None:
            parts.append(f"SDK {fmt_ms(breakdown['init_s'] * 1000)}")
        if breakdown.get("start_s") is not None:
            parts.append(f"API {fmt_ms(breakdown['start_s'] * 1000)}")
        if breakdown.get("polling_s") is not None:
            parts.append(f"Poll {fmt_ms(breakdown['polling_s'] * 1000)}")
        if breakdown.get("fetch_s") is not None:
            parts.append(f"Fetch {fmt_ms(breakdown['fetch_s'] * 1000)}")
        if parts:
            breakdown_html = (
                f"<div style=\"font-size:13px; opacity:0.55; margin-top:10px;"
                f"font-family:'Courier New',monospace; color:{accent_color}; letter-spacing:0.3px;\">"
                + " \u00b7 ".join(parts) + "</div>"
            )

    st.markdown(
        f"""<div style="
    background: linear-gradient(135deg, {bg_from} 0%, {bg_to} 100%);
    color: #1A1A1A; border-radius: 16px; padding: 14px 24px;
    text-align: center; margin: 4px 0 16px 0;
    border: 1px solid rgba(0,74,151,0.2);
    box-shadow: 0 4px 15px rgba(0,74,151,0.1);
">
    <div style="font-size:11px; color:rgba(0,0,0,0.5); letter-spacing:3px;
                text-transform:uppercase; margin-bottom:4px;">
        Letzte Ausf\u00fchrung \u00b7 Gesamtzeit
    </div>
    <div style="font-size:52px; font-weight:700;
                font-family:'Courier New',monospace;
                line-height:1.1; color:{accent_color};">
        {label}
    </div>
    {breakdown_html}
</div>""",
        unsafe_allow_html=True,
    )
