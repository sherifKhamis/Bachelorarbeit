# =============================================================================
# panel_layout.py — Collapsible two-column panel layout (Ontology + SQL)
#
# Renders the side-by-side Ontology Explorer (left) and SQL Datenbank-Chat
# (right) panels with dynamic collapse/expand controls.  Called from app.py
# via render_panel_layout().
#
# Layout behaviour:
#   • Both visible:       [1 : 1] equal width
#   • Left collapsed:     [1 : 30] (thin button | full SQL panel)
#   • Right collapsed:    [30 : 1] (full ONT panel | thin button)
#
# Collapse state persists in session_state ("ont_collapsed", "sql_collapsed")
# and survives fragment reruns during benchmark execution.
#
# The Oxigraph store is passed through to render_ontology_panel() for fast
# SPARQL execution.  The SQL panel does not need it — it queries the
# Databricks SQL warehouse directly.
#
# Exported function:
#   • render_panel_layout(ontology_graph, ontology_schema_context, store=None)
# =============================================================================

import streamlit as st

from ontology_panel import render_ontology_panel
from sql_panel import render_sql_panel
from genie_panel import render_genie_panels

# Column ratio when one side is collapsed vs. both visible
_COLLAPSED_RATIO = 1
_EXPANDED_RATIO  = 30


def render_panel_layout(ontology_graph, ontology_schema_context, store=None, ont_rag=None) -> None:
    """Render the side-by-side Ontology/SQL panels with collapse/expand controls.

    Collapse state is stored in session_state keys:
      - ``ont_collapsed`` (bool)
      - ``sql_collapsed`` (bool)
    """
    ont_c = st.session_state["ont_collapsed"]
    sql_c = st.session_state["sql_collapsed"]

    ratios = _column_ratios(ont_c, sql_c)
    col_left, col_right = st.columns(ratios, gap="small")

    with col_left:
        _render_left_panel(ont_c, ontology_graph, ontology_schema_context, store=store, ont_rag=ont_rag)

    with col_right:
        _render_right_panel(sql_c)

    # ── Genie Space chat panels (below ONT/SQL) ────────────────────────────────
    st.divider()
    render_genie_panels()


# =============================================================================
# Internal helpers
# =============================================================================

def _column_ratios(ont_c: bool, sql_c: bool) -> list[int]:
    """Return [left, right] column ratios based on collapse state."""
    if ont_c and not sql_c:
        return [_COLLAPSED_RATIO, _EXPANDED_RATIO]
    if sql_c and not ont_c:
        return [_EXPANDED_RATIO, _COLLAPSED_RATIO]
    return [1, 1]


def _render_left_panel(collapsed: bool, ontology_graph, ontology_schema_context, store=None, ont_rag=None) -> None:
    """Render the Ontologie Explorer panel (collapsed = just an expand button)."""
    if collapsed:
        if st.button("▶", key="expand_ont", help="Ontologie Explorer einblenden",
                     use_container_width=True):
            st.session_state["ont_collapsed"] = False
            st.rerun(scope="fragment")
    else:
        btn_col, title_col = st.columns([1, 8])
        with btn_col:
            if st.button("◄", key="collapse_ont", help="Ontologie Explorer ausblenden"):
                st.session_state["ont_collapsed"] = True
                st.rerun(scope="fragment")
        with title_col:
            st.subheader("Ontologie Explorer")
        render_ontology_panel(ontology_graph, ontology_schema_context, store=store, ont_rag=ont_rag)


def _render_right_panel(collapsed: bool) -> None:
    """Render the SQL Datenbank-Chat panel (collapsed = just an expand button)."""
    if collapsed:
        if st.button("◄", key="expand_sql", help="SQL Datenbank-Chat einblenden",
                     use_container_width=True):
            st.session_state["sql_collapsed"] = False
            st.rerun(scope="fragment")
    else:
        title_col, btn_col = st.columns([8, 1])
        with title_col:
            st.subheader("SQL Datenbank-Chat")
        with btn_col:
            if st.button("▶", key="collapse_sql", help="SQL Datenbank-Chat ausblenden"):
                st.session_state["sql_collapsed"] = True
                st.rerun(scope="fragment")
        render_sql_panel()
