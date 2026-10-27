# =============================================================================
# genie_panel.py — Interactive Genie Space chat panels
#
# Renders side-by-side chat windows for each configured Genie Space,
# analogous to the Ontology and SQL panels.  Each panel provides:
#   - A text input for natural language questions
#   - A "Fragen" button to send the question to the Genie Space API
#   - Result display: status badge, latency, generated SQL, description
#
# Uses genie_client.run_genie_space() for API calls with timeout handling.
# Session state keys per space: genie_result_{space_id}, genie_input_{space_id}
#
# Exported function:
#   - render_genie_panels() -> None
# =============================================================================

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import GENIE_SPACES
from genie_client import run_genie_space
from ui_components import render_genie_timer


# Mercedes-Benz BKK Design: Corporate Blue + Action Blue
_GENIE_COLORS = ["#003A7D", "#0057A8"]


def render_genie_panels() -> None:
    """Render interactive chat panels for all configured Genie Spaces.

    Creates a column layout with one panel per Genie Space.  Each panel
    has its own text input, submit button, and result display area.
    """
    if not GENIE_SPACES:
        return

    # Ensure Genie ask buttons are at least as tall as ONT/SQL panel buttons.
    # Both use use_container_width=True, but Genie is in a 2-column layout
    # (50 % width each), so we compensate with a fixed min-height.
    st.markdown(
        "<style>button[data-testid='baseButton-primary']"
        " { min-height: 42px !important; font-size: 0.93rem !important; }</style>",
        unsafe_allow_html=True,
    )

    space_items = list(GENIE_SPACES.items())
    cols = st.columns(len(space_items), gap="small")

    for idx, (label, cfg) in enumerate(space_items):
        space_id = cfg["space_id"]
        timeout_s = cfg.get("timeout_s", 120)
        color = _GENIE_COLORS[idx % len(_GENIE_COLORS)]
        short_label = label.replace("Genie: ", "Genie Spaces ")

        # Session state keys
        result_key = f"genie_result_{space_id}"
        input_key = f"genie_input_{space_id}"

        with cols[idx]:
            # Panel header — matches st.subheader() used by ONT/SQL panels
            st.subheader(f"\U0001f52e {short_label}")

            # Timer from previous run
            if result_key in st.session_state:
                _render_timer(st.session_state[result_key], color)

            # Question input
            user_input = st.text_input(
                f"Frage an {short_label}",
                placeholder="z.B. Zeige alle Checkpoints in Halle H01",
                key=input_key,
                label_visibility="collapsed",
            )

            # Submit button — same width behaviour as ONT/SQL panels
            if st.button(
                "AI fragen",
                type="primary",
                key=f"genie_ask_{space_id}",
                use_container_width=True,
            ):
                if not user_input or not user_input.strip():
                    st.warning("Bitte eine Frage eingeben.")
                else:
                    with st.spinner(f"{short_label} wird abgefragt..."):
                        result = run_genie_space(
                            question=user_input.strip(),
                            space_id=space_id,
                            timeout_s=timeout_s,
                        )
                    st.session_state[result_key] = result
                    st.rerun(scope="fragment")

            # Display results
            if result_key in st.session_state:
                _render_result(st.session_state[result_key], color)


# =============================================================================
# Internal display helpers
# =============================================================================

def _render_timer(result: dict, color: str) -> None:
    """Render a Genie Space latency card — same gradient style as SPARQL/SQL panels."""
    elapsed   = result.get("total_elapsed", 0.0)
    breakdown = result.get("latency_breakdown")
    render_genie_timer(elapsed, breakdown, accent_color=color)


def _render_result(result: dict, color: str) -> None:
    """Render the full result panel for a Genie Space response."""
    status = result.get("status", "UNKNOWN")
    error = result.get("error")
    generated_sql = result.get("generated_sql")
    description = result.get("description")
    row_count = result.get("row_count")

    # Status badge
    if error:
        st.markdown(
            f"<span style='background:#FEE2E2;color:#991B1B;border-radius:6px;"
            f"padding:4px 10px;font-size:12px;font-weight:600;border:1px solid #FECACA;'>"
            f"\u2717 {status}</span>",
            unsafe_allow_html=True,
        )
        with st.expander("\u26a0\ufe0f Fehlermeldung", expanded=True):
            st.code(str(error), language=None)
    else:
        row_info = f" | {row_count} Zeilen" if row_count is not None else ""
        st.markdown(
            f"<span style='background:#DBEAFE;color:#003A7D;border-radius:6px;"
            f"padding:4px 10px;font-size:12px;font-weight:600;border:1px solid #BFDBFE;'>"
            f"\u2713 {status}{row_info}</span>",
            unsafe_allow_html=True,
        )

    # Description from Genie
    if description:
        st.markdown(
            f"<div style='font-size:12px;color:#003A7D;margin:8px 0;"
            f"padding:6px 10px;background:#EEF4FB;border-radius:6px;"
            f"border-left:3px solid {color};'>{description}</div>",
            unsafe_allow_html=True,
        )

    # Result preview table
    result_preview = result.get("result_preview")
    if isinstance(result_preview, list) and len(result_preview) > 0:
        st.dataframe(pd.DataFrame(result_preview), use_container_width=True, height=200)

    # Generated SQL
    if generated_sql:
        with st.expander("Generated SQL", expanded=False):
            st.code(generated_sql, language="sql")
