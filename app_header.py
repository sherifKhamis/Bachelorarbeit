# =============================================================================
# app_header.py — App header rendering & Azure keepalive
#
# Extracted from app.py to keep the main entry point as a pure orchestrator.
#
# Contains:
#   • render_app_header(stats) — Professional header card with ontology stats (no Graph RAG)
#   • render_azure_keepalive() — Hidden iframe preventing WebSocket idle timeout
#
# Both functions call st.* directly and produce no return values.
# =============================================================================

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


def render_app_header(stats: dict) -> None:
    """Render a professional app header with ontology stats.

    Displays a gradient card showing:
      - App title and subtitle
      - Four ontology stats (Tripel, Klassen, Properties, TTL-Datei)

    Args:
        stats: Dict with keys num_triples, num_properties,
               num_instance_types, file_size_mb (all numeric).
    """
    n_triples = stats.get("num_triples", 0)
    n_props = stats.get("num_properties", 0)
    n_instances = stats.get("num_instance_types", 0)
    file_mb = stats.get("file_size_mb", 0)

    # Build full HTML card as a list of parts (avoids indentation/markdown issues)
    parts = []
    parts.append('<div style="background:linear-gradient(135deg,#003A7D 0%,#0057A8 50%,#003A7D 100%);border:1px solid rgba(255,255,255,0.2);border-radius:16px;padding:24px 32px 20px 32px;margin:-8px 0 24px 0;box-shadow:0 8px 32px rgba(0,58,125,0.35);">')
    # Title row
    parts.append('<div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">')
    parts.append('<div style="font-size:36px;">\U0001f9ea</div>')
    parts.append('<div>')
    parts.append('<div style="font-size:22px;font-weight:700;color:#ffffff;letter-spacing:-0.5px;">Ontologiebasiertes GraphRAG vs Text2SQL vs Genie Spaces</div>')
    parts.append('<div style="font-size:12px;color:rgba(255,255,255,0.5);margin-top:2px;letter-spacing:0.5px;">SPARQL + SQL + Genie Spaces \u00b7 Vergleichsanalyse</div>')
    parts.append('</div></div>')

    # Row 1: Ontology stats
    _grid_style = 'display:grid;grid-template-columns:repeat(4,1fr);gap:12px;background:rgba(255,255,255,0.12);border-radius:10px;padding:12px 16px;border:1px solid rgba(255,255,255,0.2);'
    parts.append(f'<div style="{_grid_style}">')
    _stat_label_style = 'font-size:10px;color:rgba(255,255,255,0.45);text-transform:uppercase;letter-spacing:1px;'
    for value, color, label, tooltip in [
        (f"{n_triples:,}", "#FFFFFF", "Tripel", "RDF-Tripel im Oxigraph Store (Subjekt-Praedikat-Objekt)"),
        (str(n_instances), "#B8D4F4", "Klassen", "Benannte Instanz-Typen (Named Individuals wie Areas, Plants)"),
        (str(n_props),     "#FFD080", "Properties", "OWL Object- und Datatype-Properties (Relationen + Attribute)"),
        (f"{file_mb:.1f} MB", "#A8C8E8", "TTL-Datei", "Groesse der Turtle-Ontologiedatei aus dem UC Volume"),
    ]:
        parts.append(f'<div style="text-align:center;cursor:help;" title="{tooltip}"><div style="font-size:20px;font-weight:700;color:{color};">{value}</div><div style="{_stat_label_style}">{label}</div></div>')
    parts.append('</div>')

    parts.append('</div>')  # close outer card

    st.markdown('\n'.join(parts), unsafe_allow_html=True)


def render_azure_keepalive() -> None:
    """Render a hidden iframe that pings /_stcore/health every 4s.

    Prevents the Azure Application Gateway from closing the WebSocket
    connection due to its ~5s idle timeout.
    """
    components.html("""
<iframe srcdoc="
<html><body><script>
setInterval(function() {
    fetch('/_stcore/health', {method: 'HEAD', cache: 'no-store'}).catch(function(){});
}, 4000);
</script></body></html>
" style="display:none; width:0; height:0; border:0;"></iframe>
""", height=0)
