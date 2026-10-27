# =============================================================================
# graph_vis.py — Interactive graph visualization using pyvis
#
# Transforms CONSTRUCT query results (rdflib.Graph of triples) into an
# interactive HTML graph rendered inline via streamlit.components.v1.html().
#
# Key features:
#   • TYPE_STYLE_MAP: per-class node styling (shape, colour, size, icon)
#     e.g. VIN=diamond, Checkpoint=dot, Fault=triangle, Hall=box
#   • PREDICATE_TYPE_INFERENCE: infers node types from edge predicates
#     when explicit rdf:type triples are missing in the CONSTRUCT result
#   • Spatial group boxes: Halls are grouped into Areas (Bodyshop,
#     Assembly, Paintshop) with coloured background rectangles
#   • Edge colouring: EDGE_COLOR_MAP provides distinct colours for each
#     predicate family (caused=red, detected=blue, reworked=green,
#     spatial=purple, VIN-related=orange)
#   • Physics layout: Barnes-Hut with configurable repulsion, damping,
#     and spring constants for readable graph spread
#   • JS interactivity: click-to-highlight, hover tooltips, zoom/pan
#   • Export: PNG and SVG download buttons
#
# Exported function:
#   visualize_construct_graph_interactive(graph: rdflib.Graph) -> str
#     Returns a self-contained HTML string ready for st.components.html().
# =============================================================================

import os
import json
import tempfile
from collections import defaultdict

from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF
from pyvis.network import Network

from config import (
    NODE_COLOR_PALETTE, AREA_COLORS, EDGE_COLOR_MAP,
    SPATIAL_PREDICATES, PREDICATE_TYPE_INFERENCE,
)
from utils import shorten_uri


# =============================================================================
# Node type → (shape, color, size, border_width)
# Rule: no (shape, color) pair appears twice.
# Shapes: dot ●  diamond ◆  square ■  triangle ▲  triangleDown ▼
#         star ★  hexagon ⬡  ellipse ⬭  box ▭
# =============================================================================

TYPE_STYLE_MAP: dict[str, tuple[str, str, int, int]] = {
    # ── Regular spatial hierarchy (always visible on load) ─────────────────
    'Checkpoint':           ('diamond',      '#5A9FD4', 22, 2),  # MB EQ Blue
    'Hall':                 ('square',       '#3D8C5E', 28, 2),  # MB forest green
    'Area':                 ('triangle',     '#CC9900', 30, 2),  # MB amber
    'Plant':                ('star',         '#808080', 35, 3),  # MB steel grey
    # ── Fault core + metadata ──────────────────────────────────────────────
    'Fault':                ('dot',          '#CC0000', 22, 2),  # MB Red
    'FaultType':            ('hexagon',      '#1ABC9C', 20, 2),  # teal hexagon (keep distinct)
    'FaultVehicleLocation': ('ellipse',      '#CC6600', 20, 2),  # MB deep amber
    # ── VIN / model ────────────────────────────────────────────────────────
    'VIN':                  ('box',          '#1E3A5F', 22, 2),  # MB dark blue
    'ModelType':            ('triangleDown', '#4A4060', 22, 2),  # MB dark violet-grey
    # ── Caused fault-linked nodes  (crimson #C0392B) ───────────────────────
    # Hidden until the connected Fault is expanded.
    # Same shape family as their spatial counterpart, different colour.
    'CausedCheckpoint':     ('diamond',      '#8B0000', 20, 2),  # MB dark red
    'CausedHall':           ('square',       '#8B0000', 24, 2),  # MB dark red
    'CausedMark':           ('hexagon',      '#8B0000', 18, 2),  # MB dark red
    'CausedCategory':       ('triangleDown', '#8B0000', 18, 2),  # MB dark red
    # ── Detected fault-linked nodes  (dark blue #1565C0) ───────────────────
    'DetectedCheckpoint':   ('diamond',      '#1E3A5F', 20, 2),  # MB steel blue
    'DetectedHall':         ('square',       '#1E3A5F', 24, 2),  # MB steel blue
    'DetectedMark':         ('hexagon',      '#1E3A5F', 18, 2),  # MB steel blue
    'DetectedCategory':     ('triangleDown', '#1E3A5F', 18, 2),  # MB steel blue
    # ── Reworked fault-linked nodes  (dark green #1E8449) ──────────────────
    'ReworkedCheckpoint':   ('diamond',      '#1B5E20', 20, 2),  # MB forest green
    'ReworkedHall':         ('square',       '#1B5E20', 24, 2),  # MB forest green
    'ReworkedMark':         ('hexagon',      '#1B5E20', 18, 2),  # MB forest green
    'ReworkedCategory':     ('triangleDown', '#1B5E20', 18, 2),  # MB forest green
    # ── Fallback ───────────────────────────────────────────────────────────
    '_Unknown':             ('dot',          '#666666', 18, 1),  # MB steel grey
}

# All checkpoint sub-types (regular + fault-linked) — receive rotated labels
_CHECKPOINT_TYPES = {
    'Checkpoint',
    'CausedCheckpoint', 'DetectedCheckpoint', 'ReworkedCheckpoint',
}

# Shape → Unicode symbol for the legend
_SHAPE_SYMBOL = {
    'dot': '●', 'diamond': '◆', 'square': '■', 'triangle': '▲',
    'triangleDown': '▼', 'star': '★', 'hexagon': '⬡',
    'ellipse': '⬭', 'box': '▭',
}


# =============================================================================
# Spatial hierarchy helpers
# =============================================================================

def _extract_spatial_hierarchy(constructed_graph: Graph) -> dict:
    hierarchy = {
        'cp_to_hall': {}, 'hall_to_area': {}, 'hall_to_plant': {},
        'cp_progress': {}, 'checkpoints': set(), 'halls': set(),
        'areas': set(), 'plants': set(),
    }
    for s, p, o in constructed_graph:
        p_label, s_label = shorten_uri(p), shorten_uri(s)
        if p_label == 'isInHall' and isinstance(o, URIRef):
            o_label = shorten_uri(o)
            hierarchy['cp_to_hall'][s_label] = o_label
            hierarchy['checkpoints'].add(s_label)
            hierarchy['halls'].add(o_label)
        elif p_label == 'isInArea' and isinstance(o, URIRef):
            hierarchy['hall_to_area'][s_label] = shorten_uri(o)
            hierarchy['areas'].add(shorten_uri(o))
        elif p_label == 'isLocatedIn' and isinstance(o, URIRef):
            hierarchy['hall_to_plant'][s_label] = shorten_uri(o)
            hierarchy['plants'].add(shorten_uri(o))
        elif p_label == 'locationProgress' and isinstance(o, Literal):
            try:
                hierarchy['cp_progress'][s_label] = float(str(o))
                hierarchy['checkpoints'].add(s_label)
            except (ValueError, TypeError):
                pass
    return hierarchy


def _compute_grouped_positions(hierarchy: dict) -> tuple[dict, list]:
    positions: dict[str, tuple[float, float]] = {}
    cp_to_hall   = hierarchy['cp_to_hall']
    hall_to_area = hierarchy['hall_to_area']
    cp_progress  = hierarchy['cp_progress']

    hall_groups: dict[str, list] = defaultdict(list)
    ungrouped_cps: list[str] = []
    for cp in hierarchy['checkpoints']:
        hall = cp_to_hall.get(cp)
        if hall:
            hall_groups[hall].append(cp)
        else:
            ungrouped_cps.append(cp)

    for hall in hall_groups:
        hall_groups[hall].sort(key=lambda n: cp_progress.get(n, 0))
    ungrouped_cps.sort(key=lambda n: cp_progress.get(n, 0))

    area_hall_groups: dict[str, list] = defaultdict(list)
    for hall in hall_groups:
        area_hall_groups[hall_to_area.get(hall, 'Unknown')].append(hall)

    def avg_progress(hall):
        cps = hall_groups[hall]
        return sum(cp_progress.get(cp, 0) for cp in cps) / len(cps) if cps else 0

    for area in area_hall_groups:
        area_hall_groups[area].sort(key=avg_progress)

    sorted_areas = sorted(
        area_hall_groups.keys(),
        key=lambda a: min(avg_progress(h) for h in area_hall_groups[a]) if area_hall_groups[a] else 0,
    )

    cp_spacing_x, group_gap_x = 180, 350
    x_cursor = 0.0
    group_meta: list[dict] = []

    for area in sorted_areas:
        for hall in area_hall_groups[area]:
            x_start = x_cursor
            for cp in hall_groups[hall]:
                positions[cp] = (x_cursor, 0)
                x_cursor += cp_spacing_x
            group_meta.append({
                'x_start': x_start,
                'x_end':   x_cursor - cp_spacing_x,
                'hall':    hall,
                'area':    area,
                'plant':   hierarchy['hall_to_plant'].get(hall, ''),
            })
            x_cursor += group_gap_x

    for cp in ungrouped_cps:
        positions[cp] = (x_cursor, 0)
        x_cursor += cp_spacing_x

    return positions, group_meta


# =============================================================================
# Main visualization function
# =============================================================================

def visualize_construct_graph_interactive(constructed_graph: Graph) -> str | None:
    """Build a self-contained HTML string with an interactive pyvis graph."""
    hierarchy = _extract_spatial_hierarchy(constructed_graph)
    has_spatial_data = bool(hierarchy['checkpoints'])
    cp_positions, group_meta = {}, []
    if has_spatial_data:
        cp_positions, group_meta = _compute_grouped_positions(hierarchy)

    # ── Phase 1: node types from rdf:type triples ──────────────────────────
    node_types: dict[str, str] = {}
    for s, p, o in constructed_graph:
        if p == RDF.type and isinstance(o, URIRef):
            s_label, o_label = shorten_uri(s), shorten_uri(o)
            if o_label not in ('Class', 'ObjectProperty', 'DatatypeProperty', 'Ontology', 'Property'):
                node_types[s_label] = o_label

    # ── Phase 2: spatial hierarchy types ──────────────────────────────────
    for cp in hierarchy['checkpoints']: node_types.setdefault(cp, 'Checkpoint')
    for h  in hierarchy['halls']:       node_types.setdefault(h, 'Hall')
    for a  in hierarchy['areas']:       node_types.setdefault(a, 'Area')
    for p  in hierarchy['plants']:      node_types.setdefault(p, 'Plant')

    # ── Phase 3: infer sub-types from edge predicates ─────────────────────
    # Fault-linked nodes (Caused/Detected/Reworked Checkpoint/Hall/Mark/Category)
    # get their own sub-type so they can be styled differently from regular
    # spatial nodes and start hidden until the Fault is expanded.
    for s, p, o in constructed_graph:
        if p == RDF.type or isinstance(o, Literal):
            continue
        if isinstance(o, URIRef):
            p_label, s_label, o_label = shorten_uri(p), shorten_uri(s), shorten_uri(o)
            inference = PREDICATE_TYPE_INFERENCE.get(p_label)
            if inference:
                s_inferred, o_inferred = inference
                if s_inferred and s_label not in node_types:
                    node_types[s_label] = s_inferred
                if o_inferred and o_label not in node_types:
                    node_types[o_label] = o_inferred

    # ── Phase 4: color map (built from TYPE_STYLE_MAP, not palette index) ──
    type_color_map: dict[str, str] = {
        t: TYPE_STYLE_MAP.get(t, TYPE_STYLE_MAP['_Unknown'])[1]
        for t in set(node_types.values())
    }

    # ── Determine primary (initially visible) nodes ────────────────────────
    # Only regular spatial Checkpoints are visible on load.
    # Fault-linked nodes (CausedCheckpoint, DetectedCheckpoint, etc.) are
    # hidden and revealed when the user expands a Fault node via the menu.
    primary_nodes: set[str] = set()
    if has_spatial_data:
        primary_nodes = set(hierarchy['checkpoints'])
    else:
        out_degree: dict[str, int] = defaultdict(int)
        in_degree:  dict[str, int] = defaultdict(int)
        for s, p, o in constructed_graph:
            if isinstance(o, URIRef) and p != RDF.type:
                p_label = shorten_uri(p)
                if p_label not in SPATIAL_PREDICATES:
                    out_degree[shorten_uri(s)] += 1
                    in_degree[shorten_uri(o)]  += 1
        if out_degree:
            for node in out_degree:
                if in_degree.get(node, 0) == 0:
                    primary_nodes.add(node)
            if not primary_nodes:
                sorted_by_out = sorted(out_degree.keys(), key=lambda n: out_degree[n], reverse=True)
                primary_nodes = set(sorted_by_out[:max(5, len(sorted_by_out) // 3)])

    # ── Build pyvis Network ────────────────────────────────────────────────
    net = Network(height="750px", width="100%", directed=False, bgcolor="#FFFFFF", font_color="#1A1A1A")
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {"gravitationalConstant":-5000,"centralGravity":0.4,"springLength":150,"springConstant":0.03,"damping":0.4,"avoidOverlap":0.6},
        "stabilization": {"enabled":true,"iterations":400,"updateInterval":25,"fit":true}
      },
      "interaction": {"dragNodes":true,"dragView":true,"zoomView":true,"hover":false,"tooltipDelay":300,"navigationButtons":true},
      "edges": {"smooth":{"type":"curvedCW","roundness":0.12}}
    }
    """)

    added_nodes: set[str] = set()

    def add_node_if_missing(label: str, node_type: str = '_Unknown', hidden: bool = False):
        if label in added_nodes:
            return
        added_nodes.add(label)
        node_types.setdefault(label, node_type)
        style  = TYPE_STYLE_MAP.get(node_type, TYPE_STYLE_MAP['_Unknown'])
        shape, color, size, bw = style
        is_cp_type = node_type in _CHECKPOINT_TYPES
        node_kwargs = {
            'label': ' ' if is_cp_type else label,
            'color': {'background': color, 'border': color,
                      'highlight': {'background': color, 'border': '#333'}},
            'shape': shape, 'size': size,
            'font': ({'size': 1, 'color': 'rgba(0,0,0,0)', 'face': 'Arial'}
                     if is_cp_type
                     else {'size': 9, 'face': 'Arial'}),
            'borderWidth': bw,
            'hidden': hidden,
        }
        if label in cp_positions:
            x, y = cp_positions[label]
            node_kwargs.update({'x': x, 'y': y, 'fixed': {'x': True, 'y': True}, 'physics': False})
        net.add_node(label, **node_kwargs)

    # ── Process triples into nodes / edges ─────────────────────────────────
    for s, p, o in constructed_graph:
        if p == RDF.type:
            s_label = shorten_uri(s)
            add_node_if_missing(s_label, node_types.get(s_label, '_Unknown'),
                                hidden=s_label not in primary_nodes)
            continue
        if isinstance(o, Literal):
            s_label = shorten_uri(s)
            add_node_if_missing(s_label, node_types.get(s_label, '_Unknown'),
                                hidden=s_label not in primary_nodes)
            continue

        s_label, p_label = shorten_uri(s), shorten_uri(p)
        s_type = node_types.get(s_label, '_Unknown')

        if p_label in SPATIAL_PREDICATES and has_spatial_data:
            add_node_if_missing(s_label, s_type, hidden=s_label not in primary_nodes)
            if isinstance(o, URIRef):
                o_label = shorten_uri(o)
                add_node_if_missing(o_label, node_types.get(o_label, '_Unknown'),
                                    hidden=o_label not in primary_nodes)
            continue

        if isinstance(o, URIRef):
            o_label = shorten_uri(o)
            o_type  = node_types.get(o_label, '_Unknown')
            add_node_if_missing(s_label, s_type, hidden=s_label not in primary_nodes)
            add_node_if_missing(o_label, o_type, hidden=o_label not in primary_nodes)
            edge_color = EDGE_COLOR_MAP.get(p_label, '#7F8C8D')
            net.add_edge(s_label, o_label, label=p_label, color=edge_color,
                         font={'size': 7, 'align': 'middle', 'color': '#555'}, width=1.5)
        elif isinstance(o, BNode):
            o_label = f"_:{o}"
            add_node_if_missing(s_label, s_type, hidden=s_label not in primary_nodes)
            add_node_if_missing(o_label, '_Unknown', hidden=True)
            net.add_edge(s_label, o_label, label=p_label, color='#CCC',
                         font={'size': 7, 'align': 'middle'})

    if not added_nodes:
        return None

    # ── Generate HTML ──────────────────────────────────────────────────────
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w')
    tmp_path = tmp_file.name
    tmp_file.close()
    net.save_graph(tmp_path)
    with open(tmp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    os.unlink(tmp_path)

    node_type_map_json = json.dumps(node_types)
    # Rotated labels for ALL checkpoint sub-types (spatial + fault-linked)
    checkpoint_nodes_json = json.dumps(
        [label for label, ntype in node_types.items() if ntype in _CHECKPOINT_TYPES]
    )
    # Build final color map from TYPE_STYLE_MAP for JS use
    all_types = sorted(set(node_types.values()))
    final_type_color_map = {
        t: TYPE_STYLE_MAP.get(t, TYPE_STYLE_MAP['_Unknown'])[1]
        for t in all_types
    }
    type_color_map_json = json.dumps(final_type_color_map)

    legend_html       = _build_legend_html(all_types)
    export_html       = _build_export_buttons_js()
    stabilize_js      = _build_stabilize_js()
    category_menu_js  = _build_category_menu_js(node_type_map_json, type_color_map_json)
    group_boxes_js    = _build_group_boxes_js(group_meta) if group_meta else ''
    rotated_labels_js = _build_rotated_labels_js(checkpoint_nodes_json)

    html_content = html_content.replace(
        '</body>',
        f'{legend_html}{export_html}{stabilize_js}{category_menu_js}{group_boxes_js}{rotated_labels_js}</body>',
    )
    return html_content


# =============================================================================
# JS / HTML builder helpers
# =============================================================================

def _build_stabilize_js() -> str:
    return """
<script>
network.on("stabilizationIterationsDone", function() {
    network.setOptions({ physics: { enabled: false } });
    network.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
});
</script>
"""


def _build_rotated_labels_js(checkpoint_nodes_json: str) -> str:
    return f"""
<script>
(function() {{
    var checkpointNodes = {checkpoint_nodes_json};
    network.on("afterDrawing", function(ctx) {{
        var positions = network.getPositions(checkpointNodes);
        ctx.font = '9px Arial';
        ctx.fillStyle = '#333';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        checkpointNodes.forEach(function(nodeId) {{
            var node = network.body.data.nodes.get(nodeId);
            if (!node || node.hidden) return;
            var p = positions[nodeId];
            if (!p) return;
            ctx.save();
            ctx.translate(p.x, p.y + 32);
            ctx.rotate(-Math.PI / 4);
            ctx.fillText(nodeId, 0, 0);
            ctx.restore();
        }});
    }});
}})();
</script>
"""


def _build_group_boxes_js(group_meta: list[dict]) -> str:
    if not group_meta:
        return ''
    groups_js = "[" + ",".join(
        f'{{"xStart":{g["x_start"]},"xEnd":{g["x_end"]},'
        f'"hall":"{g["hall"]}","area":"{g["area"]}","plant":"{g["plant"]}",'
        f'"bgColor":"{AREA_COLORS.get(g["area"], "#F5F5F5")}"}}'
        for g in group_meta
    ) + "]"
    return f"""
<script>
(function() {{
    var groups = {groups_js};
    network.on("afterDrawing", function(ctx) {{
        groups.forEach(function(g) {{
            var padding=90,x=g.xStart-padding,y=-160,w=(g.xEnd-g.xStart)+padding*2,h=360;
            ctx.fillStyle=g.bgColor;ctx.globalAlpha=0.18;ctx.beginPath();ctx.roundRect(x,y,w,h,12);ctx.fill();
            ctx.globalAlpha=1.0;ctx.strokeStyle='#666';ctx.lineWidth=2;ctx.setLineDash([8,4]);ctx.stroke();ctx.setLineDash([]);
            ctx.fillStyle='#222';ctx.font='bold 18px Arial';ctx.textAlign='center';
            ctx.fillText('Halle: '+g.hall,(g.xStart+g.xEnd)/2,y+26);
            ctx.fillStyle='#444';ctx.font='bold 16px Arial';
            ctx.fillText('Bereich: '+g.area,(g.xStart+g.xEnd)/2,y+h-50);
            if (g.plant) {{
                ctx.fillStyle='#555';ctx.font='bold 14px Arial';
                ctx.fillText('Werk: '+g.plant,(g.xStart+g.xEnd)/2,y+h-22);
            }}
        }});
    }});
}})();
</script>
"""


def _build_export_buttons_js() -> str:
    return """
<div id="export-toolbar" style="position:absolute;top:10px;right:10px;z-index:1000;display:flex;gap:8px;">
  <button id="btn-export-png" style="padding:6px 14px;font-size:12px;font-weight:600;cursor:pointer;background:#003A7D;color:#fff;border:none;border-radius:4px;box-shadow:0 2px 4px rgba(0,58,125,0.2);">Export PNG</button>
  <button id="btn-export-svg" style="padding:6px 14px;font-size:12px;font-weight:600;cursor:pointer;background:#0057A8;color:#fff;border:none;border-radius:4px;box-shadow:0 2px 4px rgba(0,58,125,0.2);">Export SVG</button>
</div>
<script>
document.getElementById('btn-export-png').addEventListener('click', function() {
    var canvas=document.querySelector('canvas');if(!canvas)return;
    var s=2,ec=document.createElement('canvas');ec.width=canvas.width*s;ec.height=canvas.height*s;
    var ctx=ec.getContext('2d');ctx.fillStyle='#FFF';ctx.fillRect(0,0,ec.width,ec.height);ctx.scale(s,s);ctx.drawImage(canvas,0,0);
    var a=document.createElement('a');a.download='knowledge_graph.png';a.href=ec.toDataURL('image/png');a.click();
});
document.getElementById('btn-export-svg').addEventListener('click', function() {
    var nodes=network.body.data.nodes.get(),edges=network.body.data.edges.get(),pos=network.getPositions();
    var minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
    nodes.forEach(function(n){var p=pos[n.id];if(p){minX=Math.min(minX,p.x);maxX=Math.max(maxX,p.x);minY=Math.min(minY,p.y);maxY=Math.max(maxY,p.y);}});
    var pad=100,sw=(maxX-minX)+pad*2,sh=(maxY-minY)+pad*2,ox=-minX+pad,oy=-minY+pad;
    var svg='<?xml version="1.0"?>\\n<svg xmlns="http://www.w3.org/2000/svg" width="'+sw+'" height="'+sh+'" viewBox="0 0 '+sw+' '+sh+'">\\n<rect width="100%" height="100%" fill="#FFF"/>\\n';
    svg+='<defs><marker id="ah" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#666"/></marker></defs>\\n';
    edges.forEach(function(e){var f=pos[e.from],t=pos[e.to];if(!f||!t)return;var c=(e.color&&e.color.color)||'#7F8C8D';svg+='<line x1="'+(f.x+ox)+'" y1="'+(f.y+oy)+'" x2="'+(t.x+ox)+'" y2="'+(t.y+oy)+'" stroke="'+c+'" stroke-width="1.5" marker-end="url(#ah)"/>\\n';});
    nodes.forEach(function(n){var p=pos[n.id];if(!p||n.hidden)return;var x=p.x+ox,y=p.y+oy,c=(n.color&&n.color.background)||'#95A5A6',r=(n.size||20)*0.8;svg+='<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="'+c+'" stroke="#fff" stroke-width="1.5"/>\\n';if(n.label)svg+='<text x="'+x+'" y="'+(y+r+12)+'" font-size="9" fill="#333" text-anchor="middle">'+n.label+'</text>\\n';});
    svg+='</svg>';
    var blob=new Blob([svg],{type:'image/svg+xml'}),a=document.createElement('a');a.download='knowledge_graph.svg';a.href=URL.createObjectURL(blob);a.click();
});
</script>
"""


def _build_legend_html(all_types: list[str]) -> str:
    """Build a grouped legend using TYPE_STYLE_MAP symbols and colours."""
    # Group order and section titles
    groups = [
        ('Räumliche Hierarchie',    ['Plant', 'Area', 'Hall', 'Checkpoint']),
        ('Fehler',                  ['Fault', 'FaultType', 'FaultVehicleLocation']),
        ('VIN / Modell',            ['VIN', 'ModelType']),
        ('Verursacht (Caused)',      ['CausedCheckpoint', 'CausedHall', 'CausedMark', 'CausedCategory']),
        ('Erkannt (Detected)',       ['DetectedCheckpoint', 'DetectedHall', 'DetectedMark', 'DetectedCategory']),
        ('Repariert (Reworked)',     ['ReworkedCheckpoint', 'ReworkedHall', 'ReworkedMark', 'ReworkedCategory']),
    ]
    type_set = set(all_types)

    html = (
        '<div id="graph-legend" style="position:absolute;top:10px;left:10px;'
        'background:rgba(255,255,255,0.96);border:1px solid #ddd;border-radius:8px;'
        'padding:10px 14px;z-index:1000;box-shadow:0 2px 8px rgba(0,0,0,0.14);'
        'max-width:210px;font-family:Arial,sans-serif;">'
        '<div style="font-size:12px;font-weight:bold;margin-bottom:6px;">Legende</div>'
    )

    for section_title, type_list in groups:
        visible = [t for t in type_list if t in type_set]
        if not visible:
            continue
        html += (
            f'<div style="font-size:10px;font-weight:700;color:#888;'
            f'text-transform:uppercase;letter-spacing:1px;margin:6px 0 3px 0;">'
            f'{section_title}</div>'
        )
        for t in visible:
            style = TYPE_STYLE_MAP.get(t, TYPE_STYLE_MAP['_Unknown'])
            shape, color, _, _ = style
            symbol = _SHAPE_SYMBOL.get(shape, '●')
            html += (
                f'<div style="display:flex;align-items:center;margin:2px 0;">'
                f'<span style="width:16px;font-size:13px;color:{color};'
                f'margin-right:6px;line-height:1;">{symbol}</span>'
                f'<span style="font-size:11px;">{t}</span></div>'
            )

    html += (
        '<div style="margin-top:8px;border-top:1px solid #eee;padding-top:5px;">'
        '<div style="font-size:10px;color:#999;">'
        '<span style="color:#CC0000;">─ isCaused</span> '
        '<span style="color:#2B5F8A;">─ isDetected</span> '
        '<span style="color:#1B5E20;">─ isReworked</span>'
        '</div>'
        '<div style="font-size:10px;color:#aaa;margin-top:2px;">Klick → Menü</div>'
        '</div></div>'
    )
    return html


def _build_category_menu_js(node_type_map_json: str, type_color_map_json: str) -> str:
    return f"""
<style>
#category-menu {{
    display: none; position: fixed; background: #fff; border: 1px solid #ccc;
    border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.18); padding: 6px 0;
    z-index: 9999; min-width: 200px; font-family: Arial, sans-serif; font-size: 12px;
}}
#category-menu .menu-title {{
    padding: 6px 14px; font-weight: 700; font-size: 11px; color: #666;
    border-bottom: 1px solid #eee; margin-bottom: 2px;
}}
#category-menu .menu-item {{
    padding: 7px 14px; cursor: pointer; display: flex; align-items: center;
    gap: 8px; transition: background 0.1s;
}}
#category-menu .menu-item:hover {{ background: #f0f4ff; }}
#category-menu .menu-item .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
#category-menu .menu-item .count {{ color: #999; margin-left: auto; font-size: 11px; }}
#category-menu .menu-divider {{ height: 1px; background: #eee; margin: 4px 0; }}
#category-menu .slider-panel {{
    padding: 10px 14px; border-top: 1px solid #eee; margin-top: 4px;
}}
#category-menu .slider-panel .slider-header {{
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;
}}
#category-menu .slider-panel .slider-label {{
    font-size: 11px; color: #666; font-weight: 600;
}}
#category-menu .slider-panel .slider-value {{
    font-size: 13px; font-weight: 700; color: #333;
}}
#category-menu .slider-panel input[type="range"] {{
    width: 100%; margin: 4px 0 10px 0; cursor: pointer;
}}
#category-menu .slider-panel .btn-row {{
    display: flex; gap: 6px;
}}
#category-menu .slider-panel button {{
    flex: 1; padding: 6px 10px; font-size: 11px; font-weight: 600;
    border: none; border-radius: 4px; cursor: pointer;
}}
#category-menu .slider-panel .btn-confirm {{
    background: #003A7D; color: #fff;
}}
#category-menu .slider-panel .btn-confirm:hover {{ background: #002D5E; }}
#category-menu .slider-panel .btn-cancel {{
    background: #eee; color: #666;
}}
#category-menu .slider-panel .btn-cancel:hover {{ background: #ddd; }}
</style>
<div id="category-menu"></div>
<script>
(function() {{
    var nodeTypeMap      = {node_type_map_json};
    var typeColorMap     = {type_color_map_json};
    var expandedNodes    = {{}};
    var revealedNodes    = {{}};
    var hiddenEdgesByNode = {{}};
    var activeHaloTargetIds = [];
    var activeHaloColor     = '#F39C12';
    var activeHaloSourceId  = null;
    var menuEl             = document.getElementById('category-menu');
    var menuJustOpened     = false;
    var currentClickedId   = null;
    var currentNodeList    = [];
    var currentCategory    = null;
    var lastHaloDraw       = 0;
    var HALO_THROTTLE      = 50;

    var FAULT_KEEP_PREFIXES = ['isCaused', 'isDetected', 'isReworked'];

    // ── Draw halo glow on parent nodes ─────────────────────────────────────
    network.on("afterDrawing", function(ctx) {{
        if (activeHaloTargetIds.length === 0) return;
        var now = Date.now();
        if (now - lastHaloDraw < HALO_THROTTLE) return;
        lastHaloDraw = now;
        var positions = network.getPositions(activeHaloTargetIds);
        activeHaloTargetIds.forEach(function(nodeId) {{
            var node = network.body.data.nodes.get(nodeId);
            if (!node || node.hidden) return;
            var pos = positions[nodeId];
            if (!pos) return;
            var r = (node.size || 20) * 1.8;
            ctx.save();
            var grad = ctx.createRadialGradient(pos.x, pos.y, r * 0.5, pos.x, pos.y, r * 2.4);
            grad.addColorStop(0, activeHaloColor + '55');
            grad.addColorStop(1, activeHaloColor + '00');
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, r * 2.4, 0, 2 * Math.PI);
            ctx.fillStyle = grad;
            ctx.fill();
            ctx.restore();
        }});
    }});

    function clearHalo() {{
        if (activeHaloTargetIds.length > 0 || activeHaloSourceId !== null) {{
            activeHaloTargetIds = [];
            activeHaloSourceId  = null;
            network.redraw();
        }}
    }}

    function hideMenu() {{ menuEl.style.display = 'none'; currentClickedId = null; }}

    function showMenuAt(x, y, html) {{
        menuEl.innerHTML = html;
        menuEl.style.left = x + 'px';
        menuEl.style.top  = y + 'px';
        menuEl.style.display = 'block';
        menuJustOpened = true;
        setTimeout(function() {{ menuJustOpened = false; }}, 100);
    }}

    function getClickPosition(params) {{
        var domPos = params.pointer.DOM;
        var container = document.getElementById('mynetwork');
        if (container) {{
            var rect = container.getBoundingClientRect();
            return {{ x: rect.left + domPos.x, y: rect.top + domPos.y }};
        }}
        return {{ x: domPos.x, y: domPos.y }};
    }}

    document.addEventListener('mousedown', function(e) {{
        if (!menuJustOpened && !menuEl.contains(e.target)) {{ hideMenu(); }}
    }});

    function getVisibleParents(nodeId) {{
        return network.getConnectedNodes(nodeId).filter(function(nid) {{
            var nd = network.body.data.nodes.get(nid);
            return nd && !nd.hidden;
        }});
    }}

    function appendParentHighlightItem(html, clickedId) {{
        if (revealedNodes[clickedId] === undefined) return html;
        var parents = getVisibleParents(clickedId);
        if (parents.length === 0) return html;
        html += '<div class="menu-divider"></div>';
        if (activeHaloSourceId === clickedId) {{
            html += '<div class="menu-item" data-action="clear-highlight">'
                  + '<span class="dot" style="background:#F39C12;border-radius:0;transform:rotate(45deg);"></span>'
                  + '<span>Hervorhebung entfernen</span></div>';
        }} else {{
            html += '<div class="menu-item" data-action="show-parent-connections">'
                  + '<span class="dot" style="background:#F39C12;border-radius:0;transform:rotate(45deg);"></span>'
                  + '<span>Elternknoten hervorheben</span>'
                  + '<span class="count">' + parents.length + '</span></div>';
        }}
        return html;
    }}

    function handleHaloAction(action, clickedId) {{
        if (action === 'show-parent-connections') {{
            var parents = getVisibleParents(clickedId);
            activeHaloTargetIds = parents;
            activeHaloColor     = revealedNodes[clickedId] || '#F39C12';
            activeHaloSourceId  = clickedId;
            network.redraw();
            hideMenu();
            return true;
        }}
        if (action === 'clear-highlight') {{
            clearHalo();
            hideMenu();
            return true;
        }}
        return false;
    }}

    function _restoreNodeEdgesAndHalo(nodeId) {{
        delete revealedNodes[nodeId];
        if (activeHaloSourceId === nodeId) {{ clearHalo(); }}
        if (hiddenEdgesByNode[nodeId]) {{
            var allEdges = network.body.data.edges;
            hiddenEdgesByNode[nodeId].forEach(function(eid) {{
                allEdges.update({{id: eid, hidden: false}});
            }});
            delete hiddenEdgesByNode[nodeId];
        }}
    }}

    function revealNodes(clickedId, nodeIds, count) {{
        var allNodes = network.body.data.nodes;
        var allEdges = network.body.data.edges;
        var toReveal = nodeIds.slice(0, count);
        var revealed = [];
        toReveal.forEach(function(nid) {{
            allNodes.update({{id: nid, hidden: false}});
            revealed.push(nid);
            var connEdges = network.getConnectedEdges(nid);
            var edgesToHide = [];
            connEdges.forEach(function(eid) {{
                var edge  = allEdges.get(eid);
                var label = (edge && edge.label) ? edge.label : '';

                // Always keep the edge connecting back to the parent that revealed this node.
                var connectsToParent = edge && (
                    String(edge.from) === String(clickedId) ||
                    String(edge.to)   === String(clickedId)
                );
                if (connectsToParent) {{
                    allEdges.update({{id: eid, hidden: false}});
                    return;
                }}

                // Keep isCaused*/isDetected*/isReworked* edges (Fault → fault-linked node).
                // Also ensure the far endpoint is visible.
                var keepFaultEdge = false;
                for (var pi = 0; pi < FAULT_KEEP_PREFIXES.length; pi++) {{
                    if (label.indexOf(FAULT_KEEP_PREFIXES[pi]) === 0) {{
                        keepFaultEdge = true; break;
                    }}
                }}
                if (keepFaultEdge) {{
                    allEdges.update({{id: eid, hidden: false}});
                    var otherId = String(edge.from === nid ? edge.to : edge.from);
                    var otherNode = allNodes.get(otherId);
                    if (otherNode && otherNode.hidden) {{
                        allNodes.update({{id: otherId, hidden: false}});
                    }}
                    return;
                }}

                edgesToHide.push(eid);
                allEdges.update({{id: eid, hidden: true}});
            }});
            hiddenEdgesByNode[nid] = edgesToHide;
            revealedNodes[nid] = typeColorMap[nodeTypeMap[nid] || '_Unknown'] || '#95A5A6';
        }});
        if (revealed.length > 0) {{
            positionNodesAround(clickedId, revealed);
            if (!expandedNodes[clickedId]) expandedNodes[clickedId] = [];
            expandedNodes[clickedId] = expandedNodes[clickedId].concat(revealed);
        }}
        hideMenu();
    }}

    function showSliderPanel(clickedId, nodeIds, categoryName, categoryColor) {{
        var total = nodeIds.length;
        var defaultCount = Math.min(10, total);
        currentClickedId = clickedId;
        currentNodeList  = nodeIds;
        currentCategory  = categoryName;

        var html = '<div class="menu-title">' + (categoryName || 'Alle') + ' anzeigen</div>';
        html += '<div class="slider-panel">';
        html += '<div class="slider-header">';
        html += '<span class="slider-label">Anzahl wählen:</span>';
        html += '<span class="slider-value" id="slider-display">' + defaultCount + ' von ' + total + '</span>';
        html += '</div>';
        html += '<input type="range" id="node-count-slider" min="1" max="' + total + '" value="' + defaultCount + '">';
        html += '<div class="btn-row">';
        html += '<button class="btn-cancel" id="btn-slider-cancel">Abbrechen</button>';
        html += '<button class="btn-confirm" id="btn-slider-confirm">Anzeigen</button>';
        html += '</div></div>';

        menuEl.innerHTML = html;

        var slider  = document.getElementById('node-count-slider');
        var display = document.getElementById('slider-display');
        slider.addEventListener('input', function() {{
            display.textContent = this.value + ' von ' + total;
        }});
        document.getElementById('btn-slider-cancel').addEventListener('click', function(e) {{
            e.stopPropagation(); hideMenu();
        }});
        document.getElementById('btn-slider-confirm').addEventListener('click', function(e) {{
            e.stopPropagation();
            revealNodes(currentClickedId, currentNodeList, parseInt(slider.value, 10));
        }});
    }}

    network.on("click", function(params) {{
        if (params.nodes.length !== 1) {{
            hideMenu();
            clearHalo();
            return;
        }}

        var clickedId = params.nodes[0];
        var allNodes  = network.body.data.nodes;
        var pos = getClickPosition(params);

        if (activeHaloSourceId !== null && activeHaloSourceId !== clickedId) {{
            clearHalo();
        }}

        if (expandedNodes[clickedId]) {{ showCollapseMenu(pos, clickedId); return; }}

        var connectedIds    = network.getConnectedNodes(clickedId);
        var hiddenNeighbors = connectedIds.filter(function(nid) {{
            var node = allNodes.get(nid); return node && node.hidden;
        }});

        if (hiddenNeighbors.length === 0) {{
            if (connectedIds.length > 0) {{ showCollapseMenu(pos, clickedId); }}
            return;
        }}

        var categories = {{}};
        hiddenNeighbors.forEach(function(nid) {{
            var ntype = nodeTypeMap[nid] || '_Unknown';
            if (!categories[ntype]) categories[ntype] = [];
            categories[ntype].push(nid);
        }});

        var isChildNode = false;
        Object.keys(expandedNodes).forEach(function(pid) {{
            if (expandedNodes[pid].indexOf(clickedId) >= 0) isChildNode = true;
        }});

        var html = '<div class="menu-title">Kategorie auswählen</div>';
        html += '<div class="menu-item" data-action="all"><span class="dot" style="background:#333;"></span>'
              + '<span>Alle</span><span class="count">' + hiddenNeighbors.length + '</span></div>';
        html += '<div class="menu-divider"></div>';
        Object.keys(categories).sort().forEach(function(cat) {{
            var color = typeColorMap[cat] || '#95A5A6';
            html += '<div class="menu-item" data-action="category" data-category="' + cat + '" data-color="' + color + '">'
                  + '<span class="dot" style="background:' + color + ';"></span>'
                  + '<span>' + cat + '</span>'
                  + '<span class="count">' + categories[cat].length + '</span></div>';
        }});
        if (isChildNode) {{
            html += '<div class="menu-divider"></div>';
            html += '<div class="menu-item" data-action="hide-self">'
                  + '<span class="dot" style="background:#E74C3C;"></span>'
                  + '<span>Ausblenden</span></div>';
        }}
        html = appendParentHighlightItem(html, clickedId);

        showMenuAt(pos.x, pos.y, html);

        menuEl.querySelectorAll('.menu-item').forEach(function(item) {{
            item.addEventListener('click', function(e) {{
                e.stopPropagation();
                var action = this.getAttribute('data-action');
                if (handleHaloAction(action, clickedId)) return;
                if (action === 'hide-self') {{
                    hideNodeAndCleanup(clickedId); hideMenu();
                }} else if (action === 'all') {{
                    showSliderPanel(clickedId, hiddenNeighbors, null, '#333');
                }} else if (action === 'category') {{
                    showSliderPanel(clickedId, categories[this.getAttribute('data-category')],
                                   this.getAttribute('data-category'), this.getAttribute('data-color'));
                }}
            }});
        }});
    }});

    function showCollapseMenu(pos, clickedId) {{
        var allNodes = network.body.data.nodes;
        var html = '<div class="menu-title">Aktionen</div>';
        html += '<div class="menu-item" data-action="collapse">'
              + '<span class="dot" style="background:#E74C3C;"></span><span>Ausblenden</span></div>';

        var connectedIds    = network.getConnectedNodes(clickedId);
        var hiddenNeighbors = connectedIds.filter(function(nid) {{
            var node = allNodes.get(nid); return node && node.hidden;
        }});
        if (hiddenNeighbors.length > 0) {{
            html += '<div class="menu-divider"></div>';
            var categories = {{}};
            hiddenNeighbors.forEach(function(nid) {{
                var ntype = nodeTypeMap[nid] || '_Unknown';
                if (!categories[ntype]) categories[ntype] = [];
                categories[ntype].push(nid);
            }});
            html += '<div class="menu-item" data-action="all"><span class="dot" style="background:#333;"></span>'
                  + '<span>Alle</span><span class="count">' + hiddenNeighbors.length + '</span></div>';
            Object.keys(categories).sort().forEach(function(cat) {{
                var color = typeColorMap[cat] || '#95A5A6';
                html += '<div class="menu-item" data-action="category" data-category="' + cat + '" data-color="' + color + '">'
                      + '<span class="dot" style="background:' + color + ';"></span>'
                      + '<span>' + cat + '</span>'
                      + '<span class="count">' + categories[cat].length + '</span></div>';
            }});
        }}
        html = appendParentHighlightItem(html, clickedId);

        showMenuAt(pos.x, pos.y, html);

        menuEl.querySelectorAll('.menu-item').forEach(function(item) {{
            item.addEventListener('click', function(e) {{
                e.stopPropagation();
                var action = this.getAttribute('data-action');
                if (handleHaloAction(action, clickedId)) return;
                if (action === 'collapse') {{
                    if (expandedNodes[clickedId]) {{ collapseNode(clickedId); }}
                    else {{ hideNodeAndCleanup(clickedId); }}
                    hideMenu();
                }} else if (action === 'all') {{
                    var hidden = connectedIds.filter(function(nid) {{
                        var nd = allNodes.get(nid); return nd && nd.hidden;
                    }});
                    showSliderPanel(clickedId, hidden, null, '#333');
                }} else if (action === 'category') {{
                    var cat   = this.getAttribute('data-category');
                    var color = this.getAttribute('data-color');
                    var catNodes = connectedIds.filter(function(nid) {{
                        var nd = allNodes.get(nid);
                        return nd && nd.hidden && (nodeTypeMap[nid] || '_Unknown') === cat;
                    }});
                    showSliderPanel(clickedId, catNodes, cat, color);
                }}
            }});
        }});
    }}

    function positionNodesAround(parentId, childIds) {{
        var positions = network.getPositions([parentId]);
        var parentPos = positions[parentId];
        if (!parentPos || childIds.length === 0) return;
        var radius    = Math.max(100, 100 + (childIds.length - 3) * 20);
        var angleStep = (2 * Math.PI) / childIds.length;
        var allNodes  = network.body.data.nodes;
        childIds.forEach(function(nid, i) {{
            var angle = -Math.PI / 2 + i * angleStep;
            var x = parentPos.x + radius * Math.cos(angle);
            var y = parentPos.y + radius * Math.sin(angle);
            allNodes.update({{id: nid, x: x, y: y}});
            network.moveNode(nid, x, y);
        }});
    }}

    function hideNodeAndCleanup(nodeId) {{
        var allNodes = network.body.data.nodes;
        if (expandedNodes[nodeId]) {{ collapseNode(nodeId); }}
        allNodes.update({{id: nodeId, hidden: true}});
        _restoreNodeEdgesAndHalo(nodeId);
        Object.keys(expandedNodes).forEach(function(parentId) {{
            var idx = expandedNodes[parentId].indexOf(nodeId);
            if (idx >= 0) {{
                expandedNodes[parentId].splice(idx, 1);
                if (expandedNodes[parentId].length === 0) {{ delete expandedNodes[parentId]; }}
            }}
        }});
    }}

    function collapseNode(nodeId) {{
        var revealed = expandedNodes[nodeId] || [];
        var allNodes = network.body.data.nodes;
        revealed.forEach(function(nid) {{
            var keep = false;
            Object.keys(expandedNodes).forEach(function(otherId) {{
                if (otherId !== nodeId && expandedNodes[otherId] && expandedNodes[otherId].indexOf(nid) >= 0) {{
                    keep = true;
                }}
            }});
            if (!keep) {{
                allNodes.update({{id: nid, hidden: true}});
                _restoreNodeEdgesAndHalo(nid);
                if (expandedNodes[nid]) {{ collapseNode(nid); }}
            }}
        }});
        delete expandedNodes[nodeId];
    }}
}})();
</script>
"""
