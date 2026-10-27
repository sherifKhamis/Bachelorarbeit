# =============================================================================
# ontology_loader.py — Ontologie-Bootstrap für den AI Benchmark
#
# Lädt die produktive RDF/OWL-Ontologie aus einem Unity Catalog Volume
# und erzeugt einen strukturierten Schema-Kontext-String für die LLM-Prompts.
#
# ─── ARCHITEKTUR (rdflib-frei) ───────────────────────────────────────────────
#
#   pyoxigraph.Store   Einzige Engine — übernimmt sowohl SPARQL-Ausführung
#                      zur Laufzeit ALS AUCH Schema-Introspection beim Laden.
#                      Rust-native, 10–100× schneller als rdflib.
#
#   Prefix-Handling    @prefix-Deklarationen werden per Regex direkt aus dem
#                      TTL-Header extrahiert. Eigene get_prefixed_name()-
#                      Funktion via Longest-Match (kein rdflib namespace_manager).
#
# ─── PERFORMANCE (>300 MB sicher) ────────────────────────────────────────────
#
#   • Kein rdflib-Parsing: 400 MB TTL in ~15s statt 5–10 Min
#   • Speicher: nur Oxigraph Store im RAM (~1× Dateigröße statt 3–4×)
#   • O(1) Property-URI-Lookup via SPARQL statt Python-Full-Scan
#   • Early-Exit nach 20 Sample-Triples
#
# ─── RÜCKGABEWERT ────────────────────────────────────────────────────────────
#
#   4-Tuple: (None, schema_context: str, pyoxigraph.Store, stats: dict)
#   Das erste Element ist None (Rückwärtskompatibilität — graph wird nicht
#   mehr benötigt, da der Store alle Queries übernimmt).
#
# =============================================================================

import io
import re
import time

import streamlit as st
import pyoxigraph as ox

from databricks.sdk import WorkspaceClient

from config import VOLUME_PATH


# =============================================================================
# Prefix extraction & URI shortening (replaces rdflib namespace_manager)
# =============================================================================

def _extract_prefixes(ttl_bytes: bytes) -> dict[str, str]:
    """Extract @prefix declarations from TTL header bytes.

    Scans the first 20 KB of the file for lines like:
        @prefix mrs: <http://example.org/mrs#> .
        PREFIX mrs: <http://example.org/mrs#>

    Returns:
        dict mapping prefix (without colon) → namespace URI string.
        Example: {"mrs": "http://example.org/mrs#", "rdf": "http://..."}
    """
    # Read first 20 KB — all prefixes are at the top of well-formed TTL
    header = ttl_bytes[:20_480].decode("utf-8", errors="ignore")

    prefix_map: dict[str, str] = {}
    # Matches both "@prefix pfx: <uri> ." and "PREFIX pfx: <uri>"
    pattern = re.compile(
        r'(?:@prefix|PREFIX)\s+(\w*)\s*:\s*<([^>]+)>', re.IGNORECASE
    )
    for match in pattern.finditer(header):
        prefix = match.group(1)
        namespace = match.group(2)
        if prefix:  # Skip empty/default prefix
            prefix_map[prefix] = namespace

    # Add standard prefixes if not already present
    _STANDARD = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
    }
    for pfx, ns in _STANDARD.items():
        if pfx not in prefix_map:
            prefix_map[pfx] = ns

    return prefix_map


def _build_prefixed_name_fn(prefix_map: dict[str, str]):
    """Build a get_prefixed_name(uri) function using longest-match.

    Returns a closure that converts full URI strings to prefixed form
    (e.g., "http://example.org/mrs#Checkpoint" → "mrs:Checkpoint").
    """
    # Sort namespaces by length descending for longest-match
    ns_to_prefix = sorted(
        [(ns, pfx) for pfx, ns in prefix_map.items()],
        key=lambda x: len(x[0]),
        reverse=True,
    )

    def get_prefixed_name(uri) -> str:
        """Convert a URI (str or ox.NamedNode) to prefixed form."""
        if isinstance(uri, ox.NamedNode):
            uri_str = uri.value
        elif isinstance(uri, str):
            uri_str = uri
        else:
            return str(uri)

        for namespace, prefix in ns_to_prefix:
            if uri_str.startswith(namespace):
                local = uri_str[len(namespace):]
                return f"{prefix}:{local}"
        return f"<{uri_str}>"

    return get_prefixed_name


# =============================================================================
# Oxigraph SPARQL helpers
# =============================================================================

def _sparql_select(store: ox.Store, query: str) -> list[dict[str, ox.NamedNode]]:
    """Run a SELECT query and return results as list of dicts."""
    result = store.query(query)
    rows = []
    for solution in result:
        row = {}
        for var in result.variables:
            val = solution[var]
            if val is not None:
                row[str(var.value)] = val
        rows.append(row)
    return rows


def _sparql_count(store: ox.Store, query: str) -> int:
    """Run a COUNT query and return the integer result."""
    result = store.query(query)
    for solution in result:
        for var in result.variables:
            val = solution[var]
            if val is not None:
                return int(val.value)
    return 0


# =============================================================================
# Module-level constants for schema introspection
# =============================================================================

# Standard RDF/OWL prefixes excluded from schema context (always assumed known).
_STANDARD_PREFIXES: frozenset[str] = frozenset({
    "rdf", "rdfs", "owl", "xsd", "xml", "dc", "dcterms", "skos",
})

# OWL meta-types that represent schema structure, not data instances.
# Excluded from instance_types so only data classes appear in schema context.
_SCHEMA_TYPES: frozenset[str] = frozenset({
    "owl:Class", "owl:ObjectProperty", "owl:DatatypeProperty",
    "owl:Ontology", "owl:NamedIndividual", "rdf:Property", "rdfs:Class",
    "owl:AllDisjointClasses", "owl:AnnotationProperty",
    "owl:FunctionalProperty", "owl:TransitiveProperty",
})


def _compact_uri(name: str, max_len: int = 35) -> str:
    """Shorten a prefixed URI to at most max_len characters.

    Keeps the prefix and the tail of the local name to stay readable.
    Example: ``mrs:..kpointWithLongLocalName`` for a 60-char prefixed URI.
    """
    if ':' in name and len(name) > max_len:
        prefix, local = name.split(':', 1)
        return f"{prefix}:..{local[-(max_len - len(prefix) - 4):]}"
    return name


# =============================================================================
# Schema introspection helpers
# =============================================================================

def _build_schema_context(
    store: ox.Store,
    prefix_map: dict[str, str],
) -> tuple[str, set[str], set[str], dict[str, list]]:
    """Build the LLM schema context string via SPARQL introspection on the Store.

    Queries the Oxigraph store for classes, properties (with domain/range),
    sample triples for undocumented predicates, and one instance example per
    data class.  Assembles all fragments into a single schema context string
    that is injected into every LLM system prompt.

    Args:
        store:      Loaded pyoxigraph.Store.
        prefix_map: Dict of prefix → namespace URI extracted from the TTL header.

    Returns:
        schema_summary: Structured multi-section string for LLM prompts.
        classes:        Set of prefixed class names.
        properties:     Set of prefixed property names.
        instance_types: Dict class-name → [example_uri] (max 1 per class).
    """
    get_prefixed_name = _build_prefixed_name_fn(prefix_map)

    # ── Prefix declarations ───────────────────────────────────────────────
    prefix_declarations = ""
    for pfx, ns in sorted(prefix_map.items()):
        if pfx not in _STANDARD_PREFIXES:
            prefix_declarations += f"PREFIX {pfx}: <{ns}>\n"
    prefix_declarations += "# Standard prefixes (rdf:, rdfs:, owl:, xsd:) are also available.\n"

    # ── OWL/RDFS class collection ─────────────────────────────────────────
    classes: set[str] = set()
    for row in _sparql_select(store, """
        SELECT DISTINCT ?c WHERE {
            { ?c a <http://www.w3.org/2002/07/owl#Class> }
            UNION
            { ?c a <http://www.w3.org/2000/01/rdf-schema#Class> }
        }
    """):
        c = row.get("c")
        if c and isinstance(c, ox.NamedNode):
            classes.add(get_prefixed_name(c))

    # ── Property collection with domain / range lookup ────────────────────
    properties: set[str] = set()
    prop_uri_map: dict[str, ox.NamedNode] = {}
    for row in _sparql_select(store, """
        SELECT DISTINCT ?p WHERE {
            { ?p a <http://www.w3.org/2002/07/owl#ObjectProperty> }
            UNION
            { ?p a <http://www.w3.org/2002/07/owl#DatatypeProperty> }
            UNION
            { ?p a <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property> }
        }
    """):
        p = row.get("p")
        if p and isinstance(p, ox.NamedNode):
            pn = get_prefixed_name(p)
            properties.add(pn)
            prop_uri_map[pn] = p

    property_details: list[str] = []
    _fully_documented_props: set[str] = set()
    for prop in sorted(properties):
        prop_node = prop_uri_map.get(prop)
        if not prop_node:
            property_details.append(f"  {prop}")
            continue
        uri_str = prop_node.value
        domains = [
            get_prefixed_name(r["d"])
            for r in _sparql_select(store, f"SELECT ?d WHERE {{ <{uri_str}> <http://www.w3.org/2000/01/rdf-schema#domain> ?d }}")
            if "d" in r and isinstance(r["d"], ox.NamedNode)
        ]
        ranges = [
            get_prefixed_name(r["r"])
            for r in _sparql_select(store, f"SELECT ?r WHERE {{ <{uri_str}> <http://www.w3.org/2000/01/rdf-schema#range> ?r }}")
            if "r" in r and isinstance(r["r"], ox.NamedNode)
        ]
        if domains and ranges:
            property_details.append(
                f"  {prop} | {', '.join(domains)} → {', '.join(ranges)}"
            )
            _fully_documented_props.add(prop)
        elif domains or ranges:
            d = ', '.join(domains) if domains else '?'
            r = ', '.join(ranges) if ranges else '?'
            property_details.append(f"  {prop} | {d} → {r}")
        else:
            property_details.append(f"  {prop}")

    # ── Sample triples for undocumented predicates (max 20) ──────────────
    sample_triples: list[str] = []
    seen_preds: set[str] = set()
    rdf_type_node = ox.NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    for quad in store.quads_for_pattern(None, None, None, None):
        if len(sample_triples) >= 20:
            break
        s, p, o = quad.subject, quad.predicate, quad.object
        if not isinstance(s, ox.NamedNode) or not isinstance(p, ox.NamedNode):
            continue
        if p == rdf_type_node:
            continue
        p_name = get_prefixed_name(p)
        if p_name in _fully_documented_props or p_name in seen_preds:
            continue
        if isinstance(o, ox.NamedNode):
            o_name = _compact_uri(get_prefixed_name(o))
        elif isinstance(o, ox.Literal):
            val = o.value
            o_name = f'"{ val[:30]}"' if len(val) > 30 else f'"{val}"'
        else:
            continue
        s_name = _compact_uri(get_prefixed_name(s))
        seen_preds.add(p_name)
        sample_triples.append(f"  {s_name} {p_name} {o_name} .")

    # ── One instance example per data class ───────────────────────────────
    instance_types: dict[str, list] = {}
    for row in _sparql_select(store, "SELECT ?s ?o WHERE { ?s a ?o }"):
        s, o = row.get("s"), row.get("o")
        if not (isinstance(s, ox.NamedNode) and isinstance(o, ox.NamedNode)):
            continue
        o_name = get_prefixed_name(o)
        if o_name in _SCHEMA_TYPES:
            continue
        if o_name not in instance_types:
            instance_types[o_name] = []
        if len(instance_types[o_name]) < 1:
            instance_types[o_name].append(_compact_uri(get_prefixed_name(s)))

    instance_summary = "".join(
        f"  {cls}: {', '.join(instances)}\n"
        for cls, instances in sorted(instance_types.items())
    )

    # ── Assemble schema context string ────────────────────────────────────
    schema_summary = (
        f"=== PREFIXES (prepend to every query) ===\n"
        f"{prefix_declarations}\n"
        f"=== CLASSES ===\n"
        f"{', '.join(sorted(classes)) if classes else 'None detected'}\n"
        f"\n"
        f"=== PROPERTIES (name | Domain → Range) ===\n"
        f"{chr(10).join(property_details) if property_details else '  None detected'}\n"
        f"\n"
        f"=== INSTANCE EXAMPLES (class: URI pattern) ===\n"
        f"{instance_summary if instance_summary else '  No typed instances found'}\n"
        f"\n"
        f"=== SAMPLE TRIPLES (only for undocumented predicates) ===\n"
        f"{chr(10).join(sample_triples) if sample_triples else '  None needed (all predicates documented)'}\n"
    )

    return schema_summary, classes, properties, instance_types


def _collect_stats(
    store: ox.Store,
    classes: set[str],
    properties: set[str],
    instance_types: dict[str, list],
    file_size: int,
    schema_context_chars: int,
) -> dict:
    """Compute ontology statistics for the UI header card.

    Counts triples via SPARQL COUNT and assembles the stats dict used by
    app_header.render_app_header().

    Args:
        store:                 pyoxigraph.Store (to run the triple count query).
        classes:               Set of prefixed class names from _build_schema_context.
        properties:            Set of prefixed property names from _build_schema_context.
        instance_types:        Dict class-name → examples from _build_schema_context.
        file_size:             Raw TTL file size in bytes.
        schema_context_chars:  Character count of the assembled schema context string.

    Returns:
        Dict with keys: num_triples, num_classes, num_properties,
        num_instances, num_instance_types, file_size_mb, schema_context_chars.
    """
    num_triples = _sparql_count(store, "SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o }")
    return {
        "num_triples":          num_triples,
        "num_classes":          len(classes),
        "num_properties":       len(properties),
        "num_instances":        sum(len(v) for v in instance_types.values()),
        "num_instance_types":   len(instance_types),
        "file_size_mb":         file_size / (1024 * 1024),
        "schema_context_chars": schema_context_chars,
    }


# =============================================================================
# Main loader
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_ontology_and_schema() -> tuple[None, str, ox.Store, dict, dict]:
    """Lade, parse und analysiere die Produktions-Ontologie (rdflib-frei).

    Pipeline:
      1. Download TTL aus UC Volume (1 MB Chunks, Fortschrittsbalken)
      2. Extrahiere @prefix-Deklarationen per Regex
      3. Parse in pyoxigraph.Store (einzige Engine)
      4. Schema-Introspection via SPARQL auf dem Store
      5. Baue Schema-Kontext-String für LLM-Prompt

    Returns:
        None:           Platzhalter (rdflib.Graph nicht mehr benötigt).
        schema_summary: Strukturierter Kontext-String für LLM-Prompts.
        store:          pyoxigraph.Store — Rust-native SPARQL-Engine.
        stats:          Dict mit Ontologie-Statistiken für UI-Übersicht.
    """
    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Download + Parse
    # ══════════════════════════════════════════════════════════════════════
    w = WorkspaceClient()
    response = w.files.download(VOLUME_PATH)
    file_size = response.content_length

    with st.status("Ontologie wird geladen…", expanded=True) as status:

        # ── Download TTL aus UC Volume ─────────────────────────────────
        t0 = time.perf_counter()
        st.write("⬇️  TTL-Datei aus Unity Catalog Volume laden…")
        progress_bar = st.progress(0)
        buffer = io.BytesIO()
        downloaded = 0
        chunk_size = 1024 * 1024

        for chunk in iter(lambda: response.contents.read(chunk_size), b""):
            buffer.write(chunk)
            downloaded += len(chunk)
            progress_bar.progress(min(downloaded / file_size, 1.0))

        dl_time = time.perf_counter() - t0
        progress_bar.empty()
        st.write(
            f"✅  Download abgeschlossen — "
            f"**{file_size / (1024*1024):.1f} MB** in **{dl_time:.1f}s**"
        )

        # ── Oxigraph Store laden (einzige Engine) ─────────────────────
        t1 = time.perf_counter()
        st.write("⚡  Oxigraph SPARQL-Engine initialisieren…")
        buffer.seek(0)
        ttl_bytes = buffer.read()
        del buffer  # Sofort freigeben

        store = ox.Store()
        store.load(ttl_bytes, "text/turtle")
        ox_time = time.perf_counter() - t1
        st.write(f"✅  Oxigraph bereit — **{ox_time:.1f}s**")

        # ── Prefixes extrahieren ──────────────────────────────────────
        st.write("🔍  Schema-Kontext erstellen…")
        prefix_map = _extract_prefixes(ttl_bytes)
        del ttl_bytes  # Bytes freigeben — Store hält die Daten

        status.update(
            label=f"Ontologie geladen · {time.perf_counter() - t0:.1f}s",
            state="complete",
            expanded=False,
        )

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Schema introspection + stats
    # ══════════════════════════════════════════════════════════════════════
    schema_summary, classes, properties, instance_types = _build_schema_context(store, prefix_map)
    stats = _collect_stats(store, classes, properties, instance_types, file_size, len(schema_summary))

    return None, schema_summary, store, stats, prefix_map
