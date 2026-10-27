# =============================================================================
# graph_rag.py — Ontology-based Graph RAG Engine for MO360
#
# Implements a proper Retrieval-Augmented Generation pipeline that actively
# retrieves and enriches LLM prompt context from the live OWL ontology graph
# stored in Oxigraph.
#
# ─── SPARQL PIPELINE: OntologyGraphRAG ───────────────────────────────────────
#
#   Startup — build_ontology_index(store, prefix_map):
#     1. SPARQL-introspect classes, properties, and named instances from Oxigraph
#     2. Embed them via Carina embeddings → float[] vectors
#     3. Store in EmbeddingIndex for O(n) cosine-similarity retrieval
#
#   Query time — OntologyGraphRAG.retrieve(question):
#     1. Embed user question with Carina
#     2. k-NN cosine search → semantically relevant classes, properties + instances
#        (replaces hardcoded DE/EN keyword→class dict)
#     2b. k-NN cosine search → directly relevant OWL properties
#         (their domain/range classes become secondary seeds)
#     3. Live SPARQL against Oxigraph for each seed class:
#        - Fetch OWL-declared properties with domain/range
#        - Fetch sample instance URIs (live data evidence)
#        - Fetch sample triples showing how predicates are actually used
#        - Discover direct connectivity paths between seed classes
#     4. Assemble a dynamic context string built from LIVE graph data
#        (not a filtered slice of a pre-built static string)
#
#   What this adds BEYOND token reduction:
#     - Evidence grounding: LLM sees real triples, not just schema
#     - Semantic coverage: DE/EN/synonyms matched by embedding, not by dict
#     - Dynamic paths: property chains discovered per query, not hardcoded
#     - Instance context: actual URIs in context so LLM can anchor queries
#
# ─── EXPORTS ─────────────────────────────────────────────────────────────────
#
#   build_ontology_index(store, prefix_map) → EmbeddingIndex
#   OntologyGraphRAG(store, index, prefix_map)
#     .retrieve(question, static_context="", w=None) → GraphRAGContext
#
# ─── UTILITIES (also used by sql_panel.py for few-shot selection) ─────────────
#
#   _embed_texts(texts, w=None)  → list[list[float]]
#   _cosine_sim(a, b)            → float
#
# ─── THREADING ───────────────────────────────────────────────────────────────
#
#   retrieve() accepts an optional WorkspaceClient so callers inside daemon
#   threads can reuse the thread-local client (no extra OAuth round-trip).
#   When w=None, a fresh WorkspaceClient is created (safe for interactive use).
#
# ─── NOTE: pyoxigraph term API ───────────────────────────────────────────────
#
#   str(NamedNode("http://...")) returns "<http://...>" (N-Triples notation).
#   Use .value to get the plain URI string "http://..." instead.
#   This is critical for building valid SPARQL queries from retrieved URIs.
#
#   QuerySolution does NOT support .get() in pyoxigraph >= 0.4.x (incl. 0.5.x).
#   Use sol["variable"] instead — returns None for unbound OPTIONAL variables,
#   identical to the dict.get() behaviour.
#
#   BlankNode IDs are NOT valid IRIs.  Interpolating a blank-node .value into
#   a SPARQL angle-bracket literal — <{bnode.value}> — raises:
#       SyntaxError: expected IRI parsing failed
#   All index-building queries use FILTER(isIRI(?var)) to exclude blank nodes
#   so they never enter the EmbeddingIndex.  The runtime helpers
#   _properties_for_class() and _sample_instances() also guard with
#   `if not class_uri.startswith("http"): return []` as a second safety net.
#   Without these guards the except:pass blocks silently swallow the
#   SyntaxError, leaving all_props empty and forcing rag_applied=False.
# =============================================================================

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import pyoxigraph as ox


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDINGS_ENDPOINT = "MO360_Data_Platform_Nexus_Carina_3-0_Embeddings_Model_Endpoint"

_TOP_K_CLASSES    = 5   # semantic k-NN for ontology classes
_TOP_K_INSTANCES  = 3   # semantic k-NN for named instances
_TOP_K_PROPERTIES = 5   # semantic k-NN for OWL properties (direct match)
_MAX_PROPS_PER_CLASS = 8  # max OWL properties per seed class
_MAX_SAMPLE_TRIPLES  = 2  # sample triples per property (evidence)
_MAX_INSTANCES       = 3  # max instance URIs per class


# =============================================================================
# Embedding utilities
# =============================================================================

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_sim(a: list[float], b: list[float]) -> float:
    denom = _norm(a) * _norm(b)
    return _dot(a, b) / denom if denom > 0 else 0.0


def _embed_texts(texts: list[str], w=None) -> list[list[float]]:
    """Embed a list of texts using the Carina 3.0 embeddings endpoint.

    Uses direct HTTP (requests) with OpenAI-format {"input": [...]},
    since the Carina 3.0 endpoint does NOT accept the MLflow tensor
    format sent by w.serving_endpoints.query(inputs=...).

    Returns a list of float-vector embeddings (dim=1024), one per input.
    """
    import requests as _req
    if w is None:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
    cfg  = w.config
    url  = f"{cfg.host}/serving-endpoints/{EMBEDDINGS_ENDPOINT}/invocations"
    auth = cfg.authenticate()           # {'Authorization': 'Bearer <token>'}
    resp = _req.post(
        url,
        headers={**auth, "Content-Type": "application/json"},
        json={"input": texts},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]


# =============================================================================
# EmbeddingIndex — simple in-memory nearest-neighbour store
# =============================================================================

@dataclass
class IndexEntry:
    """A single indexed ontology element."""
    key:          str           # URI (ontology)
    label:        str           # Human-readable embed text
    element_type: str           # "class" | "property" | "instance"
    embedding:    list[float]
    metadata:     dict = field(default_factory=dict)


class EmbeddingIndex:
    """Flat in-memory embedding index with cosine-similarity k-NN.

    Stores IndexEntry objects.  query() performs a linear scan; the index
    is small (<200 entries) so this is fast enough (< 1 ms).
    """

    def __init__(self) -> None:
        self._entries: list[IndexEntry] = []

    def add(self, entry: IndexEntry) -> None:
        self._entries.append(entry)

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        element_types: list[str] | None = None,
    ) -> list[IndexEntry]:
        """Return the top-k most similar entries by cosine similarity."""
        candidates = (
            [e for e in self._entries if e.element_type in element_types]
            if element_types
            else self._entries
        )
        scored = [(e, _cosine_sim(embedding, e.embedding)) for e in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:k]]

    def __len__(self) -> int:
        return len(self._entries)


# =============================================================================
# Index builder (called once at startup, cached in session_state)
# =============================================================================

def build_ontology_index(
    store: ox.Store,
    prefix_map: dict[str, str],
    w=None,
) -> EmbeddingIndex:
    """Build an embedding index of ontology classes, properties, and instances.

    Introspects the live Oxigraph store via SPARQL, assembles natural-language
    descriptions for each element, and embeds them with Carina.  The resulting
    index enables semantic k-NN retrieval at query time.

    Returns an empty EmbeddingIndex (RAG disabled, legacy fallback active) if
    the embedding endpoint is unreachable so the app does not crash.

    Args:
        store:      The loaded pyoxigraph.Store.
        prefix_map: Dict {prefix_name: namespace_uri} from ontology_loader.
        w:          Optional WorkspaceClient (reused across calls).

    NOTE: pyoxigraph QuerySolution does not support .get() — use sol["var"]
    which returns None for unbound OPTIONAL variables.
    """
    def shorten(uri: str) -> str:
        for ns, pfx in sorted(
            ((v, k) for k, v in prefix_map.items()), key=lambda x: -len(x[0])
        ):
            if uri.startswith(ns):
                return f"{pfx}:{uri[len(ns):]}"
        return uri.split("#")[-1].split("/")[-1]

    prefix_block = "\n".join(f"PREFIX {k}: <{v}>" for k, v in prefix_map.items())
    items: list[tuple[str, str, str, dict]] = []  # (key, embed_text, type, meta)

    # ── 1. OWL Classes ───────────────────────────────────────────────────
    try:
        for sol in store.query(
            f"{prefix_block}\n"
            "SELECT DISTINCT ?cls ?label WHERE {\n"
            "  { ?cls a owl:Class } UNION { ?cls a rdfs:Class }\n"
            "  FILTER(isIRI(?cls))\n"
            "  OPTIONAL { ?cls rdfs:label ?label }\n"
            "} LIMIT 100"
        ):
            cls_node = sol["cls"]
            if cls_node is None:
                continue
            cls_uri = cls_node.value                          # .value not str()
            lv = sol["label"]                                 # None when unbound
            label = lv.value if lv is not None else shorten(cls_uri)
            short = shorten(cls_uri)
            embed_text = f"ontology class: {short} — represents a {label} in the MO360 production ontology"
            items.append((cls_uri, embed_text, "class", {"short": short}))
    except Exception as _exc:
        print(f"[graph_rag] build_ontology_index: classes SPARQL failed ({type(_exc).__name__}: {_exc})", file=sys.stderr)

    # ── 2. OWL Properties ────────────────────────────────────────────────
    try:
        for sol in store.query(
            f"{prefix_block}\n"
            "SELECT DISTINCT ?prop ?label ?domain ?range WHERE {\n"
            "  { ?prop a owl:ObjectProperty } UNION { ?prop a owl:DatatypeProperty }\n"
            "  FILTER(isIRI(?prop))\n"
            "  OPTIONAL { ?prop rdfs:label ?label }\n"
            "  OPTIONAL { ?prop rdfs:domain ?domain }\n"
            "  OPTIONAL { ?prop rdfs:range  ?range  }\n"
            "} LIMIT 200"
        ):
            prop_node = sol["prop"]
            if prop_node is None:
                continue
            prop_uri = prop_node.value                        # .value not str()
            lv  = sol["label"]                                # None when unbound
            dv  = sol["domain"]
            rv  = sol["range"]
            label = lv.value if lv is not None else shorten(prop_uri)
            dom   = shorten(dv.value) if dv is not None else "?"
            rng   = shorten(rv.value) if rv is not None else "?"
            embed_text = (
                f"ontology property: {shorten(prop_uri)} "
                f"connects {dom} to {rng} — {label}"
            )
            items.append((
                prop_uri, embed_text, "property",
                {"domain": dom, "range": rng, "short": shorten(prop_uri)},
            ))
    except Exception as _exc:
        print(f"[graph_rag] build_ontology_index: properties SPARQL failed ({type(_exc).__name__}: {_exc})", file=sys.stderr)

    # ── 3. Named individuals: Area, Plant (small, high-value instances) ───
    try:
        for sol in store.query(
            f"{prefix_block}\n"
            "SELECT DISTINCT ?inst ?type ?label WHERE {\n"
            "  ?inst a ?type .\n"
            "  FILTER(isIRI(?inst))\n"
            "  FILTER(?type IN (mrs:Area, mrs:Plant))\n"
            "  OPTIONAL { ?inst rdfs:label ?label }\n"
            "} LIMIT 50"
        ):
            inst_node = sol["inst"]
            if inst_node is None:
                continue
            inst_uri  = inst_node.value                       # .value not str()
            type_node = sol["type"]                           # None when unbound
            lv        = sol["label"]
            type_s    = shorten(type_node.value) if type_node is not None else "instance"
            label     = lv.value if lv is not None else shorten(inst_uri)
            embed_text = (
                f"{type_s} named {shorten(inst_uri)} — "
                f"a {type_s} area/plant in the MO360 production facility"
            )
            items.append((inst_uri, embed_text, "instance", {"type": type_s, "short": shorten(inst_uri)}))
    except Exception as _exc:
        print(f"[graph_rag] build_ontology_index: instances SPARQL failed ({type(_exc).__name__}: {_exc})", file=sys.stderr)

    if not items:
        print(
            "[graph_rag] build_ontology_index: no classes/properties/instances found "
            "— check SPARQL queries and prefix_map. Graph RAG disabled.",
            file=sys.stderr,
        )
        return EmbeddingIndex()

    # ── Batch-embed all element descriptions ─────────────────────────────
    texts = [t for _, t, _, _ in items]
    try:
        embeddings = _embed_texts(texts, w=w)
    except Exception as exc:
        # Embedding endpoint unavailable (e.g. NotFound, network error).
        # Return empty index so OntologyGraphRAG falls back to static context.
        print(
            f"[graph_rag] build_ontology_index: embedding failed "
            f"({type(exc).__name__}) — Graph RAG disabled, legacy fallback active.",
            file=sys.stderr,
        )
        return EmbeddingIndex()

    index = EmbeddingIndex()
    for (key, label, etype, meta), emb in zip(items, embeddings):
        index.add(IndexEntry(key=key, label=label, element_type=etype,
                             embedding=emb, metadata=meta))
    print(
        f"[graph_rag] build_ontology_index: indexed {len(index)} elements "
        f"({sum(1 for i in index._entries if i.element_type=='class')} classes, "
        f"{sum(1 for i in index._entries if i.element_type=='property')} properties, "
        f"{sum(1 for i in index._entries if i.element_type=='instance')} instances).",
        file=sys.stderr,
    )
    return index


# =============================================================================
# OntologyGraphRAG — SPARQL query context from live graph traversal
# =============================================================================

@dataclass
class GraphRAGContext:
    """Dynamic SPARQL context assembled from live Oxigraph graph traversal.

    Attributes:
        seed_shorts:        Prefixed class URIs that were used as seeds.
        relevant_properties: List of {short, domain, range} dicts from OWL.
        sample_instances:   List of "<class>: <prefixed_uri>" strings.
        sample_triples:     Live triples showing actual predicate usage.
        connectivity_paths: Discovered direct property chains between seeds.
        context_string:     Assembled context block for injection into the
                            LLM system prompt (replaces static schema_context).
        embedding_ms:       Carina embedding call latency in milliseconds.
        traversal_ms:       Oxigraph SPARQL traversal latency in milliseconds.
        rag_applied:        True if retrieval produced useful results.
        fallback_reason:    Non-empty string if full static context is used.
    """
    seed_shorts:          list[str]
    relevant_properties:  list[dict]
    sample_instances:     list[str]
    sample_triples:       list[str]
    connectivity_paths:   list[str]
    context_string:       str
    embedding_ms:         float = 0.0
    traversal_ms:         float = 0.0
    rag_applied:          bool  = True
    fallback_reason:      str   = ""


class OntologyGraphRAG:
    """Proper Graph RAG engine for SPARQL query generation.

    At query time:
      1. Embeds the question → cosine k-NN in the ontology index
      2. SPARQL-traverses the live Oxigraph store for each seed node:
         properties, sample instances, sample triples, connectivity paths
      3. Assembles a dynamic context string with retrieved graph data.

    This replaces the old apply_graph_rag() which only filtered a
    pre-built static string for token reduction.
    """

    def __init__(
        self,
        store:      ox.Store,
        index:      EmbeddingIndex,
        prefix_map: dict[str, str],
    ) -> None:
        self.store      = store
        self.index      = index
        self.prefix_map = prefix_map
        self._ns_sorted = sorted(
            ((v, k) for k, v in prefix_map.items()),
            key=lambda x: -len(x[0]),
        )  # [(namespace_uri, prefix_name), ...] longest-match order
        self._prefix_block = "\n".join(
            f"PREFIX {k}: <{v}>" for k, v in prefix_map.items()
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _shorten(self, uri: str) -> str:
        for ns, pfx in self._ns_sorted:
            if uri.startswith(ns):
                return f"{pfx}:{uri[len(ns):]}"
        return f"<{uri}>"

    def _q(self, sparql: str):
        """Execute a SPARQL SELECT against the store. Returns iterator."""
        return self.store.query(sparql)

    # ------------------------------------------------------------------
    # Live graph retrieval methods
    # ------------------------------------------------------------------

    def _properties_for_class(self, class_uri: str) -> list[dict]:
        """Fetch OWL-declared properties where this class is domain or range."""
        if not class_uri.startswith("http"):   # blank-node ID — not a valid IRI
            return []
        results: list[dict] = []
        try:
            for sol in self._q(
                f"{self._prefix_block}\n"
                "SELECT ?prop ?domain ?range WHERE {\n"
                f"  {{ ?prop rdfs:domain <{class_uri}> . BIND(<{class_uri}> AS ?domain) OPTIONAL {{ ?prop rdfs:range ?range }} }}\n"
                f"  UNION\n"
                f"  {{ ?prop rdfs:range  <{class_uri}> . BIND(<{class_uri}> AS ?range ) OPTIONAL {{ ?prop rdfs:domain ?domain }} }}\n"
                f"}} LIMIT {_MAX_PROPS_PER_CLASS * 3}"
            ):
                pv = sol["prop"]
                dv = sol["domain"]
                rv = sol["range"]
                if pv is None:
                    continue
                results.append({
                    "uri":   pv.value,
                    "short": self._shorten(pv.value),
                    "dom":   self._shorten(dv.value) if dv is not None else "?",
                    "rng":   self._shorten(rv.value) if rv is not None else "?",
                })
        except Exception:
            pass
        return results[:_MAX_PROPS_PER_CLASS]

    def _sample_instances(self, class_uri: str) -> list[str]:
        """Fetch sample instance URIs for a class from the live store."""
        if not class_uri.startswith("http"):   # blank-node ID — not a valid IRI
            return []
        instances: list[str] = []
        try:
            for sol in self._q(
                f"{self._prefix_block}\n"
                f"SELECT ?inst WHERE {{ ?inst rdf:type <{class_uri}> }} LIMIT {_MAX_INSTANCES}"
            ):
                iv = sol["inst"]
                if iv is not None:
                    instances.append(self._shorten(iv.value))
        except Exception:
            pass
        return instances

    def _sample_triples_for_property(self, prop_uri: str) -> list[str]:
        """Fetch sample subject/object pairs showing how a property is used."""
        triples: list[str] = []
        try:
            for sol in self._q(
                f"{self._prefix_block}\n"
                f"SELECT ?s ?o WHERE {{ ?s <{prop_uri}> ?o }} LIMIT {_MAX_SAMPLE_TRIPLES}"
            ):
                sv = sol["s"]
                ov = sol["o"]
                if sv is None or ov is None:
                    continue
                s_str = self._shorten(sv.value)
                # use isinstance to distinguish NamedNode (URI) from Literal
                o_str = (
                    self._shorten(ov.value)
                    if isinstance(ov, ox.NamedNode)
                    else f'"{ov.value}"'
                )
                short_p = self._shorten(prop_uri)
                triples.append(f"  {s_str}  {short_p}  {o_str} .")
        except Exception:
            pass
        return triples

    def _connectivity_paths(self, class_uris: list[str]) -> list[str]:
        """Discover direct OWL property chains between pairs of seed classes."""
        paths: list[str] = []
        for i, src in enumerate(class_uris):
            for dst in class_uris[i + 1:]:
                try:
                    for sol in self._q(
                        f"{self._prefix_block}\n"
                        "SELECT ?prop WHERE {\n"
                        f"  {{ ?prop rdfs:domain <{src}> ; rdfs:range <{dst}> }}\n"
                        f"  UNION\n"
                        f"  {{ ?prop rdfs:domain <{dst}> ; rdfs:range <{src}> }}\n"
                        "} LIMIT 3"
                    ):
                        pv = sol["prop"]
                        if pv is not None:
                            paths.append(
                                f"  {self._shorten(src)} ←→ {self._shorten(dst)}"
                                f"  via  {self._shorten(pv.value)}"
                            )
                except Exception:
                    pass
        return paths

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question:        str,
        static_context:  str = "",
        w=None,
    ) -> GraphRAGContext:
        """Retrieve a question-specific subgraph context from the live ontology.

        Pipeline:
          1. Embed the user question with Carina.
          2. k-NN cosine search → top-K ontology classes + named instances.
          3. SPARQL-traverse the Oxigraph store for each seed:
             properties, sample instances, sample triples, connectivity paths.
          4. Assemble a dynamic context_string for the LLM system prompt.

        Falls back to the static schema context if embedding or traversal
        produces no useful results (safety net).

        Args:
            question:       User's natural-language question.
            static_context: Full static schema context (fallback + prefix source).
            w:              Optional WorkspaceClient (reused in daemon threads).

        Returns:
            GraphRAGContext with assembled context_string.
        """
        # ── 0. Guard: empty index → fall back immediately ─────────────────
        if len(self.index) == 0:
            return GraphRAGContext(
                seed_shorts=[], relevant_properties=[], sample_instances=[],
                sample_triples=[], connectivity_paths=[],
                context_string=static_context,
                rag_applied=False,
                fallback_reason="Ontology index is empty — Graph RAG disabled.",
            )

        # ── 1. Embed question ─────────────────────────────────────────────
        emb_t0 = time.perf_counter()
        try:
            q_emb = _embed_texts([question], w=w)[0]
        except Exception as exc:
            return GraphRAGContext(
                seed_shorts=[], relevant_properties=[], sample_instances=[],
                sample_triples=[], connectivity_paths=[],
                context_string=static_context,
                rag_applied=False,
                fallback_reason=f"Embedding failed: {exc}",
            )
        emb_ms = (time.perf_counter() - emb_t0) * 1000

        # ── 2. Semantic k-NN retrieval ────────────────────────────────────
        top_classes    = self.index.query(q_emb, k=_TOP_K_CLASSES,     element_types=["class"])
        top_properties = self.index.query(q_emb, k=_TOP_K_PROPERTIES,  element_types=["property"])
        top_instances  = self.index.query(q_emb, k=_TOP_K_INSTANCES,   element_types=["instance"])

        seed_uris   = [e.key   for e in top_classes]
        seed_shorts = [e.metadata.get("short", e.key) for e in top_classes]

        # Properties matched directly by embedding may connect classes not
        # in the top-K.  Add their domain/range classes as secondary seeds
        # so _properties_for_class() and _connectivity_paths() cover them.
        for entry in top_properties:
            dom_short = entry.metadata.get("domain", "")
            rng_short = entry.metadata.get("range", "")
            # Resolve short→URI from the index for domain/range classes
            for e2 in self.index._entries:
                if e2.element_type == "class":
                    short2 = e2.metadata.get("short", "")
                    if short2 and short2 in (dom_short, rng_short):
                        if e2.key not in seed_uris:
                            seed_uris.append(e2.key)
                            seed_shorts.append(short2)

        if not seed_uris:
            return GraphRAGContext(
                seed_shorts=[], relevant_properties=[], sample_instances=[],
                sample_triples=[], connectivity_paths=[],
                context_string=static_context,
                rag_applied=False,
                fallback_reason="No seed classes found in index.",
            )

        # ── 3. Live Oxigraph traversal ────────────────────────────────────
        trav_t0 = time.perf_counter()

        all_props:     list[dict] = []
        all_instances: list[str]  = []
        all_triples:   list[str]  = []
        seen_prop_uris: set[str]  = set()

        # 3a. Inject directly-matched properties (from k-NN on property
        #     embeddings).  These are the highest-relevance properties the
        #     user asked about — they must appear in the context regardless
        #     of whether their domain class made it into the top-K classes.
        for entry in top_properties:
            p_uri = entry.key
            if p_uri not in seen_prop_uris:
                seen_prop_uris.add(p_uri)
                all_props.append({
                    "uri":   p_uri,
                    "short": entry.metadata.get("short", self._shorten(p_uri)),
                    "dom":   entry.metadata.get("domain", "?"),
                    "rng":   entry.metadata.get("range", "?"),
                })
                all_triples.extend(self._sample_triples_for_property(p_uri))

        # 3b. Class-driven property traversal (existing logic)
        for entry in top_classes:
            props = self._properties_for_class(entry.key)
            for p in props:
                if p["uri"] not in seen_prop_uris:
                    seen_prop_uris.add(p["uri"])
                    all_props.append(p)
                    # Live triple evidence for each property
                    all_triples.extend(self._sample_triples_for_property(p["uri"]))

            insts = self._sample_instances(entry.key)
            cls_s = entry.metadata.get("short", entry.key)
            all_instances.extend(f"{cls_s}: {i}" for i in insts)

        # Add named instances from semantic search
        for entry in top_instances:
            all_instances.append(
                f"{entry.metadata.get('type', 'instance')}: {entry.metadata.get('short', entry.key)}"
            )

        # Connectivity paths between seed classes
        conn_paths = self._connectivity_paths(seed_uris)
        trav_ms = (time.perf_counter() - trav_t0) * 1000

        # ── 4. Assemble dynamic context string ────────────────────────────
        lines: list[str] = []

        # Always include prefixes
        lines.append("=== PREFIXES ===")
        for pfx, ns in self.prefix_map.items():
            lines.append(f"PREFIX {pfx}: <{ns}>")
        lines.append("")

        # Semantically matched classes
        lines.append("=== RELEVANT CLASSES (semantically matched to question) ===")
        lines.append(", ".join(seed_shorts))
        lines.append("")

        # Properties retrieved live from OWL declarations
        if all_props:
            lines.append("=== PROPERTIES (live from OWL — name | Domain → Range) ===")
            for p in all_props:
                lines.append(f"  {p['short']}  |  {p['dom']} → {p['rng']}")
            lines.append("")

        # Discovered connectivity paths (dynamic, not hardcoded)
        if conn_paths:
            lines.append("=== CONNECTIVITY PATHS (discovered from OWL declarations) ===")
            lines.extend(conn_paths)
            lines.append("")

        # Live instance examples
        if all_instances:
            lines.append("=== INSTANCE EXAMPLES (live from Oxigraph store) ===")
            for inst in all_instances[:15]:
                lines.append(f"  {inst}")
            lines.append("")

        # Live sample triples — actual graph evidence
        if all_triples:
            lines.append("=== SAMPLE TRIPLES (live graph evidence for predicate usage) ===")
            for triple in all_triples[:12]:
                lines.append(triple)
            lines.append("")

        context_string = "\n".join(lines)

        # Safety: fall back to static context if retrieval yielded nothing useful
        if not all_props and not all_instances:
            return GraphRAGContext(
                seed_shorts=seed_shorts, relevant_properties=[], sample_instances=[],
                sample_triples=[], connectivity_paths=[],
                context_string=static_context,
                embedding_ms=emb_ms, traversal_ms=trav_ms,
                rag_applied=False,
                fallback_reason="Live traversal returned no properties or instances.",
            )

        return GraphRAGContext(
            seed_shorts=seed_shorts,
            relevant_properties=all_props,
            sample_instances=all_instances,
            sample_triples=all_triples,
            connectivity_paths=conn_paths,
            context_string=context_string,
            embedding_ms=emb_ms,
            traversal_ms=trav_ms,
            rag_applied=True,
        )
