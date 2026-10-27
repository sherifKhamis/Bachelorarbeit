# =============================================================================
# sparql_generator.py — LLM-powered natural-language → SPARQL translation
#
# Responsibilities:
#   1. Accept a user question and the ontology schema context (built by
#      ontology_loader.py) as inputs.
#   2. Apply Graph-RAG retrieval to build a question-specific subgraph context.
#   3. Sanitize user input against prompt injection (via prompt_guard.py).
#   4. Send retrieved context + question to the configured LLM endpoint with a
#      structured system prompt that teaches the model the ontology vocabulary.
#   5. Post-process the raw LLM response:
#      a) Strip markdown fences (```sparql ... ```)
#      b) Auto-inject missing PREFIX declarations via _ensure_prefixes()
#      c) Validate output via prompt_guard.validate_sparql_output()
#   6. Return the SPARQL query string + wall-clock latency + token metrics.
#
# Graph-RAG pipeline (OntologyGraphRAG — see graph_rag.py):
#   1. Embed user question with Carina → cosine k-NN against ontology index
#   2. Live SPARQL traversal of Oxigraph for each seed class:
#      - OWL-declared properties (domain/range)  → replaces hardcoded TRAVERSAL PATHS
#      - Named Area/Plant instances               → replaces hardcoded TERMINOLOGY
#      - Sample triples (evidence)
#      - Connectivity paths between seed classes
#   Legacy fallback (apply_graph_rag): keyword→class mapping + static text
#   filtering, used when ont_rag=None (e.g. index not yet built).
#
# Prompt Injection Defense (3 layers, via prompt_guard.py):
#   Layer 1: Input sanitization — blocks instruction overrides, role reassignment,
#            system prompt extraction, code injection, delimiter attacks
#   Layer 2: Output validation — only SELECT/CONSTRUCT/DESCRIBE/ASK allowed,
#            SERVICE clause blocked, SPARQL Update forms blocked
#   Layer 3: Sandwich defense — security reminder appended to system prompt
#
# System prompt structure:
#   The _SYSTEM_PROMPT_TEMPLATE contains:
#   - {schema_context}: injected at runtime — dynamic Graph-RAG context
#     (or full/filtered static context in legacy fallback mode).
#     Includes PREFIXES, RELEVANT CLASSES, PROPERTIES, CONNECTIVITY PATHS,
#     INSTANCE EXAMPLES, and SAMPLE TRIPLES — all retrieved live.
#   - RULES: 10 generation constraints (type anchors, STRENDS, LIMIT, etc.)
#   - EXAMPLES: 3 reference queries (area filter, VIN traversal, UNION)
#   - ANTI-PATTERNS: common LLM mistakes to avoid
#   - SECURITY REMINDER: sandwich defense suffix (always enforce)
#
# Token breakdown (returned in 6th element of generate_sparql() tuple):
#   - schema_context_tokens: filtered context size (after Graph-RAG)
#   - instructions_tokens: rules + examples + anti-patterns + security suffix
#     + user-message prefix ("Translate this request to SPARQL: ")
#   - user_message_tokens: raw user question only (no prefix;
#     consistent with sql_panel.user_message_tokens)
#   - graph_rag_applied: bool — whether filtering was applied
#   - graph_rag_reduction_pct: float — percentage token savings vs. full context
#   - injection_blocked: bool — True if input was rejected (only on block)
#
# Benchmark parity note:
#   This module is the SPARQL counterpart to sql_panel.generate_sql_query().
#   Both accept an optional model_config dict so the same model is used for
#   both panels in a given benchmark run.
#
# Threading architecture:
#   Each LLM call creates a FRESH WorkspaceClient() inside a daemon thread.
#   This ensures authentication, HTTP connect, and the API call are ALL
#   covered by the timeout budget.  Thread death closes the connection.
#   See the detailed architecture decision comment in llm_client.py.
# =============================================================================

import time

from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

from config import LLM_ENDPOINT
from llm_client import call_llm_with_timeout
from prompt_guard import sanitize_user_input, validate_sparql_output, sandwich_suffix_sparql
from utils import estimate_token_count

# ---------------------------------------------------------------------------
# LLM call timeout (seconds).  Fallback for interactive (non-benchmark) calls
# where no model_config is provided.  Matches GPT-4o's configured budget so
# interactive queries don't silently wait 120s on endpoint issues.
# Benchmark runs always use model_config["llm_timeout_s"] (30s / 120s).
# ---------------------------------------------------------------------------
_LLM_TIMEOUT_S = 30


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def strip_prefixes_for_display(query: str) -> str:
    """Return the SPARQL query with PREFIX lines removed (display only)."""
    lines = query.strip().splitlines()
    return "\n".join(
        line for line in lines if not line.strip().upper().startswith("PREFIX")
    ).strip()


def _ensure_prefixes(query: str, schema_context: str) -> str:
    """Prepend any PREFIX declarations from schema_context missing in the query.

    LLMs sometimes omit PREFIX lines, causing pyoxigraph to fail with
    'Prefix not found' parse errors.  This extracts all PREFIX lines from
    the schema context and injects those not already present in the query.
    """
    import re

    # Extract PREFIX declarations from schema_context
    context_prefixes: dict[str, str] = {}  # prefix_name -> full PREFIX line
    for line in schema_context.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("PREFIX"):
            match = re.match(r"PREFIX\s+(\S+:)\s+<([^>]+)>", stripped, re.IGNORECASE)
            if match:
                context_prefixes[match.group(1).lower()] = stripped

    # Also ensure standard prefixes are available
    _STANDARD = {
        "rdf:": "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
        "rdfs:": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
        "owl:": "PREFIX owl: <http://www.w3.org/2002/07/owl#>",
        "xsd:": "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>",
    }
    for k, v in _STANDARD.items():
        if k not in context_prefixes:
            context_prefixes[k] = v

    # Determine which prefixes are already declared in the query
    declared: set[str] = set()
    for line in query.splitlines():
        match = re.match(r"PREFIX\s+(\S+:)", line.strip(), re.IGNORECASE)
        if match:
            declared.add(match.group(1).lower())

    # Determine which prefixes are actually used in the query body
    body = strip_prefixes_for_display(query)
    used_prefixes = set(re.findall(r"\b(\w+):", body))

    # Inject missing but used prefixes
    missing_lines: list[str] = []
    for prefix_key, prefix_line in sorted(context_prefixes.items()):
        short = prefix_key.rstrip(":")
        if prefix_key not in declared and short in used_prefixes:
            missing_lines.append(prefix_line)

    if missing_lines:
        return "\n".join(missing_lines) + "\n" + query
    return query


def is_construct_query(query: str) -> bool:
    """Return True if the query is a CONSTRUCT (graph-building) query."""
    return strip_prefixes_for_display(query).strip().upper().startswith("CONSTRUCT")


# ---------------------------------------------------------------------------
# Token estimation helper
# ---------------------------------------------------------------------------

def _estimate_token_count(text: str) -> int:
    """Estimate token count for a text string.

    Uses tiktoken (cl100k_base) if available, otherwise falls back to
    a character-based heuristic (len / 4).
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except (ImportError, Exception):
        # Fallback: ~4 chars per token for English/mixed text
        return max(1, len(text) // 4)




# ---------------------------------------------------------------------------
# Graph-RAG: Relevance-based schema context filtering
#
# Instead of sending the full ontology schema to the LLM (~3000+ tokens),
# Graph-RAG extracts only the relevant subgraph based on the user's question:
#   1. Check for multi-hop keywords → if found, use full context (too risky)
#   2. Map keywords from the user question to ontology classes (seed classes)
#   3. Filter schema_context sections using SEED classes only
#
# Why seeds-only (no BFS)?
#   - The ontology is dense: mrs:Fault has 7 neighbors, VIN has 3.
#   - BFS from any seed reaches ≥80% of classes in 1 hop → no filtering.
#   - Seeds-only gives 38-77% token reduction while keeping needed properties.
#
# Tested: 10/10 correctness tests pass (all needed properties preserved).
# Effect: 38-77% schema token reduction, falls back for multi-hop/unknown.
# ---------------------------------------------------------------------------

import re as _re

# German → English class keyword mapping (legacy fallback only)
_KEYWORD_TO_CLASS = {
    # VIN / Fahrzeug
    "vin": "mrs:VIN", "fahrzeug": "mrs:VIN", "fin": "mrs:VIN", "auto": "mrs:VIN",
    "vehicle": "mrs:VIN", "wagen": "mrs:VIN",
    # Fault / Fehler
    "fehler": "mrs:Fault", "fault": "mrs:Fault", "mangel": "mrs:Fault",
    "defekt": "mrs:Fault", "problem": "mrs:Fault",
    # FaultType
    "fehlerart": "mrs:FaultType", "fehlertyp": "mrs:FaultType", "faulttype": "mrs:FaultType",
    # Checkpoint
    "checkpoint": "mrs:Checkpoint", "prüfpunkt": "mrs:Checkpoint", "station": "mrs:Checkpoint",
    "checkpoints": "mrs:Checkpoint",
    # Hall
    "halle": "mrs:Hall", "hall": "mrs:Hall", "hallen": "mrs:Hall",
    # Area
    "bereich": "mrs:Area", "area": "mrs:Area", "gewerk": "mrs:Area",
    "montage": "mrs:Area", "assembly": "mrs:Area",
    "lackiererei": "mrs:Area", "paintshop": "mrs:Area", "lackierung": "mrs:Area",
    "karosseriebau": "mrs:Area", "bodyshop": "mrs:Area", "rohbau": "mrs:Area",
    # Plant
    "werk": "mrs:Plant", "plant": "mrs:Plant", "fabrik": "mrs:Plant",
    # ModelType
    "fahrzeugtyp": "mrs:ModelType", "modeltype": "mrs:ModelType", "modell": "mrs:ModelType",
    "baureihe": "mrs:ModelType",
    # Category / Mark
    "kategorie": "mrs:Category", "category": "mrs:Category",
    "markierung": "mrs:Mark", "mark": "mrs:Mark",
    # FaultVehicleLocation
    "fahrzeugposition": "mrs:FaultVehicleLocation", "vehiclelocation": "mrs:FaultVehicleLocation",
    "position": "mrs:FaultVehicleLocation",
}

# Spatial/fault keywords that imply multi-hop traversal (always include full chain)
_CHAIN_KEYWORDS = {
    "verursacht", "caused", "erkannt", "detected", "nachgearbeitet", "reworked",
    "produktionsweg", "spatial", "hierarchy", "kette", "chain", "vollständig",
    "über alle", "alle bereiche",
}



def _extract_relevant_classes(user_question: str) -> set[str]:
    """Map keywords from the user question to ontology class URIs."""
    question_lower = user_question.lower()
    relevant: set[str] = set()

    for keyword, cls in _KEYWORD_TO_CLASS.items():
        if keyword in question_lower:
            relevant.add(cls)

    # If VIN number detected, always include VIN class
    if _re.search(r"W1[A-Z0-9]{14,}", user_question):
        relevant.add("mrs:VIN")

    return relevant



def _is_multi_hop_question(user_question: str) -> bool:
    """Detect if the question likely requires deep multi-hop traversal."""
    question_lower = user_question.lower()
    return any(kw in question_lower for kw in _CHAIN_KEYWORDS)


def _filter_schema_context(schema_context: str, relevant_classes: set[str]) -> str:
    """Filter schema_context to only include relevant classes, properties, instances.

    Matching strategy per section:
      PROPERTIES:      exact token match (line split on whitespace) so that
                       e.g. ``mrs:Fault`` does NOT accidentally match lines
                       that only reference ``mrs:FaultType``.
      INSTANCE EXAMPLES: ``cls + ":"`` matches the class label on the left
                       (``mrs:Fault: mrs:Fault_12345``); ``cls + "_"`` matches
                       any instance URI on the right side.
      SAMPLE TRIPLES:  ``cls + "_"`` matches instance subject/object tokens
                       (format ``mrs:ClassName_identifier``) without triggering
                       on property names like ``mrs:hasFault``.
    """
    sections = {
        "prefixes": [],
        "classes": [],
        "properties": [],
        "instances": [],
        "samples": [],
    }

    current_section = None
    for line in schema_context.splitlines():
        if "=== PREFIXES" in line:
            current_section = "prefixes"
            sections["prefixes"].append(line)
        elif "=== CLASSES" in line:
            current_section = "classes"
            sections["classes"].append(line)
        elif "=== PROPERTIES" in line:
            current_section = "properties"
            sections["properties"].append(line)
        elif "=== INSTANCE EXAMPLES" in line:
            current_section = "instances"
            sections["instances"].append(line)
        elif "=== SAMPLE TRIPLES" in line:
            current_section = "samples"
            sections["samples"].append(line)
        elif current_section:
            sections[current_section].append(line)

    # Always include prefixes fully
    result_lines = sections["prefixes"][:]

    # Filter classes: only include relevant ones
    result_lines.append("=== CLASSES ===")
    class_line = sections["classes"][1] if len(sections["classes"]) > 1 else ""
    if class_line:
        all_classes = [c.strip() for c in class_line.split(",")]
        filtered_classes = [c for c in all_classes if c in relevant_classes]
        if filtered_classes:
            result_lines.append(", ".join(filtered_classes))
        else:
            result_lines.append(class_line)  # Fallback: keep all
    result_lines.append("")

    # Filter properties: keep those where domain OR range is relevant
    result_lines.append("=== PROPERTIES (name | Domain → Range) ===")
    for line in sections["properties"][1:]:
        if not line.strip():
            continue
        # Exact token match: split the line so that mrs:Fault does not
        # accidentally match a line that only contains mrs:FaultType.
        line_tokens = set(line.split())
        if any(cls in line_tokens for cls in relevant_classes) or "|" not in line_tokens:
            result_lines.append(line)
    result_lines.append("")

    # Filter instances: keep those whose class is relevant
    result_lines.append("=== INSTANCE EXAMPLES (class: URI pattern) ===")
    for line in sections["instances"][1:]:
        if not line.strip():
            continue
        # Use cls+":": for the class label on the left side of the entry
        # (e.g. "mrs:Fault: mrs:Fault_12345") and cls+"_" for instance URIs
        # on the right side.  Both guards prevent mrs:Fault from matching
        # mrs:FaultType entries.
        if any((cls + ":") in line or (cls + "_") in line for cls in relevant_classes):
            result_lines.append(line)
    result_lines.append("")

    # Filter sample triples: keep those mentioning relevant classes
    result_lines.append("=== SAMPLE TRIPLES (only for undocumented predicates) ===")
    for line in sections["samples"][1:]:
        if not line.strip():
            continue
        # Check for cls+"_" so we match instance URIs (mrs:Fault_12345) but
        # NOT property names that contain the class name (mrs:hasFault).
        if any((cls + "_") in line for cls in relevant_classes):
            result_lines.append(line)
    result_lines.append("")

    return "\n".join(result_lines)


def apply_graph_rag(user_question: str, schema_context: str) -> tuple[str, bool]:
    """Apply Graph-RAG filtering to reduce schema context size.

    Strategy: Extract SEED classes from keywords in the user's question,
    then filter the schema context to only include properties/instances
    mentioning those classes. No BFS expansion needed.

    Args:
        user_question: The user's natural-language question.
        schema_context: The full schema context from ontology_loader.

    Returns:
        filtered_context: The (potentially reduced) schema context.
        was_filtered: True if Graph-RAG was applied, False if full context used.
    """
    # Multi-hop questions need full context (too risky to filter)
    if _is_multi_hop_question(user_question):
        return schema_context, False

    # Extract relevant classes from the question
    seed_classes = _extract_relevant_classes(user_question)

    # If we can't identify any classes, use full context
    if not seed_classes:
        return schema_context, False

    # Apply filtering using SEED classes only.
    # No coverage check needed: seeds are always 1-3 classes (9-27% of 11),
    # so filtering always gives meaningful reduction (40-85%).
    # The BFS coverage check was removed because mrs:Fault (7 neighbors)
    # caused BFS to hit ≥80% for every Fault-related question, blocking
    # filtering even though seeds-only would still cut 40-60% of tokens.
    filtered = _filter_schema_context(schema_context, seed_classes)
    return filtered, True


# ---------------------------------------------------------------------------
# LLM prompt — system instructions
#
# Two templates:
#   _SYSTEM_PROMPT_RAG_TEMPLATE  — minimal core rules (~580 tok), used when
#                                   OntologyGraphRAG provides dynamic context.
#   _SYSTEM_PROMPT_TEMPLATE      — full legacy prompt (~2300 tok + schema_context),
#                                   used as fallback when RAG is unavailable.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_RAG_TEMPLATE = """\
You are a SPARQL expert for an RDF/OWL ontology. Translate the user's question \
into a single efficient SPARQL query.

{rag_context}

=== RULES ===
0. GROUNDING: Only use mrs: properties and classes that appear explicitly in
   the schema context above or in the EXAMPLES. A name absent from the context
   DOES NOT exist in the ontology — never invent it. If the required concept
   has no matching property in the context, output the cannot-answer query
   (see UNANSWERABLE QUESTIONS below).
1. Output ONLY the raw SPARQL — no markdown fences, no explanation.
2. Use EXACT prefixed names from the context above — names are CASE-SENSITIVE.
3. Anchor every query with a type constraint: ?x rdf:type mrs:SomeClass.
4. Use STRENDS for VIN filtering: FILTER(STRENDS(STR(?vin), "VIN_VALUE")).
5. Prefer CONSTRUCT for relationship/structure questions; SELECT for aggregations.
6. In CONSTRUCT queries, include rdf:type triples so nodes are correctly typed.
7. Use OPTIONAL only for genuinely optional data — never wrap required patterns.
8. NEVER add LIMIT on SELECT queries — always return the full dataset so
   users can count and aggregate all results. For CONSTRUCT queries, always
   add LIMIT 100 to prevent browser freeze from thousands of triples.
9. For combined queries (Checkpoints + Faults for same VIN): use UNION to keep
   independent subgraph patterns separate. Place FILTER and rdf:type anchor
   BEFORE the UNION blocks.
10. CHECKPOINT ORDER — ALWAYS use mrs:locationProgress, NEVER ?checkpointName:
    Bind in WHERE: ?cp mrs:locationProgress ?locationProgress .
    a) Sorted list ("sortiert", "geordnet", "Reihenfolge", "in Reihenfolge"):
         ORDER BY ASC(?locationProgress)
    b) First/earliest ("erste", "erster", "erste Station", "first", "earliest"):
         ORDER BY ASC(?locationProgress)
         Multiple checkpoints may share the same minimum locationProgress (ties).
         Return ALL of them — no LIMIT. The first rows of the result are the answer.
    c) Last/latest ("letzte", "letzter", "letzte Station", "last", "latest"):
         ORDER BY DESC(?locationProgress)
         Multiple checkpoints may share the same maximum locationProgress (ties).
         Return ALL of them — no LIMIT. The first rows of the result are the answer.
    NEVER use ORDER BY ASC(?checkpointName) for production-order questions.
11. ALWAYS use SELECT DISTINCT — never plain SELECT — to eliminate duplicate rows.
    Duplicates arise when the same variable is reachable via multiple RDF paths
    (e.g. a VIN linked to a Hall via both Checkpoints and Faults). Without DISTINCT
    result counts inflate — sometimes by 2× or more — producing wrong totals.
    Exception: CONSTRUCT queries are inherently deduplicated; DISTINCT does not apply.
12. FAULT HALL PREDICATE — choose based on the question's intent:
    - "entdeckt" / "detected" / "gefunden" / "erkannt"  → mrs:isDetectedInHall / mrs:isDetectedAtCheckpoint
    - "verursacht" / "caused" / "entstanden"            → mrs:isCausedInHall / mrs:isCausedAtCheckpoint
    - "nachgearbeitet" / "reworked" / "repariert"       → mrs:isReworkedInHall / mrs:isReworkedAtCheckpoint
    When the question says ONLY "Fehler im [Area]" without specifying detected/caused,
    default to mrs:isDetectedInHall (where the fault was FOUND, not where it originated).
    NEVER blindly default to isCausedInHall — read the question keywords carefully.

=== AREA vs CATEGORY — SPATIAL HIERARCHY ONLY USES AREA ===
# mrs:Area = spatial production area (named individuals — use EXACT URIs):
#   Montage / Endmontage / Assembly / Gewerk Montage  →  mrs:Assembly
#   Karosseriebau / Rohbau / Bodyshop / Gewerk Rohbau →  mrs:Bodyshop
#   Lackiererei / Lack / Paintshop / Gewerk Lack       →  mrs:Paintshop
# Spatial path (Checkpoint→Hall→Area):
#   ?cp mrs:isInHall ?hall . ?hall mrs:isInArea mrs:Assembly
# Spatial paths (Fault→Hall→Area) — THREE predicates, choose by question intent (Rule 12):
#   "entdeckt/detected/gefunden"        → ?fault mrs:isDetectedInHall ?hall . ?hall mrs:isInArea ?area
#   "verursacht/caused/entstanden"      → ?fault mrs:isCausedInHall ?hall . ?hall mrs:isInArea ?area
#   "nachgearbeitet/reworked/repariert" → ?fault mrs:isReworkedInHall ?hall . ?hall mrs:isInArea ?area
#   Default (no keyword): mrs:isDetectedInHall (where fault was FOUND)
# ModelType path:
#   ?modelType mrs:isProducedInArea mrs:Paintshop
#
# mrs:Category = NON-spatial concept (product release categories / Freigabestatus).
#   mrs:isReworkedInCategory stores release status, NOT a location.
#   Use mrs:Category ONLY when the question is explicitly about
#   Freigabekategorien or product release status — NEVER for spatial questions.

=== AREA EXAMPLE A: All checkpoints in an Area (sorted by production order) ===
User: "Pruefpunkte der Montagelinie mit Hallenzuordnung nach Fortschritt sortiert"

SELECT DISTINCT ?checkpointName ?hallName ?locationProgress WHERE {{
  ?cp rdf:type mrs:Checkpoint ;
      mrs:isInHall ?hall ;
      mrs:locationProgress ?locationProgress .
  ?hall mrs:isInArea mrs:Assembly .
  BIND(STRAFTER(STR(?cp),   "#") AS ?checkpointName)
  BIND(STRAFTER(STR(?hall), "#") AS ?hallName)
}} ORDER BY ASC(?locationProgress)

=== AREA EXAMPLE B: First/last station in an Area (superlative — ties possible, no LIMIT) ===
User: "Assembly-Startpunkt: Pruefpunkt mit geringstem location_progress-Wert"

SELECT DISTINCT ?hallName ?checkpointName ?locationProgress WHERE {{
  ?cp rdf:type mrs:Checkpoint ;
      mrs:isInHall ?hall ;
      mrs:locationProgress ?locationProgress .
  ?hall mrs:isInArea mrs:Assembly .
  BIND(STRAFTER(STR(?cp),   "#") AS ?checkpointName)
  BIND(STRAFTER(STR(?hall), "#") AS ?hallName)
}} ORDER BY ASC(?locationProgress)

=== AREA EXAMPLE C: Last checkpoint name/description in an Area (superlative DESC — "Checkpoint-Beschreibung" = URI local-part via STRAFTER, ties possible, no LIMIT) ===
User: "Letzter Paintshop-Pruefpunkt: URI-Name und Halle bei maximalem Fortschritt"

SELECT DISTINCT ?checkpointName ?hallName ?locationProgress WHERE {{
  ?cp rdf:type mrs:Checkpoint ;
      mrs:isInHall ?hall ;
      mrs:locationProgress ?locationProgress .
  ?hall mrs:isInArea mrs:Paintshop .
  BIND(STRAFTER(STR(?cp),   "#") AS ?checkpointName)
  BIND(STRAFTER(STR(?hall), "#") AS ?hallName)
}} ORDER BY DESC(?locationProgress)

=== UNANSWERABLE QUESTIONS ===
If the question refers to a concept absent from this ontology, output EXACTLY:
SELECT ?note WHERE {{ BIND("Diese Frage kann mit der verf\u00fcgbaren Ontologie nicht beantwortet werden." AS ?note) }} LIMIT 1

Trigger this when:
- A required mrs: property or class is NOT shown in the schema context above.
- The user asks about: Preis / Farbe / Lieferdatum / Kundendaten /
  Freitext-Fehlerbeschreibung / Lagerbestand / Produktionsplan /
  Schichtzeiten / Sachnummern / Maschinendaten / Temperaturmessung.
- "Checkpoint-Beschreibung" / checkpoint name IS answerable via
  BIND(STRAFTER(STR(?cp), "#") AS ?checkpointName) — see AREA EXAMPLE C.
  NEVER trigger unanswerable for checkpoint name/description questions.
  Only trigger when the concept truly has no representation in the ontology.

=== ANTI-PATTERNS (NEVER DO THESE) ===
- ?s ?p ?o wildcard patterns → full-graph scan, timeout
- Missing rdf:type anchor → scans entire graph
- Stations + Faults in one WHERE without UNION → M×N cross-product, timeout
- CONSTRUCT without LIMIT → thousands of triples, browser freeze
- Invented property names → 0 results, silent failure
- Invented Area URIs (mrs:Montage, mrs:Lack, mrs:Karosserie) → 0 results.
  Use ONLY: mrs:Assembly / mrs:Bodyshop / mrs:Paintshop
- CRITICAL: Do NOT use mrs:Category / mrs:hasCategory for ANY spatial question
  (area, bereich, gewerk, produktionsbereich, montage, lackiererei, karosseriebau,
  rohbau, lack, assembly, bodyshop, paintshop). Category is NOT spatial.
  For spatial questions ALWAYS use: ?cp mrs:isInHall ?hall . ?hall mrs:isInArea ?area .
- mrs:isCausedInArea / mrs:isDetectedInArea / mrs:isReworkedInArea DO NOT EXIST.
  Always derive area via 2-hop (choose predicate per Rule 12):
    ?fault mrs:isDetectedInHall ?hall . ?hall mrs:isInArea ?area .   (detected — DEFAULT)
    ?fault mrs:isCausedInHall ?hall . ?hall mrs:isInArea ?area .     (caused)
    ?fault mrs:isReworkedInHall ?hall . ?hall mrs:isInArea ?area .   (reworked)
- mrs:isReworkedInCategory is NOT a rework-location property (stores release status).
  Use FILTER EXISTS with mrs:isReworkedInHall for real rework (~245 faults).
- mrs:hasDescription does NOT exist on Checkpoints. rdfs:label exists ONLY on
  SOME Checkpoint types (H46PC_, H46PDA_, H56PC_, etc.) — NOT on H46AT_, H46SPS_,
  H46_IS_ and similar types (0 label triples for those). Using ?cp rdfs:label ?name
  silently returns 0 rows for VINs that pass through unlabelled Checkpoints.
  For any query involving passedThroughCheckpoint or a checkpoint name, ALWAYS use:
    BIND(STRAFTER(STR(?cp), "#") AS ?checkpointName)
- mrs:hasVINs / mrs:affectsVIN / mrs:hasAffectedVIN are OWL schema-only (0 instance triples).
  Oxigraph does NOT evaluate owl:inverseOf — use the MATERIALIZED direction instead:
    mrs:hasVINs        → use: ?vin mrs:belongsToModelType ?modelType .
    mrs:affectsVIN     → use: ?vin mrs:hasFault ?fault .
    mrs:hasAffectedVIN → use: ?vin mrs:belongsToModelType ?modelType .
- Hall instances have NO rdfs:label — ?hall rdfs:label ?label → 0 rows (verified: 0 triples).
  Filter hall by name: FILTER(STRAFTER(STR(?hall), "#") = "44")
- Checkpoints have NO mrs:isInArea. For area-filtered checkpoint queries use 2-hop:
  ?cp mrs:isInHall ?hall . ?hall mrs:isInArea mrs:Bodyshop
  (NOT ?cp mrs:isInArea mrs:Bodyshop — returns 0 rows)
- mrs:hasMark takes a URI NamedNode, NOT a string literal.
  Correct: mrs:hasMark mrs:RZ1 — NEVER: mrs:hasMark "RZ1"
- SELECT without DISTINCT → duplicate rows and inflated COUNT(*). ALWAYS: SELECT DISTINCT.
  Exception: CONSTRUCT queries are inherently deduplicated.

=== UNION PATTERN (multi-subgraph) ===
CONSTRUCT {{
  ?vin rdf:type mrs:VIN .
  ?vin mrs:passedThroughCheckpoint ?cp .
  ?cp  rdf:type mrs:Checkpoint .
  ?cp  mrs:isInHall ?hall .
  ?hall rdf:type mrs:Hall .
  ?vin mrs:hasFault ?fault .
  ?fault rdf:type mrs:Fault .
  ?fault mrs:hasType ?faultType .
}}
WHERE {{
  ?vin rdf:type mrs:VIN .
  FILTER(STRENDS(STR(?vin), "VIN_VALUE"))
  {{
    ?vin mrs:passedThroughCheckpoint ?cp .
    ?cp  mrs:isInHall ?hall .
  }}
  UNION
  {{
    ?vin mrs:hasFault ?fault .
    OPTIONAL {{ ?fault mrs:hasType ?faultType . }}
  }}
}}
LIMIT 100
""" + sandwich_suffix_sparql()


_SYSTEM_PROMPT_TEMPLATE = """\
You are a SPARQL expert for an RDF/OWL ontology. Translate the user's question \
into a single efficient SPARQL query.

{schema_context}

=== RULES ===
0. GROUNDING: Only use mrs: properties and classes that appear explicitly in
   the schema context above or in the EXAMPLES. A name absent from the context
   DOES NOT exist in the ontology — never invent it. If the required concept
   has no matching property in the context, output the cannot-answer query
   (see UNANSWERABLE QUESTIONS below).
1. Output ONLY the raw SPARQL — no markdown fences, no explanation.
2. Use EXACT prefixed names from the schema — names are CASE-SENSITIVE.
3. Anchor every query with a type constraint: ?x rdf:type mrs:SomeClass.
4. Use STRENDS for VIN filtering: FILTER(STRENDS(STR(?vin), "VIN_VALUE")).
5. Prefer CONSTRUCT for relationship/structure questions; SELECT for aggregations.
6. In CONSTRUCT queries, include rdf:type triples so nodes are correctly typed.
7. Use OPTIONAL only for genuinely optional data — never wrap required patterns.
8. NEVER add LIMIT on SELECT queries — always return the full dataset so
   users can count and aggregate all results. For CONSTRUCT queries, always
   add LIMIT 100 to prevent browser freeze from thousands of triples.
9. For combined queries (e.g. Checkpoints + Faults for same VIN): use UNION
   to keep independent subgraph patterns separate and avoid cross-products.
   Place the FILTER and rdf:type anchor BEFORE the UNION blocks.
10. CHECKPOINT ORDER — ALWAYS use mrs:locationProgress, NEVER ?checkpointName:
    Always fetch in WHERE: ?cp mrs:locationProgress ?locationProgress .
    a) Sorted list ("sortiert", "geordnet", "Reihenfolge", "in Reihenfolge"):
         ORDER BY ASC(?locationProgress)
    b) First/earliest ("erste", "erster", "erste Station", "first", "earliest"):
         ORDER BY ASC(?locationProgress)
         Multiple checkpoints may share the same minimum locationProgress (ties).
         Return ALL of them — no LIMIT. The first rows of the result are the answer.
    c) Last/latest ("letzte", "letzter", "letzte Station", "last", "latest"):
         ORDER BY DESC(?locationProgress)
         Multiple checkpoints may share the same maximum locationProgress (ties).
         Return ALL of them — no LIMIT. The first rows of the result are the answer.
    NEVER use ORDER BY ASC(?checkpointName) for production-order questions.
11. ALWAYS use SELECT DISTINCT — never plain SELECT — to eliminate duplicate rows.
    Duplicates arise when the same variable is reachable via multiple RDF paths
    (e.g. a VIN linked to a Hall via both Checkpoints and Faults). Without DISTINCT
    result counts inflate — sometimes by 2× or more — producing wrong totals.
    Exception: CONSTRUCT queries are inherently deduplicated; DISTINCT does not apply.
12. FAULT HALL PREDICATE — choose based on the question's intent:
    - "entdeckt" / "detected" / "gefunden" / "erkannt"  → mrs:isDetectedInHall / mrs:isDetectedAtCheckpoint
    - "verursacht" / "caused" / "entstanden"            → mrs:isCausedInHall / mrs:isCausedAtCheckpoint
    - "nachgearbeitet" / "reworked" / "repariert"       → mrs:isReworkedInHall / mrs:isReworkedAtCheckpoint
    When the question says ONLY "Fehler im [Area]" without specifying detected/caused,
    default to mrs:isDetectedInHall (where the fault was FOUND, not where it originated).
    NEVER blindly default to isCausedInHall — read the question keywords carefully.

=== AREA vs CATEGORY — SPATIAL HIERARCHY ONLY USES AREA ===
# mrs:Area = spatial production area. Named individuals (EXACT URIs, no variants):
#   Montage / Endmontage / Assembly / Gewerk Montage  →  mrs:Assembly
#   Karosseriebau / Rohbau / Bodyshop / Gewerk Rohbau →  mrs:Bodyshop
#   Lackiererei / Lack / Paintshop / Gewerk Lack       →  mrs:Paintshop
# Spatial paths:
#   Checkpoint→Hall→Area:  ?cp mrs:isInHall ?hall . ?hall mrs:isInArea mrs:Assembly
#   Fault→Hall→Area (3 predicates — choose by question intent, see Rule 12):
#     "entdeckt/detected"        → ?fault mrs:isDetectedInHall ?hall . ?hall mrs:isInArea ?area
#     "verursacht/caused"         → ?fault mrs:isCausedInHall ?hall . ?hall mrs:isInArea ?area
#     "nachgearbeitet/reworked"   → ?fault mrs:isReworkedInHall ?hall . ?hall mrs:isInArea ?area
#     Default (no keyword): mrs:isDetectedInHall (where fault was FOUND)
#   ModelType→Area:        ?modelType mrs:isProducedInArea mrs:Paintshop
#
# mrs:Category = NON-spatial concept (product release categories / Freigabestatus).
#   mrs:isReworkedInCategory stores release status — NOT a production location.
#   NEVER use mrs:Category / mrs:hasCategory for any spatial or production area question.
#
# ModelType properties (connect mrs:VIN / mrs:Fault / mrs:Checkpoint to a model):
#   mrs:belongsToModelType   |  mrs:VIN       → mrs:ModelType
#   mrs:affectedModelTypes   |  mrs:Fault     → mrs:ModelType
#   mrs:isProducedInArea     |  mrs:ModelType → mrs:Area
#   mrs:isProducedInHall     |  mrs:ModelType → mrs:Hall
#   mrs:isProducedInPlant    |  mrs:ModelType → mrs:Plant
#   mrs:hasModelName         |  mrs:ModelType → xsd:string  (human-readable name)

=== EXAMPLE 1: Faults in a specific Area ===
User: "Show faults and their checkpoints in the Bodyshop"

CONSTRUCT {{
  ?fault rdf:type mrs:Fault .
  ?fault mrs:hasType ?faultType .
  ?fault mrs:isCausedAtCheckpoint ?cp .
  ?cp    rdf:type mrs:Checkpoint .
  ?cp    mrs:isInHall ?hall .
  ?cp    mrs:locationProgress ?progress .
  ?hall  mrs:isInArea ?area .
}}
WHERE {{
  ?fault rdf:type mrs:Fault .
  ?fault mrs:isCausedAtCheckpoint ?cp .
  ?cp    mrs:isInHall ?hall .
  ?hall  mrs:isInArea mrs:Bodyshop .
  ?cp    mrs:locationProgress ?progress .
  OPTIONAL {{ ?fault mrs:hasType ?faultType . }}
  BIND(mrs:Bodyshop AS ?area)
}}
LIMIT 50

=== EXAMPLE 2: VIN traversal with full spatial hierarchy ===
User: "Show all checkpoints for VIN W1NKM5BB3TU135231 including mark, hall, area, and plant as graph"

CONSTRUCT {{
  ?vin   rdf:type mrs:VIN .
  ?vin   mrs:passedThroughCheckpoint ?cp .
  ?cp    rdf:type mrs:Checkpoint .
  ?cp    mrs:hasMark ?mark .
  ?cp    mrs:isInHall ?hall .
  ?hall  rdf:type mrs:Hall .
  ?hall  mrs:isInArea ?area .
  ?area  rdf:type mrs:Area .
  ?area  mrs:isInPlant ?plant .
  ?plant rdf:type mrs:Plant .
}}
WHERE {{
  ?vin rdf:type mrs:VIN .
  FILTER(STRENDS(STR(?vin), "W1NKM5BB3TU135231"))
  ?vin mrs:passedThroughCheckpoint ?cp .
  ?cp  mrs:isInHall ?hall .
  ?hall mrs:isInArea ?area .
  ?area mrs:isInPlant ?plant .
  OPTIONAL {{ ?cp mrs:hasMark ?mark . }}
}}
LIMIT 50

=== EXAMPLE 3: Combined VIN checkpoints + faults (UNION pattern) ===
User: "Show all checkpoints for VIN W1NKM5BB3TU135231 with mark, hall, area, plant — also show faults with where they were caused and detected, fault type and vehicle position. As graph."

CONSTRUCT {{
  ?vin   rdf:type mrs:VIN .
  ?vin   mrs:passedThroughCheckpoint ?cp .
  ?cp    rdf:type mrs:Checkpoint .
  ?cp    mrs:hasMark ?mark .
  ?cp    mrs:isInHall ?hall .
  ?hall  rdf:type mrs:Hall .
  ?hall  mrs:isInArea ?area .
  ?area  rdf:type mrs:Area .
  ?area  mrs:isInPlant ?plant .
  ?plant rdf:type mrs:Plant .
  ?vin   mrs:hasFault ?fault .
  ?fault rdf:type mrs:Fault .
  ?fault mrs:hasType ?faultType .
  ?fault mrs:occursAtVehicleLocation ?loc .
  ?fault mrs:isCausedAtCheckpoint ?causedCp .
  ?fault mrs:isCausedInHall ?causedHall .
  ?fault mrs:isDetectedAtCheckpoint ?detectedCp .
  ?fault mrs:isDetectedInHall ?detectedHall .
}}
WHERE {{
  ?vin rdf:type mrs:VIN .
  FILTER(STRENDS(STR(?vin), "W1NKM5BB3TU135231"))
  {{
    ?vin mrs:passedThroughCheckpoint ?cp .
    ?cp  mrs:isInHall ?hall .
    ?hall mrs:isInArea ?area .
    ?area mrs:isInPlant ?plant .
      OPTIONAL {{ ?cp mrs:hasMark ?mark . }}
  }}
  UNION
  {{
    ?vin mrs:hasFault ?fault .
    OPTIONAL {{ ?fault mrs:hasType ?faultType . }}
    OPTIONAL {{ ?fault mrs:occursAtVehicleLocation ?loc . }}
    OPTIONAL {{ ?fault mrs:isCausedAtCheckpoint ?causedCp . }}
    OPTIONAL {{ ?fault mrs:isCausedInHall ?causedHall . }}
    OPTIONAL {{ ?fault mrs:isDetectedAtCheckpoint ?detectedCp . }}
    OPTIONAL {{ ?fault mrs:isDetectedInHall ?detectedHall . }}
  }}
}}
LIMIT 100

=== EXAMPLE 4: Distinct ModelTypes in a specific Area ===
User: "Paintshop-Baureihen: alle ModelType-Instanzen mit Lackiererei-Produktionszuordnung"

SELECT DISTINCT ?modelType ?name WHERE {{
  ?modelType rdf:type mrs:ModelType ;
             mrs:isProducedInArea mrs:Paintshop .
  OPTIONAL {{ ?modelType mrs:hasModelName ?name . }}
}} ORDER BY ?modelType

=== EXAMPLE 5: ModelTypes by Plant assignment (isProducedInPlant) ===
User: "ModelType-Instanzen mit Werkszuordnung ueber isProducedInPlant abfragen"

SELECT DISTINCT ?modelType ?name ?plant WHERE {{
  ?modelType rdf:type mrs:ModelType ;
             mrs:isProducedInPlant ?plant .
  OPTIONAL {{ ?modelType mrs:hasModelName ?name . }}
}} ORDER BY ?modelType

=== EXAMPLE 6: Marks in a specific Hall and Area ===
User: "Alle Mark-URIs der Halle 15 im Bodyshop-Bereich ermitteln"

SELECT DISTINCT ?mark WHERE {{
  ?hall rdf:type mrs:Hall .
  FILTER(STRAFTER(STR(?hall), "#") = "15")
  ?hall mrs:isInArea  mrs:Bodyshop .
  ?hall mrs:marksInHall ?mark .
}}

=== EXAMPLE 7: Checkpoints filtered by Mark and Area ===
User: "Alle Pruefpunkte mit Mark-Zuordnung LKD in Paintshop-Hallen"

SELECT DISTINCT ?checkpointName ?hallName WHERE {{
  ?cp rdf:type mrs:Checkpoint ;
      mrs:hasMark  mrs:LKD ;
      mrs:isInHall ?hall .
  ?hall mrs:isInArea mrs:Paintshop .
  BIND(STRAFTER(STR(?cp), "#") AS ?checkpointName)
  BIND(STRAFTER(STR(?hall), "#") AS ?hallName)
}} ORDER BY ASC(?checkpointName)

=== EXAMPLE 8: Last checkpoint in an Area (superlative DESC, ties, no LIMIT) ===
User: "Letzter Paintshop-Pruefpunkt: URI-Name und Halle bei maximalem Fortschritt"

SELECT DISTINCT ?checkpointName ?hallName ?locationProgress WHERE {{
  ?cp rdf:type mrs:Checkpoint ;
      mrs:isInHall ?hall ;
      mrs:locationProgress ?locationProgress .
  ?hall mrs:isInArea mrs:Paintshop .
  BIND(STRAFTER(STR(?cp),   "#") AS ?checkpointName)
  BIND(STRAFTER(STR(?hall), "#") AS ?hallName)
}} ORDER BY DESC(?locationProgress)

=== UNANSWERABLE QUESTIONS ===
If the question refers to a concept absent from this ontology, output EXACTLY:
SELECT ?note WHERE {{ BIND("Diese Frage kann mit der verf\u00fcgbaren Ontologie nicht beantwortet werden." AS ?note) }} LIMIT 1

Trigger this when:
- A required mrs: property or class is NOT shown in the schema context above.
- The user asks about: Preis / Farbe / Lieferdatum / Kundendaten /
  Freitext-Fehlerbeschreibung / Lagerbestand / Produktionsplan /
  Schichtzeiten / Sachnummern / Maschinendaten / Temperaturmessung.
- "Checkpoint-Beschreibung" / checkpoint name IS answerable via
  BIND(STRAFTER(STR(?cp), "#") AS ?checkpointName) — see AREA EXAMPLE C.
  NEVER trigger unanswerable for checkpoint name/description questions.
  Only trigger when the concept truly has no representation in the ontology.

=== ANTI-PATTERNS (NEVER DO THESE) ===
- ?s ?p ?o wildcard patterns → full-graph scan, timeout
- Missing rdf:type anchor → scans entire graph
- Stations + Faults in one WHERE without UNION → M×N cross-product, timeout
- CONSTRUCT without LIMIT → thousands of triples, browser freeze
- Invented property names → 0 results, silent failure
- Invented Area URIs (mrs:Montage, mrs:Lack, mrs:Karosserie, mrs:Assembly_area, etc.) → 0 results.
  Use ONLY the three named individuals defined in AREA TERMINOLOGY above:
  mrs:Assembly / mrs:Bodyshop / mrs:Paintshop
- CRITICAL: Do NOT use mrs:Category / mrs:hasCategory for ANY spatial question
  (area, bereich, gewerk, produktionsbereich, montage, lackiererei, karosseriebau,
  rohbau, lack, assembly, bodyshop, paintshop). Category is NOT spatial — it
  stores product release status (Freigabestatus). For spatial questions ALWAYS use:
  ?cp mrs:isInHall ?hall . ?hall mrs:isInArea mrs:Assembly   (or Bodyshop/Paintshop)
- mrs:isCausedInArea / mrs:isDetectedInArea / mrs:isReworkedInArea DO NOT EXIST.
  Always derive area via 2-hop (choose predicate per Rule 12):
    ?fault mrs:isDetectedInHall ?hall . ?hall mrs:isInArea ?area .   (detected — DEFAULT)
    ?fault mrs:isCausedInHall ?hall . ?hall mrs:isInArea ?area .     (caused)
    ?fault mrs:isReworkedInHall ?hall . ?hall mrs:isInArea ?area .   (reworked)
- mrs:hasDescription does NOT exist on Checkpoint individuals. rdfs:label exists
  only on SOME Checkpoint name-types (H46PC_, H46PDA_, H56PC_) — e.g.
  "H46_Q-LIVE_RÜCKMELDUNG". Checkpoint types H46AT_, H46SPS_, H46_IS_ and
  similar have NO rdfs:label (verified: 0 triples). ?cp rdfs:label ?name silently
  returns 0 rows for VINs whose checkpoints lack labels. ALWAYS use:
    BIND(STRAFTER(STR(?cp), "#") AS ?checkpointName)
  This is robust for ALL checkpoint types.
- mrs:hasVINs / mrs:affectsVIN / mrs:hasAffectedVIN are OWL schema-only (0 instance triples).
  Oxigraph does NOT evaluate owl:inverseOf — use the MATERIALIZED direction instead:
    mrs:hasVINs        → use: ?vin mrs:belongsToModelType ?modelType .
    mrs:affectsVIN     → use: ?vin mrs:hasFault ?fault .
    mrs:hasAffectedVIN → use: ?vin mrs:belongsToModelType ?modelType .
- Hall instances have NO rdfs:label — ?hall rdfs:label ?label → 0 rows (verified: 0 triples).
  Filter hall by name: FILTER(STRAFTER(STR(?hall), "#") = "44")
- Checkpoints have NO mrs:isInArea. For area-filtered checkpoint queries use 2-hop:
  ?cp mrs:isInHall ?hall . ?hall mrs:isInArea mrs:Bodyshop
  (NOT ?cp mrs:isInArea mrs:Bodyshop — returns 0 rows)
- mrs:hasMark takes a URI NamedNode, NOT a string literal.
  Correct: mrs:hasMark mrs:RZ1 — NEVER: mrs:hasMark "RZ1"
- SELECT without DISTINCT → duplicate rows and inflated COUNT(*). ALWAYS: SELECT DISTINCT.
  Exception: CONSTRUCT queries are inherently deduplicated.
""" + sandwich_suffix_sparql()


# ---------------------------------------------------------------------------
# LLM-Aufruf — delegiert an llm_client.call_llm_with_timeout()
#
# Architekturentscheidung: Frischer WorkspaceClient pro Thread.
# Details siehe llm_client.py Modul-Header.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_sparql(
    user_prompt: str,
    schema_context: str,
    model_config: dict | None = None,
    ont_rag=None,
) -> tuple[str, float, int, int, int, dict]:
    """Translate a natural-language question into a SPARQL query.

    Pipeline: sanitize input → Graph-RAG filter → LLM call → validate output.

    Args:
        user_prompt:    The user's natural-language question.
        schema_context: The ontology schema summary produced by
                        ontology_loader.load_ontology_and_schema().
        model_config:   Optional model config dict from config.MODELS.
                        Must have keys: "endpoint", "supports_temperature".
                        Defaults to GPT-4o (LLM_ENDPOINT, temperature=0.0).

    Returns:
        sparql_query:      The raw SPARQL string ready for execution.
                           May be a safe "blocked" query if injection detected.
        elapsed:           Wall-clock seconds for the LLM round-trip (0 if blocked).
        prompt_tokens:     Input tokens reported by the serving endpoint.
        completion_tokens: Output tokens reported by the serving endpoint.
        system_prompt_tokens: Local tiktoken estimate of the full system prompt.
        token_breakdown:   Dict with consistent local estimates:
                           - schema_context_tokens: filtered context (after Graph-RAG)
                           - instructions_tokens: rules + examples + security suffix
                           - user_message_tokens: the user's question
                           - total_prompt_tokens: sum of all input components
                           - output_tokens: locally estimated output size
                           - api_prompt_tokens: raw API-reported value
                           - api_completion_tokens: raw API-reported value
                           - graph_rag_applied: bool (was context filtered?)
                           - graph_rag_reduction_pct: float (% token savings)
                           - injection_blocked: bool (only present if blocked)
                           - output_blocked: bool (only if output validation failed)
    """
    cfg      = model_config or {"endpoint": LLM_ENDPOINT, "supports_temperature": True}
    endpoint = cfg["endpoint"]
    timeout  = cfg.get("llm_timeout_s", _LLM_TIMEOUT_S)

    # ── Prompt Injection Guard: sanitize input ─────────────────────────────
    clean_prompt, is_blocked, block_reason = sanitize_user_input(user_prompt)
    if is_blocked:
        # Return a safe "cannot answer" query instead of processing the injection
        safe_query = (
            'PREFIX mrs: <http://22_Ont_Fahrzeugfehler_2025_V1#>\n'
            'SELECT ?hinweis WHERE {\n'
            f'  BIND("Anfrage blockiert: {block_reason}" AS ?hinweis)\n'
            '}\n'
            'LIMIT 1'
        )
        return (safe_query, 0.0, 0, 0, 0, {
            "schema_context_tokens": 0, "instructions_tokens": 0,
            "user_message_tokens": 0, "total_prompt_tokens": 0,
            "output_tokens": 0, "api_prompt_tokens": 0,
            "api_completion_tokens": 0, "graph_rag_applied": False,
            "graph_rag_reduction_pct": 0.0, "injection_blocked": True,
            "injection_reason": block_reason,
        })
    user_prompt = clean_prompt

    # ── Graph-RAG: filter schema context to relevant subgraph ─────────────
    if ont_rag is not None:
        # Proper Graph RAG: embedding-based k-NN + live Oxigraph traversal.
        # Pass a single WorkspaceClient so retrieve() reuses it for the Carina
        # embedding call instead of instantiating a fresh one (saves ~100-200 ms
        # OAuth overhead per request -- same pattern as sql_panel.py).
        from databricks.sdk import WorkspaceClient as _WC
        _rag_w           = _WC()
        rag_ctx          = ont_rag.retrieve(user_prompt, static_context=schema_context, w=_rag_w)
        filtered_context = rag_ctx.context_string
        rag_applied      = rag_ctx.rag_applied
        rag_meta         = {
            "seeds":        rag_ctx.seed_shorts,
            "embedding_ms": rag_ctx.embedding_ms,
            "traversal_ms": rag_ctx.traversal_ms,
            "fallback":     rag_ctx.fallback_reason,
        }
    else:
        # Legacy fallback: keyword->class static text filtering
        filtered_context, rag_applied = apply_graph_rag(user_prompt, schema_context)
        rag_meta = {}

    # ── Template selection: minimal RAG prompt vs full legacy ────────────
    # When Graph RAG provided dynamic context, use the reduced template
    # (~580 tok core rules) — schema/terminology is already in rag_context.
    # Otherwise fall back to the full legacy template (~2300 tok + schema).
    if rag_applied and ont_rag is not None:
        system_prompt = _SYSTEM_PROMPT_RAG_TEMPLATE.format(rag_context=filtered_context)
    else:
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(schema_context=filtered_context)
    user_message  = f"Translate this request to SPARQL: {user_prompt}"

    # ── Consistent LOCAL token estimates for all components ──────────────
    # These use the same tokenizer so they're guaranteed to sum correctly.
    schema_context_tokens  = estimate_token_count(filtered_context)
    system_prompt_tokens   = estimate_token_count(system_prompt)
    full_user_msg_tokens   = estimate_token_count(user_message)
    user_message_tokens    = estimate_token_count(user_prompt)
    prefix_tokens          = full_user_msg_tokens - user_message_tokens
    instructions_tokens    = system_prompt_tokens - schema_context_tokens + prefix_tokens
    total_prompt_tokens    = system_prompt_tokens + full_user_msg_tokens

    query_kwargs: dict = {}
    if cfg.get("supports_temperature", True):
        query_kwargs["temperature"] = 0.0

    t0 = time.perf_counter()
    response = call_llm_with_timeout(
        endpoint=endpoint,
        messages=[
            ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=ChatMessageRole.USER, content=user_message),
        ],
        query_kwargs=query_kwargs,
        timeout_s=timeout,
    )
    elapsed = time.perf_counter() - t0

    sparql_query = response.choices[0].message.content.strip()
    if sparql_query.startswith("```"):
        sparql_query = sparql_query.split("\n", 1)[1] if "\n" in sparql_query else sparql_query[3:]
    if sparql_query.endswith("```"):
        sparql_query = sparql_query[:-3]

    # Auto-inject missing PREFIX declarations to prevent pyoxigraph parse errors
    sparql_query = _ensure_prefixes(sparql_query.strip(), schema_context)

    # API-reported values (may be inaccurate from Databricks proxy)
    usage = getattr(response, "usage", None)
    api_prompt_tokens     = getattr(usage, "prompt_tokens",     0) or 0
    api_completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    # Local estimate of output tokens (consistent with input estimates)
    output_tokens = estimate_token_count(sparql_query.strip())

    # ── Token breakdown dict — all values use the same tokenizer ────────
    full_schema_tokens = estimate_token_count(schema_context)
    token_breakdown = {
        "schema_context_tokens":  schema_context_tokens,
        "instructions_tokens":    instructions_tokens,
        "user_message_tokens":    user_message_tokens,
        "total_prompt_tokens":    total_prompt_tokens,
        "output_tokens":          output_tokens,
        "api_prompt_tokens":      api_prompt_tokens,
        "api_completion_tokens":  api_completion_tokens,
        "graph_rag_applied":      rag_applied,
        "graph_rag_meta":          rag_meta,
        "graph_rag_reduction_pct": round((1 - schema_context_tokens / max(full_schema_tokens, 1)) * 100, 1) if rag_applied else 0.0,
    }

    # ── Prompt Injection Guard: validate output ────────────────────────────
    is_safe, unsafe_reason = validate_sparql_output(sparql_query.strip())
    if not is_safe:
        sparql_query = (
            'PREFIX mrs: <http://22_Ont_Fahrzeugfehler_2025_V1#>\n'
            'SELECT ?hinweis WHERE {\n'
            f'  BIND("Query blockiert: {unsafe_reason}" AS ?hinweis)\n'
            '}\n'
            'LIMIT 1'
        )
        token_breakdown["output_blocked"] = True
        token_breakdown["output_block_reason"] = unsafe_reason


    return (
        sparql_query.strip(),
        elapsed,
        api_prompt_tokens,
        api_completion_tokens,
        system_prompt_tokens,
        token_breakdown,
    )


# =============================================================================
# Stage 5: Feedback-driven SPARQL repair (Self-Debugging [80] / MAC-SQL [48])
#
# Mirrors sql_panel.repair_sql_with_feedback() to ensure parity in LLM-call
# budget between the SQL and ONT pipelines.  Without this loop, SPARQL gets
# exactly 1 LLM call while SQL gets up to 5 — an unfair engineering advantage
# rather than a genuine approach difference.
#
# Flow:
#   1. Receive failed SPARQL + Oxigraph error (or empty-result hint).
#   2. Build repair system prompt (schema context + rules + sandwich suffix).
#   3. LLM call (temperature=0.0) → corrected SPARQL.
#   4. Validate output (prompt guard).
#   5. Return corrected SPARQL or loop (up to _MAX_REPAIR_ROUNDS).
#
# The caller (run_ontology_pipeline) is responsible for re-executing the
# repaired query on Oxigraph and deciding success/failure.
# =============================================================================

_MAX_REPAIR_ROUNDS = 2  # Universally optimal (Survey VI.B); matches SQL pipeline


def repair_sparql_with_feedback(
    clean_question: str,
    schema_context: str,
    failed_sparql: str,
    error_message: str,
    model_config: dict | None = None,
    ont_rag=None,
) -> tuple[str | None, int, float, dict]:
    """Iteratively repair a failing SPARQL query using Oxigraph error feedback.

    Implements the same feedback-driven regeneration loop as sql_panel.py
    Stage 5 (Self-Debugging [80] / MAC-SQL Refiner [48]) to ensure both
    pipelines have an equal LLM-call budget.

    Args:
        clean_question:  Sanitized user question.
        schema_context:  Full or Graph-RAG-filtered ontology schema context.
        failed_sparql:   The SPARQL query that failed or returned 0 results.
        error_message:   Oxigraph parse/runtime error, or semantic hint for
                         empty results.
        model_config:    Model config dict (endpoint, supports_temperature, etc.).
        ont_rag:         OntologyGraphRAG instance (unused here but kept for
                         signature consistency with generate_sparql).

    Returns:
        (fixed_sparql, rounds_used, llm_elapsed, tokens_agg)
        fixed_sparql is None when all repair rounds failed.
        tokens_agg sums tiktoken parts over ALL repair calls.
    """
    cfg      = model_config or {"endpoint": LLM_ENDPOINT, "supports_temperature": True}
    endpoint = cfg["endpoint"]
    timeout  = cfg.get("llm_timeout_s", _LLM_TIMEOUT_S)

    tokens_agg = {
        "system": 0, "schema": 0, "user": 0, "output": 0,
        "api_prompt": 0, "api_completion": 0, "calls": 0,
    }
    llm_elapsed = 0.0
    current_sparql, current_err = failed_sparql, error_message

    # Build a stable repair system prompt (same context the generator used).
    system_prompt = (
        "You are a SPARQL REPAIR expert for a pyoxigraph (Oxigraph) RDF store.\n"
        "A previously generated SPARQL query failed or returned incorrect results.\n"
        "Fix it. Output ONLY the corrected raw SPARQL \u2014 no fences, no explanation.\n"
        "Keep the query a single SELECT or CONSTRUCT statement with PREFIXes.\n\n"
        f"{schema_context}\n\n"
        f"{sandwich_suffix_sparql()}"
    )
    system_tok = estimate_token_count(system_prompt)
    schema_tok = estimate_token_count(schema_context)

    for round_no in range(1, _MAX_REPAIR_ROUNDS + 1):
        user_prompt = (
            f"Original question:\n{clean_question}\n\n"
            f"Failing SPARQL:\n{current_sparql}\n\n"
            f"Oxigraph error / issue:\n{current_err}\n\n"
            "Return the corrected SPARQL only."
        )

        query_kwargs: dict = {}
        if cfg.get("supports_temperature", True):
            query_kwargs["temperature"] = 0.0

        try:
            t0 = time.perf_counter()
            response = call_llm_with_timeout(
                endpoint=endpoint,
                messages=[
                    ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
                    ChatMessage(role=ChatMessageRole.USER, content=user_prompt),
                ],
                query_kwargs=query_kwargs,
                timeout_s=timeout,
            )
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            current_err = f"Repair-LLM-Call fehlgeschlagen: {exc}"
            break

        llm_elapsed += elapsed
        usage = getattr(response, "usage", None)
        p_tok = getattr(usage, "prompt_tokens", 0) or 0
        c_tok = getattr(usage, "completion_tokens", 0) or 0

        tokens_agg["system"] += system_tok
        tokens_agg["schema"] += schema_tok
        tokens_agg["user"]   += estimate_token_count(user_prompt)
        tokens_agg["api_prompt"]     += p_tok
        tokens_agg["api_completion"] += c_tok
        tokens_agg["calls"] += 1

        # Extract and clean the repaired SPARQL.
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        fixed_sparql = _ensure_prefixes(raw.strip(), schema_context)
        tokens_agg["output"] += estimate_token_count(fixed_sparql)

        # Validate output (prompt guard).
        is_safe, unsafe_reason = validate_sparql_output(fixed_sparql)
        if not is_safe:
            current_sparql = fixed_sparql
            current_err = f"Guard-Ablehnung: {unsafe_reason}"
            continue

        # Success: return the repaired query for the caller to execute.
        return fixed_sparql, round_no, llm_elapsed, tokens_agg

    # All rounds exhausted.
    return None, _MAX_REPAIR_ROUNDS, llm_elapsed, tokens_agg
