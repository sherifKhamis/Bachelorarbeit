# =============================================================================
# config.py — Central configuration and shared constants
#
# All environment-specific values (endpoints, catalog, schema) live here so
# that the rest of the codebase never hard-codes paths or names.
#
# Contents:
#   1. MODELS             — LLM model registry (endpoint, timeouts, flags)
#   2. DEFAULT_MODEL      — Which model is used for interactive (non-benchmark)
#   3. LLM_ENDPOINT       — Convenience shortcut for default endpoint name
#   4. VOLUME_PATH        — UC Volume path to the production RDF/OWL TTL file
#   5. CATALOG / SCHEMA   — Unity Catalog coordinates for SQL panel tables
#   6. TABLES             — Fully qualified table names (7 Genie Space data-product tables)
#   7. NODE_COLOR_PALETTE — 25 distinct colours for graph node types
#   8. AREA_COLORS        — Background colours for spatial area group boxes
#   9. EDGE_COLOR_MAP     — Predicate → colour mapping for graph edges
#                           Covers all three fault chains (caused/detected/reworked),
#                           spatial predicates, and VIN/model-type relationships
#  10. SPATIAL_PREDICATES — Set of predicates that define spatial hierarchy
#  11. PREDICATE_TYPE_INFERENCE — Predicate → (source_type, target_type) map
#                           Used by graph_vis.py to infer node types from edges
#                           when rdf:type triples are missing in CONSTRUCT results
# =============================================================================

# ---------------------------------------------------------------------------
# LLM Model Registry
# ---------------------------------------------------------------------------
# Each entry describes one serving endpoint available as a benchmark parameter.
#
# Fields:
#   endpoint             — Databricks serving endpoint name
#   supports_temperature — whether temperature=0.0 is accepted by this model.
#                          Reasoning models may ignore or reject the parameter.
#   llm_timeout_s        — Wall-clock timeout (seconds) for the LLM API call.
#                          Prevents indefinite hangs when the endpoint is
#                          overloaded or rate-limiting after rapid-fire requests.
#   sql_timeout_s        — Databricks SQL warehouse wait_timeout (seconds).
#                          API CONSTRAINT: must be 0 (disables wait) or 5–50.
#                          Values above 50 are rejected with a BadRequest error.
#                          This timeout covers SQL *execution* only — LLM
#                          generation happens before execute_sql_query() is
#                          called. 50s is the API maximum; keep it identical
#                          for every model.
#   sparql_timeout_s     — Wall-clock seconds for Oxigraph SPARQL execution
#                          (_run_sparql_with_timeout in ontology_panel).
#                          Set to 50s (matching sql_timeout_s) to ensure a
#                          fair benchmark comparison between paradigms.
#   description          — short label shown in the UI

MODELS: dict[str, dict] = {
    "GPT-4o": {
        "endpoint":             "MO360_Data_Platform_Nexus_GPT_4o_Chat_Model_Endpoint",
        "supports_temperature": True,
        "llm_timeout_s":        30,    # GPT-4o responds in 2–5s; 30s covers spikes
        "sql_timeout_s":        50,    # API max; covers SQL execution only
        "sparql_timeout_s":     50,    # raised to 90s — complex aggregations need >50s on 10.9M triples
        "description":          "Schnell · ~2–5s",
    },

}

# Default model used by both panels for interactive (non-benchmark) queries.
DEFAULT_MODEL = "GPT-4o"

# Convenience: the raw endpoint name used by existing callers that import
# LLM_ENDPOINT directly. Always resolves to the default model endpoint.
LLM_ENDPOINT = MODELS[DEFAULT_MODEL]["endpoint"]


def get_model_config(name: str) -> dict:
    """Return the model config dict for the given model name.

    Raises KeyError if the name is not registered in MODELS.
    """
    return MODELS[name]


# ---------------------------------------------------------------------------
# Genie Spaces — benchmark comparison targets
# ---------------------------------------------------------------------------
# Each entry represents a Genie Space to include in the benchmark.
# Keys are short labels used in the UI and metrics dicts.

GENIE_SPACES: dict[str, dict] = {
    "Genie Space: MRS pure DPs": {
        "space_id": "01f185b10ef51b9bb0d1c93a66ddd861",
        "description": "MRS reine Dataproducts (ohne Markdown-Instruktionen)",
        "timeout_s": 120,
    },
    "Genie Space: MRS DPs + Markdown": {
        "space_id": "01f168b80522109a9f7871958d8a70d9",
        "description": "MRS Dataproducts mit Markdown-Instruktionen",
        "timeout_s": 120,
    },
}

# ---------------------------------------------------------------------------
# Unity Catalog Volume path where the production RDF/OWL ontology (TTL) lives.
# Downloaded once at startup and loaded into a pyoxigraph.Store (Rust-native
# SPARQL engine).  rdflib is no longer used for parsing or querying.
# ---------------------------------------------------------------------------
VOLUME_PATH = "/Volumes/westeurope_mo360dp_selfservice_prd/skhamis_emea_00/myvolume/mrs_production.ttl"

# ---------------------------------------------------------------------------
# Unity Catalog — production tables (SQL panel)
# These are the 7 normalised data-product tables from the MRS Genie Spaces.
# ---------------------------------------------------------------------------

CATALOG = "westeurope_mo360dp_selfservice_prd"
SCHEMA  = "skhamis_emea_00"

TABLES = {
    "Orders":                  f"{CATALOG}.{SCHEMA}.orders_v1_limited",
    "ActualDates":             f"{CATALOG}.{SCHEMA}.actual_dates_v1_limited",
    "ActualDatesMD":           f"{CATALOG}.{SCHEMA}.actual_dates_md_v1_limited",
    "Faults":                  f"{CATALOG}.{SCHEMA}.faults_v1_limited",
    "FaultTypesMD":            f"{CATALOG}.{SCHEMA}.fault_types_md_v1_limited",
    "FaultVehicleLocationsMD": f"{CATALOG}.{SCHEMA}.fault_vehicle_locations_md_v1_limited",
    "OrderModuleMapping":      f"{CATALOG}.{SCHEMA}.order_module_mapping_v1_limited",
}

# ---------------------------------------------------------------------------
# Graph visualisation — node colours and shapes (graph_vis.py)
# ---------------------------------------------------------------------------

# Mercedes-Benz design-aligned node colour palette (25 distinct colours)
NODE_COLOR_PALETTE = [
    '#CC0000', '#5A9FD4', '#3D8C5E', '#CC9900', '#808080',
    '#1ABC9C', '#2B5F8A', '#1E3A5F', '#996600', '#16A085',
    '#4A4060', '#990000', '#6ECF6E', '#C0C0C0', '#37536D',
    '#B30000', '#4A9FBF', '#A07840', '#607D8B', '#5C5C5C',
    '#2E7D32', '#AA7700', '#3D4080', '#00838F', '#888888',
]

# Pyvis Flächen-Hintergründe auf weißem Canvas — MB BKK Light Design
AREA_COLORS = {
    'Bodyshop':  '#FFF3E0',    # warmes Amber — Karosseriebau
    'Assembly':  '#E3F2FD',    # helles BKK-Blau — Montage
    'Paintshop': '#F1F8E9',    # helles Grün — Lackiererei
}

# Mercedes-Benz design edge colours
# Caused = MB Red family · Detected = MB Steel Blue · Reworked = dark green
# VIN/model = MB Amber · Spatial = MB blue-grey · Metadata = neutral
EDGE_COLOR_MAP = {
    'isCausedAtCheckpoint': '#CC0000', 'isCausedInHall':     '#990000',
    'isCausedAtMark':       '#B30000', 'isCausedInCategory': '#800000',
    'isDetectedAtCheckpoint': '#2B5F8A', 'isDetectedInHall':     '#1E3A5F',
    'isDetectedAtMark':       '#1565C0', 'isDetectedInCategory': '#0D47A1',
    'isReworkedAtCheckpoint': '#2E7D32', 'isReworkedInHall':     '#1B5E20',
    'isReworkedAtMark':       '#145A32', 'isReworkedInCategory': '#0D3B22',
    'passedThroughCheckpoint': '#CC9900',
    'hasFault':                '#CC0000',
    'affectsVIN':              '#CC0000',
    'hasAffectedVIN':          '#996600',
    'hasVINs':                 '#CC9900',
    'vinInHall':               '#AA7700',
    'vinInArea':               '#CC9900',
    'vinInPlant':              '#996600',
    'isLocatedIn':             '#4A4060',
    'isInHall':                '#2B5F8A',
    'isInArea':                '#37536D',
    'isInPlant':               '#4A4060',
    'hasType':                 '#5C5C7A',
    'hasCategory':             '#4A4060',
    'categoriesInHall':        '#3D3060',
    'hasMark':                 '#607D8B',
    'marksInHall':             '#607D8B',
    'hasFaults':               '#CC0000',
    'checkpointWithFaults':    '#990000',
    'occursAtVehicleLocation': '#1ABC9C',
    'affectedModelTypes':      '#996600',
    'belongsToModelType':      '#CC9900',
    'isProducedInArea':        '#2B5F8A',
    'isProducedInCheckpoint':  '#1E3A5F',
    'isProducedInHall':        '#37536D',
    'isProducedInPlant':       '#2B5070',
    'locationProgress':        '#5C5C5C',
    'isPartOf':                '#7A7A7A',
}

SPATIAL_PREDICATES = {
    'isInHall', 'isInArea', 'isLocatedIn', 'isInPlant', 'locationProgress',
}

PREDICATE_TYPE_INFERENCE = {
    'hasFault':                ('VIN',       'Fault'),
    'hasFaults':               ('Checkpoint','Fault'),
    'hasType':                 ('Fault',     'FaultType'),
    'occursAtVehicleLocation': ('Fault',     'FaultVehicleLocation'),
    'affectedModelTypes':      ('Fault',     'ModelType'),
    'affectsVIN':              ('Fault',     'VIN'),
    'passedThroughCheckpoint': ('VIN',       'Checkpoint'),
    'belongsToModelType':      ('VIN',       'ModelType'),
    'vinInArea':               ('VIN',       'Area'),
    'vinInHall':               ('VIN',       'Hall'),
    'vinInPlant':              ('VIN',       'Plant'),
    'hasAffectedVIN':          ('ModelType', 'VIN'),
    'hasVINs':                 ('ModelType', 'VIN'),
    'isProducedInArea':        ('ModelType', 'Area'),
    'isProducedInCheckpoint':  ('ModelType', 'Checkpoint'),
    'isProducedInHall':        ('ModelType', 'Hall'),
    'isProducedInPlant':       ('ModelType', 'Plant'),
    'isCausedAtCheckpoint':    ('Fault', 'CausedCheckpoint'),
    'isCausedAtMark':          ('Fault', 'CausedMark'),
    'isCausedInCategory':      ('Fault', 'CausedCategory'),
    'isCausedInHall':          ('Fault', 'CausedHall'),
    'isDetectedAtCheckpoint':  ('Fault', 'DetectedCheckpoint'),
    'isDetectedAtMark':        ('Fault', 'DetectedMark'),
    'isDetectedInCategory':    ('Fault', 'DetectedCategory'),
    'isDetectedInHall':        ('Fault', 'DetectedHall'),
    'isReworkedAtCheckpoint':  ('Fault', 'ReworkedCheckpoint'),
    'isReworkedAtMark':        ('Fault', 'ReworkedMark'),
    'isReworkedInCategory':    ('Fault', 'ReworkedCategory'),
    'isReworkedInHall':        ('Fault', 'ReworkedHall'),
    'hasCategory':          ('Checkpoint', 'Category'),
    'hasMark':              ('Checkpoint', 'Mark'),
    'checkpointWithFaults': ('Hall',       'Checkpoint'),
    'categoriesInHall':     ('Hall',       'Category'),
    'marksInHall':          ('Hall',       'Mark'),
    'isInHall':             ('Checkpoint', 'Hall'),
    'isInArea':             ('Hall',       'Area'),
    'isInPlant':            ('Area',       'Plant'),
    'isLocatedIn':          ('Hall',       'Plant'),
}
