# =============================================================================
# corpus.py — Fragen-Corpus management for the LLM Parameterstudie
#
# This module defines and manages the 85-question benchmark corpus.
# A "corpus" (lat. "Körper") is a standardised collection of test questions
# used to evaluate and compare LLM performance reproducibly across runs
# and models.
#
# Manages:
#   1. DEFAULT_CORPUS fixture — 85 questions as list[dict] with id + question.
#      Questions are ordered by increasing complexity and grouped into 5 levels.
#   2. Session-state CRUD — add, delete, reset questions at runtime.
#      Stored in st.session_state["benchmark_corpus"].
#   3. Model selection — which models to benchmark (multiselect UI).
#      Stored in st.session_state["benchmark_selected_models"]
#      as a list of model-name strings (keys from config.MODELS).
#   4. Pipeline selection — which pipelines to include in each benchmark run.
#      Stored via individual checkbox keys per pipeline (ONT, SQL, Genie spaces).
#      At least one pipeline must be selected for the benchmark to run.
#   5. Complexity documentation — delegated to complexity_docs.py module.
#
# Fields per question dict:
#   id          — Q-number label (Q001–Q085 fixture; Q086+ user-created)
#   question    — natural-language question in German
#   is_fixture  — True for all DEFAULT_CORPUS entries; False for user-added
#                 questions. benchmark_metrics.py uses this flag to skip
#                 auto-categorisation (question_type / join_depth /
#                 triple_path_depth) for fixture questions, since their
#                 category is defined by the Q-number range below.
#
# Note: question_type, join_depth, and triple_path_depth are automatically
# computed at benchmark runtime from the generated queries (benchmark_metrics.py)
# for USER-CREATED questions only (is_fixture=False).
# Fixture questions are categorised by their Q-number range — no auto-compute.
#
# Complexity dimensions (85 questions, increasing difficulty):
#   unbeantwortbar    — Q001–Q005  (not answerable from available data;
#                                    tests hallucination vs. graceful refusal)
#   faktisch          — Q006–Q025  (direct property lookup;
#                                    join_depth=0, triple_path_depth=1)
#   relational        — Q026–Q045  (entity relationships, 1-hop joins;
#                                    join_depth=1, triple_path_depth=1–2)
#   aggregativ        — Q046–Q065  (COUNT/SUM/AVG aggregations;
#                                    join_depth=0–2, triple_path_depth=1–2)
#   multi-hop         — Q066–Q085  (multi-relation chain queries;
#                                    join_depth=2–3, triple_path_depth=3–4)
#
# Exported functions:
#   • get_corpus()              — Returns current corpus (session state or default)
#   • get_selected_models()     — Returns list of selected model names
#   • get_selected_pipelines()  — Returns set of selected pipeline names
#   • render_corpus_manager()   — Streamlit UI: question editor + model + pipeline selector
# =============================================================================

from __future__ import annotations

import streamlit as st

from config import MODELS, DEFAULT_MODEL, GENIE_SPACES
from complexity_docs import render_complexity_docs

_SESSION_KEY         = "benchmark_corpus"
_MODEL_SELECTION_KEY = "benchmark_selected_models"
_PIPELINE_SELECTION_KEY = "benchmark_selected_pipelines"  # namespace sentinel (unused directly)

# =============================================================================
# Default fixture — 85 questions of increasing complexity
# =============================================================================

DEFAULT_CORPUS: list[dict] = [
    # ═══════════════════════════════════════════════════════════════════════════
    # UNBEANTWORTBAR — Fragen, die aus den verfügbaren Daten NICHT beantwortet
    # werden können. Testet ob das LLM halluziniert oder korrekt ablehnt.
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id":         "Q001",
        "question":   "Wie hoch ist der Hubraum des Fahrzeugs mit VIN W1KLF4HB0TA278667?",
        "is_fixture": True,
    },
    {
        "id":         "Q002",
        "question":   "Welche Dimensionen hat der Achskolben aus Untertürkheim?",
        "is_fixture": True,
    },
    {
        "id":         "Q003",
        "question":   "Wie schnell ist die Höchstgeschwindigkeit des Fahrzeugs mit VIN W1KLF4HB0TA278667 ?",
        "is_fixture": True,
    },
    {
        "id":         "Q004",
        "question":   "Welcher Zulieferer liefert die Bauteile für die Montage in Halle 56?",
        "is_fixture": True,
    },
    {
        "id":         "Q005",
        "question":   "Wie viele abwesende Mitarbeiter hat die nächste Schicht im Karosseriebau?",
        "is_fixture": True,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # FAKTISCH — direkte Eigenschaft eines Objekts
    # join_depth=0, triple_path_depth=1
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id":         "Q006",
        "question":   "Welchen Fahrzeugtyp hat das Fahrzeug mit VIN W1KLF4HB4TA284438?",
        "is_fixture": True,
    },
    {
        "id":         "Q007",
        "question":   "In welchem Werk befindet sich das Fahrzeug mit VIN W1KLF4HB4TA284438?",
        "is_fixture": True,
    },
    {
        "id":         "Q008",
        "question":   "In welcher Halle liegt der Checkpoint SCHLUSSABN. SIFI?",
        "is_fixture": True,
    },
    {
        "id":         "Q009",
        "question":   "Welche Halle hat den letzten Checkpoint in der Montage?",
        "is_fixture": True,
    },
    {
        "id":         "Q010",
        "question":   "Welche Checkpoint-Beschreibung hat der Checkpoint mit Mark 'RZ1' in Halle 9?",
        "is_fixture": True,
    },
    {
        "id":         "Q011",
        "question":   "An welchem Checkpoint wurde der Fehler 38036950501 entdeckt?",
        "is_fixture": True,
    },
    {
        "id":         "Q012",
        "question":   "Welche verschiedenen Fahrzeugtypen gibt es im Werk insgesamt?",
        "is_fixture": True,
    },
    {
        "id":         "Q013",
        "question":   "Welche Halle enthält die erste Station im Karosseriebau?",
        "is_fixture": True,
    },
    {
        "id":         "Q014",
        "question":   "Welche verschiedenen Produktionsbereiche gibt es im Werk?",
        "is_fixture": True,
    },
    {
        "id":         "Q015",
        "question":   "Welche Checkpoint-Beschreibung hat der letzte Checkpoint in der Lackiererei?",
        "is_fixture": True,
    },
    {
        "id":         "Q016",
        "question":   "Welchen Fahrzeugtyp hat das Fahrzeug mit VIN W1NKM5GB2TU136573?",
        "is_fixture": True,
    },
    {
        "id":         "Q017",
        "question":   "In welchem Produktionsbereich liegt die Halle 46?",
        "is_fixture": True,
    },
    {
        "id":         "Q018",
        "question":   "Welchen location_progress-Wert hat der letzte Checkpoint in der Montage?",
        "is_fixture": True,
    },
    {
        "id":         "Q019",
        "question":   "Welche Mark hat der Checkpoint SCHLUSSABN. SIFI?",
        "is_fixture": True,
    },
    {
        "id":         "Q020",
        "question":   "Welchen location_progress-Wert hat der Checkpoint mit Mark 'LS' in Halle 44?",
        "is_fixture": True,
    },
    {
        "id":         "Q021",
        "question":   "Welche Checkpoint-Beschreibung hat der Checkpoint mit Mark 'LS' in Halle 44?",
        "is_fixture": True,
    },
    {
        "id":         "Q022",
        "question":   "Welche verschiedenen Hallen gibt es im Werk 0500?",
        "is_fixture": True,
    },
    {
        "id":         "Q023",
        "question":   "In welchem Werk wird das Fahrzeug mit VIN W1KLF4HB4TA284438 produziert?",
        "is_fixture": True,
    },
    {
        "id":         "Q024",
        "question":   "Welche Halle enthält den letzten Checkpoint im Karosseriebau?",
        "is_fixture": True,
    },
    {
        "id":         "Q025",
        "question":   "Welcher Kategorie ist der Checkpoint mit Mark 'RZ1' in Halle 9 zugeordnet?",
        "is_fixture": True,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # RELATIONAL — Verbindung zwischen zwei Entitäten über eine Relation
    # join_depth=1, triple_path_depth=1–2
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id":         "Q026",
        "question":   "Welche Stationen hat das Fahrzeug mit VIN W1KLF4HB4TA284438 durchlaufen?",
        "is_fixture": True,
    },
    {
        "id":         "Q027",
        "question":   "Welche Fehler (ID, Typ und Position am Fahrzeug) sind dem Fahrzeug W1KLF4HB4TA284438 zugeordnet?",
        "is_fixture": True,
    },
    {
        "id":         "Q028",
        "question":   "Welche Checkpoints hat VIN W1KLF4HB4TA284438 in der Halle 44 durchlaufen?",
        "is_fixture": True,
    },
    {
        "id":         "Q029",
        "question":   "Welche Fahrzeug VINs durchlaufen die Halle '56' der Montage?",
        "is_fixture": True,
    },
    {
        "id":         "Q030",
        "question":   "Welche Checkpoints mit Mark 'RZ1' gibt es im Karosseriebau?",
        "is_fixture": True,
    },
    {
        "id":         "Q031",
        "question":   "Welche Marks hat die Halle '44' in der Lackiererei?",
        "is_fixture": True,
    },
    {
        "id":         "Q032",
        "question":   "Welche Fahrzeuge haben mindestens einen entdeckten Fehler im Bereich Montage?",
        "is_fixture": True,
    },
    {
        "id":         "Q033",
        "question":   "Welche Checkpoint-Beschreibung hat die erste Station, die VIN W1KLF4HB4TA284438 im Rohbau durchlaufen hat?",
        "is_fixture": True,
    },
    {
        "id":         "Q034",
        "question":   "Welche Fahrzeugtypen werden in der Halle '9' produziert?",
        "is_fixture": True,
    },
    {
        "id":         "Q035",
        "question":   "An welchen Checkpoints wurde mindestens ein Fehler entdeckt für VIN W1KLF4HB4TA284438?",
        "is_fixture": True,
    },
    {
        "id":         "Q036",
        "question":   "Welche Fehlertypen-Werte treten bei Fahrzeugen vom Typ 'W214' auf?",
        "is_fixture": True,
    },
    {
        "id":         "Q037",
        "question":   "Zeige 10 Fahrzeuge mit ihren Fahrzeugfehlerpositions-Angaben aus dem Karosseriebau.",
        "is_fixture": True,
    },
    {
        "id":         "Q038",
        "question":   "Welche entdeckte Fehlertypen hat das Fahrzeug mit VIN W1NKM5GB2TU136573 in der Lackiererei?",
        "is_fixture": True,
    },
    {
        "id":         "Q039",
        "question":   "An welchen Checkpoints wurde mindestens ein Fehler verursacht für VIN W1NKM5GB2TU136573?",
        "is_fixture": True,
    },
    {
        "id":         "Q040",
        "question":   "Welche Fahrzeug VINs durchlaufen die Halle '30' im Karosseriebau?",
        "is_fixture": True,
    },
    {
        "id":         "Q041",
        "question":   "Welche Fahrzeugtypen werden in der Halle '46' der Montage produziert?",
        "is_fixture": True,
    },
    {
        "id":         "Q042",
        "question":   "Welche Checkpoints hat VIN W1NKM5GB2TU136573 in der Halle 56 durchlaufen?",
        "is_fixture": True,
    },
    {
        "id":         "Q043",
        "question":   "Welche Fehler (ID und Typ) hat das Fahrzeug W1NKM5GB2TU136573 in der Lackiererei?",
        "is_fixture": True,
    },
    {
        "id":         "Q044",
        "question":   "An welcher Position am Fahrzeug treten Fehler bei VIN W1KLF4HB4TA284438 in der Lackiererei auf?",
        "is_fixture": True,
    },
    {
        "id":         "Q045",
        "question":   "Welche Fahrzeuge haben mindestens einen nachgearbeiteten Fehler in der Montage ?",
        "is_fixture": True,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # AGGREGATIV — Zählen, Summen, Durchschnitte über eine Menge
    # join_depth=0–2, triple_path_depth=1–2
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id":         "Q046",
        "question":   "Wie viele Fahrzeuge haben Fehler, welche in der Montage entdecket wurden?",
        "is_fixture": True,
    },
    {
        "id":         "Q047",
        "question":   "Wie viele einzigartige Fahrzeuge pro Fahrzeugtyp befinden sich im Karosseriebau?",
        "is_fixture": True,
    },
    {
        "id":         "Q048",
        "question":   "Welche Hallen haben die meisten entdeckten Fehler in der Lackiererei?",
        "is_fixture": True,
    },
    {
        "id":         "Q049",
        "question":   "Wie viele Checkpoints hat jede Halle in der Montage?",
        "is_fixture": True,
    },
    {
        "id":         "Q050",
        "question":   "Was ist die durchschnittliche Anzahl an entdeckten Fehlern pro Fahrzeug in der Lackiererei?",
        "is_fixture": True,
    },
    {
        "id":         "Q051",
        "question":   "Welche 5 Fehlertypen werden am häufigsten in der Montage verursacht?",
        "is_fixture": True,
    },
    {
        "id":         "Q052",
        "question":   "Wie viele verschiedene entdeckte Fehlertypen gibt es pro Bereich?",
        "is_fixture": True,
    },
    {
        "id":         "Q053",
        "question":   "Liste mir die 3 Fahrzeuge mit den meisten entdeckten Fehlern auf welche nicht in der Montage entdeckt wurden.",
        "is_fixture": True,
    },
    {
        "id":         "Q054",
        "question":   "Welcher Checkpoint-Mark hat die meisten einzigartigen durchlaufenden Fahrzeuge in der Karosseriebau?",
        "is_fixture": True,
    },
    {
        "id":         "Q055",
        "question":   "Welche 3 Stationen haben die höchste Anzahl an verursachten Fehlern ?",
        "is_fixture": True,
    },
    {
        "id":         "Q056",
        "question":   "Wie viele verschiedene Hallen gibt es in der Lackiererei?",
        "is_fixture": True,
    },
    {
        "id":         "Q057",
        "question":   "Wie viele einzigartige Fahrzeuge haben verursachte Fehler in der Lackiererei?",
        "is_fixture": True,
    },
    {
        "id":         "Q058",
        "question":   "Welche 3 Marks haben die meisten zugeordneten Checkpoints im Karosseriebau?",
        "is_fixture": True,
    },
    {
        "id":         "Q059",
        "question":   "Wie viele Fehlertypen pro Fahrzeugtyp gibt es in der Montage?",
        "is_fixture": True,
    },
    {
        "id":         "Q060",
        "question":   "Welche Halle hat die wenigsten Checkpoints in der Lackiererei?",
        "is_fixture": True,
    },
    {
        "id":         "Q061",
        "question":   "Wie viele Fahrzeuge haben genau einen entdeckten Fehler in der Montage?",
        "is_fixture": True,
    },
    {
        "id":         "Q062",
        "question":   "Wie viele verschiedene Marks gibt es pro Halle im Karosseriebau?",
        "is_fixture": True,
    },
    {
        "id":         "Q063",
        "question":   "Welche 5 Fahrzeuge haben die meisten durchlaufenen Checkpoints in der Montage?",
        "is_fixture": True,
    },
    {
        "id":         "Q064",
        "question":   "Was ist die durchschnittliche Anzahl an Checkpoints pro Halle über alle Bereiche?",
        "is_fixture": True,
    },
    {
        "id":         "Q065",
        "question":   "Welche Fehlertypen treten in mehr als einer Halle der Lackiererei auf?",
        "is_fixture": True,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-HOP — Antwort erfordert Ketten mehrerer Relationen
    # join_depth=2–3, triple_path_depth=3–4
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id":         "Q066",
        "question":   "Welche 5 VINs haben die meisten verursachten Fehler über alle Produktionsbereiche hinweg?",
        "is_fixture": True,
    },
    {
        "id":         "Q067",
        "question":   "Wie oft kommt jeder Fahrzeugtyp in jedem Produktionsbereich vor ?",
        "is_fixture": True,
    },
    {
        "id":         "Q068",
        "question":   "Welche Fahrzeuge haben mindestens einen Fehler, dessen Verursachungs-Checkpoint und Erkennungs-Checkpoint in verschiedenen Hallen liegen?",
        "is_fixture": True,
    },
    {
        "id":         "Q069",
        "question":   "Zeige den vollständigen Produktionsweg der Checkpoints für VIN W1KLF4HB4TA284438 über alle Bereiche.",
        "is_fixture": True,
    },
    {
        "id":         "Q070",
        "question":   "Welche Fahrzeugtypen haben entdeckte Fehler an Checkpoints, die zur Halle '46' in der Montage gehören?",
        "is_fixture": True,
    },
    {
        "id":         "Q071",
        "question":   "Für jede Halle in der Lackierei: Wie viele Fehler wurden dort verursacht, erkannt und nachgearbeitet?",
        "is_fixture": True,
    },
    {
        "id":         "Q072",
        "question":   "Welche Fahrzeuge haben entdeckte Fehler in mehr als einem Produktionsbereich?",
        "is_fixture": True,
    },
    {
        "id":         "Q073",
        "question":   "Zeige ein Fahrzeug, dessen Fehler an einem Checkpoint verursacht wurde, den es vorher durchlaufen hat.",
        "is_fixture": True,
    },
    {
        "id":         "Q074",
        "question":   "Welche Checkpoints im Karosseriebau verursachen Fehler, die in einem anderen Produktionsbereich entdeckt werden?",
        "is_fixture": True,
    },
    {
        "id":         "Q075",
        "question":   "Gibt es Fehlertypen, die in der Lackiererei häufiger auftreten als im Karosseriebau?",
        "is_fixture": True,
    },
    {
        "id":         "Q076",
        "question":   "Welche Hallen haben Checkpoints, an denen sowohl Fehler verursacht als auch entdeckt werden?",
        "is_fixture": True,
    },
    {
        "id":         "Q077",
        "question":   "Welche Fehler treten bei VIN W1KLF4HB4TA284438 in Hallen auf, die zu unterschiedlichen Produktionsbereichen gehören?",
        "is_fixture": True,
    },
    {
        "id":         "Q078",
        "question":   "Welche Fahrzeuge haben endeckte Fehler in mindestens zwei der drei Bereiche (Karosseriebau, Lackiererei, Montage)?",
        "is_fixture": True,
    },
    {
        "id":         "Q079",
        "question":   "Für VIN W1KLF4HB4TA284438: Zeige den Fehler 38036950501 mit Verursachungs-Checkpoint, Erkennungs-Checkpoint und Nacharbeits-Checkpoint, mit Halle und Bereich.",
        "is_fixture": True,
    },
    {
        "id":         "Q080",
        "question":   "Welche Fahrzeugtypen haben Fehler, die im Karosseriebau verursacht aber in der Lackiererei entdeckt wurden?",
        "is_fixture": True,
    },
    {
        "id":         "Q081",
        "question":   "Für VIN W1NKM5GB2TU136573: Zeige alle Fehler mit Verursachungs-Halle und Erkennungs-Halle über alle Bereiche.",
        "is_fixture": True,
    },
    {
        "id":         "Q082",
        "question":   "Welche Hallen in der Lackiererei verursachen die meisten Fehler, die in der Montage entdeckt werden?",
        "is_fixture": True,
    },
    {
        "id":         "Q083",
        "question":   "Gibt es Fahrzeuge, deren Fehler in drei verschiedenen Hallen verursacht, erkannt und nachgearbeitet wurden?",
        "is_fixture": True,
    },
    {
        "id":         "Q084",
        "question":   "Für die Hallen (8, 9, 30) im Karosseriebau: Wie viele Fehler wurden dort verursacht, erkannt und nachgearbeitet?",
        "is_fixture": True,
    },
    {
        "id":         "Q085",
        "question":   "Welche Fahrzeugtypen kommen in allen drei Produktionsbereichen gleichzeitig vor?",
        "is_fixture": True,
    },
]


# =============================================================================
# Session-state helpers
# =============================================================================

def _init_corpus() -> None:
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = [q.copy() for q in DEFAULT_CORPUS]


def _init_model_selection() -> None:
    if _MODEL_SELECTION_KEY not in st.session_state:
        st.session_state[_MODEL_SELECTION_KEY] = [DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# Pipeline selection helpers
# ---------------------------------------------------------------------------

def _pipeline_cb_key(pipeline: str) -> str:
    """Return the session-state key for a pipeline's checkbox widget."""
    return "_pipeline_cb_" + pipeline.replace(" ", "_").replace(":", "").replace("/", "_").replace("+", "plus")


def _get_all_pipeline_keys() -> list[str]:
    """Return the ordered list of all pipeline names (ONT, SQL, Genie spaces)."""
    return ["ONT", "SQL"] + list(GENIE_SPACES.keys())


def _init_pipeline_selection() -> None:
    """Initialise pipeline checkboxes to True (all selected) if not yet set."""
    for pipeline in _get_all_pipeline_keys():
        key = _pipeline_cb_key(pipeline)
        if key not in st.session_state:
            st.session_state[key] = True


def get_selected_pipelines() -> set[str]:
    """Return the set of pipeline names currently checked in session state."""
    _init_pipeline_selection()
    return {p for p in _get_all_pipeline_keys() if st.session_state.get(_pipeline_cb_key(p), True)}


def get_corpus() -> list[dict]:
    """Return the current corpus list from session state."""
    _init_corpus()
    return st.session_state[_SESSION_KEY]


def get_selected_models() -> list[str]:
    """Return the list of model names selected for the corpus benchmark."""
    _init_model_selection()
    return st.session_state[_MODEL_SELECTION_KEY]


# =============================================================================
# CRUD + model selection UI
# =============================================================================

def render_corpus_manager() -> None:
    """Render corpus management UI: question CRUD + model selection + pipeline selection."""
    _init_corpus()
    _init_model_selection()
    _init_pipeline_selection()
    corpus: list[dict] = st.session_state[_SESSION_KEY]

    # ── Model selection ───────────────────────────────────────────────────────
    _render_model_selection()

    st.divider()

    # ── Pipeline selection ────────────────────────────────────────────────────
    _render_pipeline_selection()

    st.divider()

    # ── Complexity documentation (delegated to complexity_docs module) ─────────
    render_complexity_docs()

    # ── Question list ─────────────────────────────────────────────────────────
    _render_question_list(corpus)


# =============================================================================
# Model selection sub-component
# =============================================================================

def _render_model_selection() -> None:
    """Render model multiselect with warning for high-latency models."""
    st.markdown("**Modell-Parameter**")

    # CSS: prevent multiselect pills from truncating long labels
    st.markdown(
        "<style>"
        "[data-testid='stMultiSelect'] span[data-baseweb='tag'] {"
        "  max-width: none !important;"
        "}"
        "[data-testid='stMultiSelect'] span[data-baseweb='tag'] span {"
        "  max-width: none !important;"
        "  overflow: visible !important;"
        "  text-overflow: unset !important;"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )

    model_names   = list(MODELS.keys())
    model_options = [
        f"{name}  ·  {MODELS[name]['description']}" for name in model_names
    ]
    _label_to_name = dict(zip(model_options, model_names))

    current_selection = st.session_state[_MODEL_SELECTION_KEY]
    current_labels = [
        f"{n}  ·  {MODELS[n]['description']}"
        for n in current_selection
        if n in MODELS
    ]

    selected_labels = st.multiselect(
        "Modell(e) für den Corpus-Benchmark",
        options=model_options,
        default=current_labels,
        key="model_multiselect",
        help=(
            "Wähle ein oder mehrere Modelle. Der Benchmark läuft jede Frage "
            "für jedes Modell — eine Zeile pro (Frage × Modell)-Kombination."
        ),
    )
    st.session_state[_MODEL_SELECTION_KEY] = [
        _label_to_name[lbl] for lbl in selected_labels if lbl in _label_to_name
    ]

    selected_names = st.session_state[_MODEL_SELECTION_KEY]

    if not selected_names:
        st.error("Mindestens ein Modell muss ausgewählt sein.")


# =============================================================================
# Pipeline selection sub-component
# =============================================================================

def _render_pipeline_selection() -> None:
    """Render one checkbox per pipeline for selecting which to include in the benchmark.

    Pipelines: ONT (SPARQL/Ontologie), SQL, and each configured Genie Space.
    At least one must remain selected — the Run-button is disabled otherwise.
    """
    _init_pipeline_selection()

    all_pipelines = _get_all_pipeline_keys()

    # Human-readable labels per pipeline
    _labels: dict[str, str] = {
        "ONT": "🔷 ONT (SPARQL)",
        "SQL": "🟢 SQL",
    }
    for genie_key in GENIE_SPACES:
        _labels[genie_key] = f"💬 {genie_key}"

    st.markdown("**Pipelines im Benchmark**")
    st.caption(
        "Wähle welche Pipelines pro Frage ausgeführt werden sollen. "
        "Mindestens eine muss aktiv sein."
    )

    cols = st.columns(len(all_pipelines))
    for col, pipeline in zip(cols, all_pipelines):
        with col:
            st.checkbox(
                _labels.get(pipeline, pipeline),
                key=_pipeline_cb_key(pipeline),
            )

    selected = get_selected_pipelines()
    if not selected:
        st.error("⚠️ Mindestens eine Pipeline muss ausgewählt sein, damit der Benchmark startet.")


# =============================================================================
# Question list sub-component
# =============================================================================

def _render_question_list(corpus: list[dict]) -> None:
    """Render the question list with delete buttons and add-new form."""
    # Header
    hdr_left, hdr_right = st.columns([7, 3])
    with hdr_left:
        st.caption(
            f"**{len(corpus)} Frage{'n' if len(corpus) != 1 else ''}** · "
            f"Fixture: {len(DEFAULT_CORPUS)} Standardfragen"
        )
    with hdr_right:
        if st.button("↺ Fixture zurücksetzen", key="corpus_reset", use_container_width=True):
            st.session_state[_SESSION_KEY] = [q.copy() for q in DEFAULT_CORPUS]
            st.rerun(scope="fragment")

    # Column headers
    h1, h2, h3 = st.columns([1, 9, 1])
    h1.markdown("<small><b>ID</b></small>",    unsafe_allow_html=True)
    h2.markdown("<small><b>Frage</b></small>", unsafe_allow_html=True)
    h3.markdown("<small><b></b></small>",      unsafe_allow_html=True)

    st.divider()

    # Question rows
    for i, q in enumerate(corpus):
        c1, c2, c3 = st.columns([1, 9, 1])
        with c1:
            st.code(q["id"], language=None)
        with c2:
            st.write(q["question"])
        with c3:
            if st.button("✕", key=f"del_q_{i}"):
                corpus.pop(i)
                st.session_state[_SESSION_KEY] = corpus
                st.rerun(scope="fragment")

    st.divider()

    # Add new question form
    _render_add_question_form(corpus)


def _render_add_question_form(corpus: list[dict]) -> None:
    """Render the form for adding a new question to the corpus."""
    if st.checkbox("➕ Neue Frage hinzufügen", key="show_add_form"):
        new_text = st.text_area(
            "Frage", key="new_q_text",
            placeholder="Natürlichsprachige Frage auf Deutsch…", height=68,
        )
        st.caption(
            "ℹ️ Fragetyp, Join-Tiefe und Pfadtiefe werden automatisch "
            "beim Benchmark aus den generierten Queries ermittelt."
        )

        if st.button("Hinzufügen", key="corpus_add", type="primary"):
            if new_text.strip():
                existing_nums = [
                    int(q["id"][1:]) for q in corpus if q["id"][1:].isdigit()
                ]
                next_num = max(existing_nums, default=0) + 1
                corpus.append({
                    "id":         f"Q{next_num:03d}",
                    "question":   new_text.strip(),
                    "is_fixture": False,  # user-created: auto-categorisation applies
                })
                st.session_state[_SESSION_KEY] = corpus
                st.rerun(scope="fragment")
            else:
                st.warning("Bitte eine Frage eingeben.")
