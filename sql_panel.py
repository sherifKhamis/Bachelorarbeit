# =============================================================================
# sql_panel.py - SQL-Pipeline: LLM-Chat ueber MO360-Produktionstabellen
#
# Uebersetzt natuerlichsprachliche Fragen in Spark SQL, fuehrt sie auf einem
# Databricks SQL Warehouse aus und zeigt Ergebnisse als pandas DataFrame an.
#
# =============================================================================
# STAND DER TECHNIK (Referenz, bitte lesen)
# =============================================================================
#
# Diese Pipeline wurde von "C0 - Vanilla Prompting" (Zero-/One-Shot mit
# statischem Vollschema) auf den hybriden In-Context-Learning-Stand der
# Technik angehoben, wie er in der Referenz-Uebersicht konsolidiert ist:
#
#   Hong et al., "Next-Generation Database Interfaces: A Survey of
#   LLM-Based Text-to-SQL", IEEE TKDE, Vol. 37, No. 12, Dezember 2025.
#
# Das Survey ordnet ICL-Methoden in die Kategorien C1-C4 ein und stellt als
# aktuellen Trend die HYBRIDE Kombination dieser Kategorien fest (SuperSQL,
# CHASE-SQL, ReFoRCE). Genau diese Hybridisierung setzt die Pipeline um.
# Referenzstand: Verfahren 2023 bis Ende 2025. Abbildung der Stufen:
#
#   Stufe 1  Schema Linking (Tabellen-Filterung)          -> C1/C2b
#            wie C3 [28], CHESS [56], RSL-SQL [59], LinkAlign [62].
#            Bewusst NUR auf Tabellenebene und mit konservativem Fallback
#            auf das Vollschema, weil Distillery [58] ("The death of
#            schema linking?") zeigt, dass aggressives Filtern bei starken
#            Modellen Recall kostet. Bei nur 7 Tabellen ist Tabellen-
#            granularitaet der richtige Kompromiss.
#
#   Stufe 2  Few-Shot-Auswahl per Frage-Aehnlichkeit      -> C2a
#            DAIL-SQL-Prinzip [8]: Literale (VINs, Zahlen) werden maskiert,
#            dann Skeleton-Aehnlichkeit gegen einen kuratierten In-Domain-
#            Beispielpool (ODIS [67]: In-Domain-Demos schlagen Zero-Shot).
#
#   Stufe 3  Multi-Kandidaten-Generierung                 -> C3a + C3b
#            Diversitaet ueber unterschiedliche Strategie-Prompts
#            (Direct, Divide-and-Conquer-CoT wie CHASE-SQL [77],
#            Execution-Order-Plan wie QDecomp [47]) plus Temperatur,
#            wo der Endpoint sie unterstuetzt (MCS-SQL [55] zeigt, dass
#            Prompt-Diversitaet allein bereits traegt). Self-Consistency-
#            Kurzschluss: stimmen die ersten Kandidaten ueberein, werden
#            keine weiteren erzeugt [126], [28], [8].
#
#   Stufe 4  Execution-Guided Selection                   -> C4b
#            Kandidaten werden ausgefuehrt, Ergebnismengen gefingerprintet
#            und per Mehrheitsvotum gewaehlt (Cross-Consistency wie
#            PET-SQL [52], Auswahl wie MCS-SQL [55]).
#
#   Stufe 5  Feedback-getriebene Reparatur                -> C4a
#            Schlagen alle Kandidaten fehl, wird die Warehouse-Fehlermeldung
#            an das LLM zurueckgespielt und korrigiert (Self-Debugging [80],
#            Refiner-Agent in MAC-SQL [48]), max. 2 Runden.
#
# BEWUSST NICHT uebernommen (mit Begruendung):
#   - Fine-Tuning-Paradigma (CodeS [24], XiYan-SQL [85]): Es werden
#     proprietaere Serving-Endpoints genutzt; SFT ist hier nicht anwendbar.
#   - Voll-agentische Mehrschritt-Exploration (Spider-Agent [42],
#     ReFoRCE [61] mit Column Exploration): Fuer ein interaktives Panel
#     ueber 7 gut dokumentierte Tabellen steht der Latenz-/Kostenaufwand
#     in keinem Verhaeltnis; adaptiert ist davon nur die Repair-Schleife.
#   - Value Retrieval ueber Zellinhalte (CHESS [56]): erfordert eine
#     Index-/Embedding-Infrastruktur, die im Projekt bewusst entfernt wurde.
#
# Kosten (Survey Abschnitt VI.B, Computational Efficiency): Die Pipeline
# macht bis zu 3 LLM-Calls pro Frage (1 Kandidat + bis zu 2 Repair-Runden).
# n_candidates=1 ist der Default: ein Kandidat, Repair-Schleife als Safety-Net.
# Schema Linking (regelbasiert, kein LLM): Keyword-Filter waehlt Tabellen-Docs
# pro Frage; Fallback auf Vollschema bei Ambiguitaet (recall over precision).
#
# WICHTIG — Effektiver Betriebsmodus bei n_candidates=1 (Default):
#   Im Default ist die Pipeline ein SINGLE-SHOT + REPAIR System (C2a + C4a).
#   Stufen 3 (Multi-Kandidaten-Diversitaet) und 4 (Execution-Guided Voting)
#   sind implementiert, aber erst bei n_candidates >= 2 aktiv. Diese Ent-
#   scheidung ist BEWUSST: Token-Kosten bleiben bei 1-3 Calls/Frage statt
#   n+2 Calls. Fuer einen vollstaendigen Benchmark-Vergleich C3+C4b vs. C0
#   muss n_candidates explizit auf >= 2 gesetzt werden.
#
# ----- PIPELINE (neu) --------------------------------------------------------
#
#   0. Input-Sanitisierung   (prompt_guard.sanitize_user_input, Layer 1)
#   1. Schema (gefiltert)    regelbasiertes Keyword-Linking (kein LLM, ~0 ms)
#   2. Few-Shot-Auswahl      maskierte Frage-Aehnlichkeit gegen Beispielpool
#   3. Kandidaten-Erzeugung  k SQL-Kandidaten PARALLEL via Strategie-Prompts
#   4. Output-Validierung    (prompt_guard.validate_sql_output, Layer 2,
#                             PRO KANDIDAT und pro Reparaturvorschlag)
#   5. Selektion             Ausfuehrung + Mehrheitsvotum ueber Ergebnisse
#   6. Reparatur             Fehler-Feedback-Schleife, falls alles scheitert
#   7. Ergebnis              pandas DataFrame des Gewinner-Kandidaten
#
# ----- SYSTEM-PROMPT AUFBAU --------------------------------------------------
#
#   Schema-Kontext (jetzt pro Frage GEFILTERT auf verlinkte Tabellen):
#     - Tabellen-Doku nur fuer selektierte Tabellen (+ Pflicht-Lookups)
#     - JOIN-Pfade nur, soweit die beteiligten Tabellen selektiert sind
#     - _SQL_TERMINOLOGY:    Statisches Schluessel-Mapping DE -> SQL
#     - _SQL_PIPELINE_DOCS_CORE/CASCADE: Business-Regeln; Regex-Kaskade
#                            (~65 Kategorien) in exakter Auswertungsreihen-
#                            folge als CASE-WHEN-Vorlage, Shop-Area-Mapping,
#                            Helper-Spalten-Hinweise (code_prefix, has_arrow,
#                            hall_ref, is_famos = pipeline-only); bleibt
#                            IMMER vollstaendig enthalten, da Fehlklassi-
#                            fikation hier teurer ist als Tokens
#
#   Ergaenzt durch (statisch, immer vorhanden):
#     - RULES:              Spark SQL Idiome (JOIN, CTE, Alias-Pflicht)
#     - EXAMPLES:           dynamisch selektierte Few-Shots (Stufe 2)
#     - ANTI-PATTERNS:      Haeufige Fehler (Timeout, AMBIGUOUS_REFERENCE)
#     - STRATEGY:           kandidatenspezifischer Reasoning-Block (Stufe 3)
#     - SECURITY SUFFIX:    Sandwich-Defense gegen Prompt Injection (Layer 3)
#
# ----- PROMPT-INJECTION-SCHUTZ (3 Schichten, unveraendert) -------------------
#
#   Schicht 1: Input-Sanitisierung  - blockiert Instruction-Overrides,
#              Role-Reassignment, System-Prompt-Extraction (prompt_guard.py)
#   Schicht 2: Output-Validierung   - nur SELECT/WITH erlaubt; wird jetzt
#              auf JEDEN Kandidaten und JEDE Reparatur angewendet
#   Schicht 3: Sandwich-Defense     - Security-Reminder am Prompt-Ende
#
# ----- THREADING -------------------------------------------------------------
#
#   call_llm_with_timeout():  Delegiert an llm_client.py (Daemon-Thread)
#   call_sql_with_timeout():  Delegiert an llm_client.py (Daemon-Thread)
#   Kandidaten-Generierung:   ThreadPoolExecutor (n parallele LLM-Calls)
#   Frischer WorkspaceClient pro Thread verhindert Connection-Pool-Exhaustion
#   ueber 55+ schnelle Benchmark-Requests hinweg.
#
# ----- TOKEN-METRIKEN --------------------------------------------------------
#
#   Alle Token-Werte werden lokal via utils.estimate_token_count() geschaetzt
#   (tiktoken cl100k_base / GPT-4o). API-Werte werden zusaetzlich erfasst,
#   aber NICHT fuer Metriken verwendet - nur fuer Logging.
#   WICHTIG: Bei der Mehr-Call-Pipeline sind alle Werte SUMMEN ueber alle
#   LLM-Calls einer Frage (Linking + Kandidaten + Reparaturen). Der Schema-
#   Anteil zaehlt jede Uebertragung, damit die Additivitaet erhalten bleibt:
#   total_prompt = schema_context + instructions + user_msg (aggregiert).
#
# ----- EXPORTS ---------------------------------------------------------------
#
#   - run_sql_pipeline(question, tables, wh_id, model_config) - Headless, SOTA
#   - render_sql_panel()                                      - Streamlit UI
#   - generate_sql_query(prompt, tables, model_config)        - Legacy-C0-Call,
#                                                erhalten fuer Benchmarks
#   - get_sql_schema_context()                                - Voll-Schema
#   - ALL_SQL_TABLES                                          - Tabellenliste
#
# =============================================================================

import hashlib
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

from config import TABLES, LLM_ENDPOINT
from llm_client import call_llm_with_timeout, call_sql_with_timeout
from prompt_guard import sanitize_user_input, validate_sql_output, sandwich_suffix_sql
from ui_components import render_large_timer, render_token_breakdown, SQL_ACCENT_COLOR, SQL_BG_FROM, SQL_BG_TO
from utils import estimate_token_count


# ---------------------------------------------------------------------------
# Session state keys
# Prefixed with "sql_" to avoid collisions with ontology_panel keys.
# ---------------------------------------------------------------------------
_SESSION_KEY  = "sql_last_total_time"   # total wall-clock seconds (float)
_RESULTS_KEY  = "sql_results"           # last pipeline result dict
_OVERHEAD_KEY = "sql_last_overhead_ms"  # thread + HTTP overhead in ms
_LLM_MS_KEY   = "sql_last_llm_ms"       # all LLM phases (link+gen+repair) in ms
_EXEC_MS_KEY  = "sql_last_exec_ms"      # SQL execution phase(s) in ms


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# All 7 normalised MRS data-product tables from the Genie Spaces.
ALL_SQL_TABLES = list(TABLES.values())

# Logical keys, in canonical order (mirrors config.TABLES).
_TABLE_KEYS = [
    "Orders", "ActualDates", "ActualDatesMD", "Faults",
    "FaultTypesMD", "FaultVehicleLocationsMD",
    "OrderModuleMapping",
]


# ---------------------------------------------------------------------------
# Timeout constants
#
# SQL Warehouse wait_timeout is clamped to [5, 50] s (API constraint).
# A 15 s Python-level buffer is added on top to absorb HTTP overhead
# between the API returning SUCCEEDED/RUNNING and Python processing the body.
# ---------------------------------------------------------------------------
_DEFAULT_SQL_TIMEOUT_S = 50   # default warehouse-side wait_timeout
_MAX_SQL_TIMEOUT_S     = 50   # API upper limit - values above -> BadRequest
_MIN_SQL_TIMEOUT_S     = 5    # API lower limit
_LLM_TIMEOUT_S         = 30   # wall-clock budget per LLM generation call


# ---------------------------------------------------------------------------
# SOTA pipeline knob (single user-facing control via model_config)
#
#   n_candidates   int   SQL candidates to generate (1=fast, 2=balanced, 5=max)
#                        Execution-voting auto-enabled when > 1.
#                        Repair is always on. Schema linking removed (7 tables
#                        make full-schema cheaper than an extra LLM roundtrip).
#
# Hardcoded internals (well-established defaults, no tuning needed):
# ---------------------------------------------------------------------------
_DEFAULT_N_CANDIDATES  = 1
_MAX_REPAIR_ROUNDS     = 2    # fixed: universally optimal (Survey VI.B)
_N_FEW_SHOTS           = 3    # pool has 8+ curated examples; 3 shots via Carina retrieval
_CANDIDATE_TEMP        = 0.7  # fixed: standard diversity temperature


# =============================================================================
# Schema context
#
# Table docs are stored per table so that schema linking (Stage 1) can build
# a question-specific, filtered schema context (C2b schema augmentation,
# cf. C3 [28] / CHESS [56]). _SQL_TERMINOLOGY and _SQL_PIPELINE_DOCS_CORE
# stay static and ALWAYS included. The category regex cascade (_CASCADE)
# is conditionally included only when the question mentions categories/shops.
# =============================================================================

_TABLE_DOCS: dict[str, str] = {
    "Orders": f"""{TABLES['Orders']}  (Orders / Auftraege)
   Key columns: poi STRING (PK), vin STRING, vehicle_type STRING, plant STRING,
                production_number STRING.""",

    "ActualDates": f"""{TABLES['ActualDates']}  (Checkpoint timestamps per module)
   Key columns: pmi STRING (FK->OrderModuleMapping.pmi_node), plant STRING,
                actual_checkpoint STRING (FK->ActualDatesMD.checkpoint),
                event_datetime TIMESTAMP, id_type STRING, id_value STRING
                (module-level identifier such as POI or PN - NOT a VIN filter).
   NEVER filter WHERE id_type = 'VIN' - VINs exist only in Orders.vin.""",

    "ActualDatesMD": f"""{TABLES['ActualDatesMD']}  (Checkpoint master data)
   Key columns: plant STRING, checkpoint STRING (PK), hall STRING,
                checkpoint_mark STRING, checkpoint_description STRING,
                location_progress INT (ORDER BY ASC fuer Reihenfolge).
   NOTE: checkpoint_category and shop_area are NOT stored columns - they are
   pipeline-computed. To filter by category or shop, use CASE WHEN on
   checkpoint_description or hall directly (see PIPELINE DOCS below).""",

    "Faults": f"""{TABLES['Faults']}  (Fault records - flat, one row per fault)
   Key columns: workstep_id BIGINT (PK), pmi STRING, plant STRING,
                id_type STRING, id_value STRING
                (module-level identifier such as POI or PN - NOT a VIN filter),
                s_checkpoint STRING (Symptom/Erkannt), s_fault_type STRING, s_fault_location STRING,
                s_code_system STRING (FK for vehicle location join),
                s_code_system_type STRING (FK for fault type join),
                s_input_timestamp TIMESTAMP, s_faultpriority STRING,
                c_checkpoint STRING (Cause/Verursacht), c_fault_type STRING, c_plant STRING,
                r_checkpoint STRING (Rework/Nacharbeit), r_plant STRING, r_input_timestamp TIMESTAMP.
   Symptom = where fault was discovered (s_*), Cause = root cause (c_*), Rework = fix (r_*).
   VIN lookups: Strategy A: Orders + OrderModuleMapping (omm.pmi_node = f.pmi).
   Strategy C (PN-Fallback): JOIN Orders o ON o.production_number = f.id_value
     AND o.plant = f.plant AND f.id_type = 'PN'.
   ⚠️ ~34% of validated Fault PMIs lack OMM entries (id_type='PN', mostly Assembly).
   For complete Fault→VIN coverage, ALWAYS UNION Strategy A + C (see Rule 2).""",

    "FaultTypesMD": f"""{TABLES['FaultTypesMD']}  (Fault type text lookup)
   Key columns: plant STRING, code_system STRING, fault_type STRING,
                type_of_fault_type STRING, language STRING, text STRING.
   Composite PK: (plant, code_system, fault_type, type_of_fault_type, language).
   Join: f.s_fault_type = ft.fault_type AND f.s_code_system_type = ft.code_system
         AND ft.language = 'de'.
   WARNING: Omitting code_system from the join causes fan-out (duplicate rows).
   CRITICAL: NEVER add f.plant = ft.plant to this JOIN. The plant column in
    FaultTypesMD is the code-system scope (values: 'All', 'VTAS', 'EF10' etc.),
    NOT the production plant '0500'. SPASS (the dominant code system) has
    plant='All' — joining on plant produces 0 results. See Rule 13.""",

    "FaultVehicleLocationsMD": f"""{TABLES['FaultVehicleLocationsMD']}  (Vehicle location text lookup)
   Key columns: plant STRING, code_system STRING, vehicle_location STRING,
                vehicle_location_type STRING, language STRING, text STRING.
   Composite PK: (plant, code_system, vehicle_location, language).
   Join: f.s_fault_location = fvl.vehicle_location AND f.s_code_system = fvl.code_system
         AND fvl.language = 'de'.
   WARNING: Omitting code_system from the join causes fan-out (duplicate rows).
   CRITICAL: NEVER add f.plant = fvl.plant to this JOIN. The plant column in
    FaultVehicleLocationsMD is the code-system scope, NOT the production plant.
    Joining on plant loses 90%+ of matches. See Rule 13.""",

    "OrderModuleMapping": f"""{TABLES['OrderModuleMapping']}  (Order->Module hierarchy)
   Key columns: plant STRING, poi STRING (FK->Orders.poi),
                pmi_root STRING, pmi_node STRING (FK->ActualDates.pmi, Faults.pmi),
                recursion_level INT.
   ⚠️ PARTIAL COVERAGE: Not all PMIs in Faults/ActualDates have a matching OMM
   record. In Assembly halls, ~50% of Fault PMIs lack OMM entries entirely.
   For complete PMI→VIN resolution, UNION with Strategy C (PN-based join via
   ActualDates where id_type='PN') — see Rule 2 in PIPELINE DOCS.""",

}


def _join_paths_for(keys: list[str]) -> str:
    """Return only the JOIN-path documentation relevant to the selected tables."""
    ks = set(keys)
    lines: list[str] = []
    if {"Orders", "OrderModuleMapping", "ActualDates", "ActualDatesMD"} <= ks:
        lines.append(
            "- VIN -> Checkpoints:\n"
            "    Orders o\n"
            "    JOIN OrderModuleMapping omm ON o.poi = omm.poi AND o.plant = omm.plant\n"
            "    JOIN ActualDates ad ON omm.pmi_node = ad.pmi AND omm.plant = ad.plant\n"
            "    JOIN ActualDatesMD amd ON ad.actual_checkpoint = amd.checkpoint AND ad.plant = amd.plant\n"
            "    WHERE o.vin = '<VIN>'"
        )
    if {"Orders", "OrderModuleMapping", "Faults"} <= ks:
        lines.append(
            "- VIN -> Faults (Strategy A, partial coverage ~60%):\n"
            "    Orders o\n"
            "    JOIN OrderModuleMapping omm ON o.poi = omm.poi AND o.plant = omm.plant\n"
            "    JOIN Faults f ON omm.pmi_node = f.pmi AND omm.plant = f.plant\n"
            "    WHERE o.vin = '<VIN>'\n"
            "    (NEVER via id_type/id_value - VINs exist only in Orders.vin)\n"
            "  Fault->VIN with PN-Fallback (UNION A+C for ~88% coverage):\n"
            "    -- Strategy A:\n"
            "    SELECT o.vin, f.workstep_id FROM Faults f\n"
            "      JOIN OrderModuleMapping omm ON omm.pmi_node = f.pmi AND omm.plant = f.plant\n"
            "      JOIN Orders o ON o.poi = omm.poi AND o.plant = omm.plant\n"
            "    UNION\n"
            "    -- Strategy C (PN-Fallback, +21k faults):\n"
            "    SELECT o.vin, f.workstep_id FROM Faults f\n"
            "      JOIN Orders o ON o.production_number = f.id_value\n"
            "        AND o.plant = f.plant AND f.id_type = 'PN'"
        )
    if {"Faults", "FaultTypesMD"} <= ks:
        lines.append(
            "- Fault type text:\n"
            "    JOIN FaultTypesMD ft ON f.s_fault_type = ft.fault_type AND f.s_code_system_type = ft.code_system\n"
            "    AND ft.language = 'de'"
        )
    if {"Faults", "FaultVehicleLocationsMD"} <= ks:
        lines.append(
            "- Vehicle location text:\n"
            "    JOIN FaultVehicleLocationsMD fvl ON f.s_fault_location = fvl.vehicle_location AND f.s_code_system = fvl.code_system\n"
            "    AND fvl.language = 'de'"
        )
    if {"Faults", "ActualDatesMD"} <= ks:
        lines.append(
            "- Fault checkpoint -> hall/description:\n"
            "    JOIN ActualDatesMD amd ON f.s_checkpoint = amd.checkpoint AND f.plant = amd.plant"
        )
    if not lines:
        return ""
    return "\n=== JOIN PATHS ===\n" + "\n".join(lines) + "\n"


_SQL_TERMINOLOGY = f"""
=== GERMAN -> ENGLISH TERMINOLOGY ===

Core entities:
  Auftrag / Bestellung        ->  {TABLES['Orders']} (poi = Order ID)
  Fahrzeug / FIN / VIN        ->  Orders.vin (ONLY VIN source; reach pmi tables via OrderModuleMapping)
  Fahrzeugtyp / Baureihe      ->  vehicle_type (in Orders)
  Modul / PMI                 ->  pmi (PLUS+ Module Identifier)
  Pruefpunkt / Checkpoint     ->  checkpoint / actual_checkpoint
  Halle / Hall                ->  hall (in ActualDatesMD)
  Bereich / Produktionsbereich ->  Bodyshop / Paintshop / Assembly via TRIM(hall)
                                   (see PIPELINE DOCS for the canonical hall lists and CASE template;
                                   checkpoint_category and shop_area are NOT stored columns)
  Fortschritt / Reihenfolge   ->  location_progress (INT, ORDER BY ASC)
  Zeitstempel / Datum         ->  event_datetime (ActualDates)
  Werk / Fabrik / Plant       ->  plant

Faults (Fehler):
  Fehler / Maengel            ->  {TABLES['Faults']} (one row per fault, NOT nested array)
  Fehlertyp / Fehlerart       ->  s_fault_type (code) -> JOIN FaultTypesMD for text
  Fehlerort / Fahrzeugposition->  s_fault_location (code) -> JOIN FaultVehicleLocationsMD for text
  Erkannt / Symptom           ->  s_* columns (s_checkpoint, s_input_timestamp, s_faultpriority)
  Verursacht / Ursache        ->  c_* columns (c_checkpoint, c_fault_type, c_plant)
  Nacharbeit / Rework         ->  r_* columns (r_checkpoint, r_plant, r_input_timestamp)
  Fehler-ID                   ->  workstep_id (BIGINT, unique per fault)

NEVER invent German column names. Always use the exact English column names above.
"""

_SQL_PIPELINE_DOCS_CORE = r"""
=== DERIVED / PIPELINE-ONLY LOGIC ===

The queryable source tables stay normalized. Several fields used in analysis are
computed upstream and are NOT stored columns; rebuild them inline in SQL when needed.

1. Pipeline-only fields (not queryable columns)
   checkpoint_category, shop_area, code_prefix, has_arrow, hall_ref, is_famos
   are computed from checkpoint_description / hall in Spark and must be recreated
   inline with CASE WHEN, regexp_extract and RLIKE.

2. Root POI / VIN resolution — three strategies, scope varies by area:
   Strategy A (direct module mapping — works for ALL areas):
     JOIN OrderModuleMapping omm ON omm.pmi_node = ad.pmi AND omm.plant = ad.plant
     JOIN Orders o ON o.poi = omm.poi
   Strategy B (parent traversal — PMI is a root module):
     JOIN OrderModuleMapping omm ON omm.pmi_root = ad.pmi AND omm.plant = ad.plant
     JOIN Orders o ON o.poi = omm.poi
     NOTE: Adds no new vehicle_types beyond A for any area (useful for VIN counts).
   Strategy C (production number — id_type='PN', primarily Assembly):
      For ActualDates-based queries:
       JOIN Orders o ON o.production_number = ad.id_value
                    AND o.plant             = ad.plant
                    AND ad.id_type          = 'PN'
      For Fault-based queries (DIRECT — no ActualDates intermediary needed):
       JOIN Orders o ON o.production_number = f.id_value
                    AND o.plant             = f.plant
                    AND f.id_type           = 'PN'
       ASSEMBLY: ~34% of validated Fault PMIs have NO entry in OrderModuleMapping.
       Strategy C resolves 82% of these orphaned Faults and is the ONLY path to
       8 of 18 Assembly vehicle_types. For ANY Fault→VIN query (Top-N VINs by
       fault count, faults per vehicle, etc.), ALWAYS UNION A + C.
       Coverage impact: Strategy A alone = 60% VIN coverage.
                        A + C (PN-Fallback) = 88% VIN coverage (+21k faults).
     BODYSHOP: id_type='PN' is 0.01% of events — Strategy C adds nothing.
               Strategy A alone covers all meaningful vehicle_types (10/11).
               The missing 1 ('W 223') is a dirty-data typo for 'W223' in Orders.
     PAINTSHOP: id_type='BN' (Body Number) dominates (99.99%). NO join path
               from BN to vehicle_type exists in this data model (see Rule 21).
               Use Strategy A only; coverage is structurally limited.

3. Fault enrichment
   Fault text requires joins to fault_types_md and fault_vehicle_locations_md.
   Fault checkpoint metadata comes from actual_dates_md joined on s_/c_/r_checkpoint.

4. Fault arrays
   faults_worksteps arrays exist only in the derived output table. The queryable
   source Faults table remains FLAT (one row per fault).

=== SHOP AREA CLASSIFICATION (from TRIM(hall)) ===

Bodyshop  -> 2, 3/2, 5, 7/1, 7/4, 8, 9, 15, 15/2, 15/3, 30
Paintshop -> 34, 34/4, 44, 44/7
Assembly  -> 03/10, 3/10, 32, 38, 38/5, 42, 44/6, 46, 46/1, 56, 510, 513, 515, 516, 514

IMPORTANT: Only these three areas exist in scope. Rows with NULL hall or
unmapped hall values MUST be excluded (Rule 12).

=== KEY RULES FOR AI ===

1. Sort checkpoints by location_progress — direction depends on query intent:
   - "erste/erster/first" (earliest station) → ORDER BY location_progress ASC
   - "letzte/letzter/last" (latest station)  → ORDER BY location_progress DESC
   NEVER add LIMIT 1: multiple checkpoints can share the same location_progress
   (ties). Return ALL rows; the leading rows form the answer.
   NEVER use event_datetime for production order; it is a scan timestamp only.
2. Always TRIM(hall) before hall/shop matching.
3. checkpoint_category, shop_area, code_prefix, has_arrow, hall_ref, is_famos are
   NOT stored columns. NEVER reference amd.checkpoint_category or amd.shop_area.
4. To filter by shop area, rebuild inline with:
   CASE WHEN TRIM(amd.hall) IN ('2','3/2','5','7/1','7/4','8','9','15','15/2','15/3','30') THEN 'Bodyshop'
        WHEN TRIM(amd.hall) IN ('34','34/4','44','44/7') THEN 'Paintshop'
        WHEN TRIM(amd.hall) IN ('03/10','3/10','32','38','38/5','42','44/6','46','46/1','56','510','513','515','516','514') THEN 'Assembly'
   END
   IMPORTANT: No ELSE branch — NULL halls and unmapped halls are excluded by Rule 12.
   Always pair this CASE with a WHERE filter on the canonical hall list (see Rule 12).
5. To filter by category, emit the CASE WHEN regex cascade in the EXACT order shown
   below. If patterns overlap, higher-priority rules win; when in doubt, compute the
   full cascade in a CTE and filter on the derived category.
6. All description regexes need (?i) because Spark RLIKE is case-sensitive by default.
   Keep the word boundaries (\bIO\b, \bNA\b, \bBT\b) to avoid false positives.
7. ALWAYS use COUNT(DISTINCT f.workstep_id) when counting faults.
8. r_checkpoint IS NULL does NOT mean no rework; check r_plant IS NOT NULL.
9. Orders.vin is the ONLY VIN source. Use the documented VIN join paths above.
   The module hierarchy fans out, so deduplicate vehicle-level results with DISTINCT
   or COUNT(DISTINCT ...).
10. German terminology shortcuts: Nacharbeit=Rework, Hochzeit=Marriage,
    Fahrwerk=Chassis, Vormontage=Pre-assembly, Einlauf=Entry,
    Auslauf=Exit, Schleifen=Grinding, Fertigmeldung=Completion.
11. GROUND TRUTH: ActualDates contains the REAL production events.
    a) Structural queries (halls, checkpoints, stations, description lookups,
       mark lookups, sequence listings): NEVER use ActualDatesMD alone.
       This applies to ALL queries whose answer comes from ActualDatesMD —
       including simple lookups such as "What is the description of the
       checkpoint with mark RZ1 in hall 9?". Master data can contain orphaned
       or decommissioned checkpoints that have no real scan data.
       Always EXISTS-filter against ActualDates:
         WHERE EXISTS (SELECT 1 FROM ActualDates ad
                       WHERE ad.actual_checkpoint = amd.checkpoint
                         AND ad.plant = amd.plant)
       When an analysis window applies, add the date filter inside the subquery
       to confirm the checkpoint was actively used in the requested period:
         AND ad.event_datetime BETWEEN '<date_start>' AND '<date_end>'
       Correct example — checkpoint description lookup with reality filter:
         SELECT amd.checkpoint, amd.checkpoint_description,
                amd.checkpoint_mark, TRIM(amd.hall) AS hall,
                amd.location_progress
         FROM ActualDatesMD amd
         WHERE amd.plant = '0500'
           AND TRIM(amd.hall) = '9'
           AND amd.checkpoint_mark = 'RZ1'
           AND EXISTS (SELECT 1 FROM ActualDates ad
                       WHERE ad.actual_checkpoint = amd.checkpoint
                         AND ad.plant = amd.plant)
         ORDER BY amd.location_progress ASC
       Wrong — may return checkpoints never actually scanned:
         SELECT amd.checkpoint_description
         FROM ActualDatesMD amd
         WHERE TRIM(amd.hall) = '9' AND amd.checkpoint_mark = 'RZ1'  -- WRONG
    b) Fault queries: For EVERY query that touches the Faults table — regardless of
       whether ActualDatesMD is joined or not — ALWAYS validate that the fault's PMI
       has a checkpoint scan matching the SPECIFIC fault checkpoint being analysed.
       Use the TYPE-SPECIFIC form that matches your query's focus:
         - Discovered faults (s_*):  AND ad.actual_checkpoint = f.s_checkpoint
         - Caused faults (c_*):      AND ad.actual_checkpoint = f.c_checkpoint
         - Rework faults (r_*):      AND ad.actual_checkpoint = f.r_checkpoint
         - Comparing two types:      AND ad.actual_checkpoint IN (f.s_checkpoint, f.c_checkpoint)
       Full pattern:
          AND EXISTS (SELECT 1 FROM ActualDates ad
                      WHERE ad.pmi = f.pmi AND ad.plant = f.plant
                        AND ad.actual_checkpoint = f.<prefix>_checkpoint)
       NEVER use the generic IN (f.s_checkpoint, f.c_checkpoint, f.r_checkpoint)
       unless ALL three types are genuinely relevant to the question.
    Both patterns ensure answers reflect ONLY real production data.
12. MANDATORY AREA SCOPE: Results MUST only include data from Bodyshop,
    Paintshop (Lackiererei), or Assembly. NEVER return rows with Unknown or Other
    areas. When the query touches ActualDatesMD (hall column), ALWAYS add:
      AND TRIM(amd.hall) IN ('2','3/2','5','7/1','7/4','8','9','15','15/2','15/3','30',
                             '34','34/4','44','44/7',
                             '03/10','3/10','32','38','38/5','42','44/6','46','46/1','56','510','513','515','516','514')
    This excludes NULL halls and any unmapped hall values. When the query computes
    a shop_area CASE WHEN, exclude 'Unknown' and 'Other' with:
      HAVING bereich IN ('Rohbau', 'Lackiererei', 'Montage')  -- or WHERE on outer CTE
    This rule has NO exceptions — even if the user asks for "all areas" or "all halls",
    restrict to the three known production areas only.
13. METADATA TABLE PLANT SEMANTICS: The plant column in FaultTypesMD and
    FaultVehicleLocationsMD is the code-system SCOPE (values: 'All', 'VTAS',
    'EF10'), NOT the production plant '0500'. NEVER include f.plant = ft.plant
    or f.plant = fvl.plant in JOIN conditions with these tables.
    Correct FaultTypesMD join (3 keys + language, NO plant):
      JOIN FaultTypesMD ft ON f.s_fault_type = ft.fault_type
        AND f.s_code_system_type = ft.code_system AND ft.language = 'de'
    Correct FaultVehicleLocationsMD join (2 keys + language, NO plant):
      JOIN FaultVehicleLocationsMD fvl ON f.s_fault_location = fvl.vehicle_location
        AND f.s_code_system = fvl.code_system AND fvl.language = 'de'
    Adding a plant equality condition yields 0 rows for SPASS (the dominant
    code system where plant='All').
14. VIN / VEHICLE_TYPE GROUND TRUTH: NEVER query Orders alone for VIN lists,
    vehicle_type distributions, or any VIN/model-level analysis. Orders may
    contain entries for vehicles that were never actually produced (cancelled,
    planned-only, or test records). ALWAYS join through ActualDates to confirm
    that the vehicle has real production scan data.
    Correct pattern — start FROM ActualDates, join upward to Orders:
      SELECT DISTINCT o.vehicle_type
      FROM ActualDates ad
      JOIN OrderModuleMapping omm ON omm.pmi_node = ad.pmi AND omm.plant = ad.plant
      JOIN Orders o ON o.poi = omm.poi
      WHERE o.vehicle_type IS NOT NULL
    Or with Strategy C (Assembly PN-Join):
      SELECT DISTINCT o.vehicle_type
      FROM ActualDates ad
      JOIN Orders o ON o.production_number = ad.id_value
        AND o.plant = ad.plant AND ad.id_type = 'PN'
    Wrong — returns unvalidated entries without production evidence:
      SELECT DISTINCT o.vehicle_type FROM Orders o  -- WRONG: no ActualDates join
      SELECT COUNT(DISTINCT o.vin) FROM Orders o    -- WRONG: includes non-produced VINs
    This rule applies to ALL Orders-based queries: VIN counts, vehicle_type
    listings, Baureihen-Verteilungen, production_number lookups, etc.
    The ONLY exception is when the user explicitly asks about order data
    itself (e.g. "how many orders exist") without production context.
"""

# Conditional: only included when the question involves category/shop filtering.
# Saves ~1,779 tokens x n_candidates for ~80% of questions that don't need it.
_SQL_PIPELINE_DOCS_CASCADE = r"""
=== CHECKPOINT CATEGORY CLASSIFICATION (regex cascade, FIRST MATCH WINS) ===

checkpoint_category is derived from checkpoint_description via a sequential
when/otherwise cascade. To reproduce it in SQL, build a CASE WHEN chain in
EXACTLY the order below (CASE is also first-match-wins):
  CASE WHEN amd.checkpoint_description RLIKE '<pattern>' THEN '<CATEGORY>'
       ... ELSE 'OTHER' END
Keep the (?i) flag: patterns are case-insensitive, Spark RLIKE is not.

Evaluation order (category <- RLIKE pattern):

Quality & Testing (highest priority):
   1. IS_TEST          <- (?i)IS-TEST|ISTESTER|IS TEST
   2. FUNCTION_TEST    <- (?i)KPS|FUNKTIONEN
   3. DYNAMIC_TEST     <- (?i)DYN\.|DYNAMISCH
   4. ACCEPTANCE_TEST  <- (?i)\bFAT\b|\bFOT\b
   5. QUALITY          <- (?i)FAMOS|Q-TOR|PQD|QK-|SCHLUSSPR|SCHLUSSABN|INQA|AUDIT
   6. IO_NIO           <- (?i)\bIO\b|\bNIO\b|\bi[oO]\b|n[iI][oO]
   7. CHECK_STATION    <- (?i)\bCHECK\b
   8. GEOMETRY         <- (?i)\bGEO\b|GEO-PAL

Screw stations:
   9. SCREW_STATION    <- ^SS\b   (description starts with "SS")

Logistics & Data:
  10. LOGISTICS        <- (?i)\bLOG\b.*\bABRUF\b|\bTEILEABRUF\b|\bWARENKORB\b|\bSETBILDUNG\b|\bKOBALT\b
  11. JOB_DATA         <- (?i)\bKNR\b|JOBDAT
  12. JWS_SEQUENCE     <- (?i)\bJWS\b
  13. BODY_DATA        <- (?i)KAROSSERIEDATEN|KAROSSERIE
  14. MDS_DATA         <- (?i)\bMDS\b.*LESEN|\bMDS\b
  15. SEQUENCE_PREVIEW <- (?i)SEQUENZ|SEQ[\.-]|SeqV
  16. COMMISSIONING    <- (?i)KOMMISSION

Assembly processes (specific to generic):
  17. MARRIAGE         <- (?i)HOCHZEIT|MARRIAGE
  18. PRE_ASSEMBLY     <- (?i)VORMONTAGE|COCKPITVORM|T[UÜ]RENVORM|INTEGRALTR
  19. FINAL_ASSEMBLY   <- (?i)\bEMO\b|ENDMONTAGE|AGGREGATEEND
  20. CHASSIS          <- (?i)FAHRWERK|FW\d
  21. INTERIOR         <- (?i)\bINNENEINB
  22. ASSEMBLY         <- (?i)MONTAGE|MON\d   (broad catch-all for assembly)

Seat systems:
  23. SEAT_SYSTEM      <- (?i)SITZ|SIDEBAG|GURT|LEHNE|KISSEN|FAHRERSITZ|FONDLEHNE|FONDKISSEN

Body / paint processes:
  24. BOLZEN           <- (?i)BOLZEN
  25. JOINING          <- (?i)STANZN|CLINCH
  26. KTL_COATING      <- (?i)\bKTL\b
  27. PAINT            <- (?i)LACK|FARBE|GRUNDIERT|POLIER
  28. WAX              <- (?i)WACHS
  29. GRINDING         <- (?i)SCHLEIFEN|SCHLIFF
  30. PRESSURE         <- (?i)DRUCK
  31. BIW_START        <- (?i)ROHBAU
  32. ACOUSTIC_FOAM    <- (?i)AKUSTIK|AFA_|\bAFA\b
  33. SOLA_PAINT_MEAS  <- (?i)\bSOLA\b

End-of-line & rework:
  34. END_OF_LINE      <- (?i)\bEOL\b|FINISH\b
  35. REWORK           <- (?i)NACHARBEIT|NACHBE|REWORK|NA-CLIENT|\bREP\.|REPARATUR|\bNA\b$|\bNA-\d
  36. CORRECTION       <- (?i)KORREKTUR|KORRIGIERT
  37. IMPLAUSIBLE      <- (?i)UNPLAUSIBEL

Measurement & scheduling:
  38. MEASUREMENT      <- (?i)MESS|PR[UÜ]F|VERMESS
  39. SCHEDULING       <- (?i)TERMIN\b|ST-TERMIN

Flow & status:
  40. ENTRY            <- (?i)EINLAUF|EINL\.|\bEINGANG\b
  41. EXIT             <- (?i)AUSLAUF|AUSL\.
  42. GATE_IN          <- (?i)EINSCHLEUS
  43. GATE_OUT         <- (?i)AUSSCHLEUS|AUSCHLEUS
  44. SCRAP            <- (?i)VERSCHROTT|SCRAP
  45. PILOT_SERIES     <- (?i)NULLSERIE|PILOT
  46. AMG_SPECIAL      <- (?i)\bAMG\b
  47. COMPLETION       <- (?i)FERTIGMELDUNG|FERTIGM\.|\bFERTIG\b|ABSCHNITTFERTIG
  48. CAMPAIGN         <- (?i)AKTIONIERUNG|\bAKTION\b

Build tree / PLUS:
  49. BUILD_TREE       <- (?i)\bBT\b
  50. PLUS_STATION     <- (?i)\bPLUS\b

Storage & buffer:
  51. STORAGE          <- (?i)AUSLAGERUNG|EINGELAGERT|EINLAGER|\bLAGER\b
  52. BUFFER           <- (?i)\bPUFFER\b

Infrastructure & visualization:
  53. VISUALIZATION    <- (?i)\bVISU\b
  54. PLC_SPS          <- (?i)\bSPS\b|\bPLC\b|H\d{2,3}SPS
  55. LINE_CONVEYOR    <- (?i)\bBAND\b|\bLINIE\b|\bLINE\b
  56. RING_LINE        <- (?i)RINGLEITUNG
  57. CONSOLE          <- (?i)\bPULT\b
  58. NAD_SYSTEM       <- (?i)\bNAD\b

Equipment / device types:
  59. BOLTING_TECH     <- (?i)H\d{2,3}SC\b
  60. PDA_DEVICE       <- (?i)H\d{2,3}PDA\b
  61. PC_STATION       <- (?i)H\d{2,3}PC\b
  62. ANDON_TRACK      <- (?i)H\d{2,3}AT\b
  63. AGV_TRANSPORT    <- (?i)H\d{2,3}FTS
  64. CONTROL_POINT    <- (?i)\bCP\b|NC46|BLOCK\b

Capacity status:
  65. RESERVED         <- (?i)\bRESERVIERT\b
  66. FREE_STATUS      <- (?i)\bFREI\b

Transfer / routing:
  67. TRANSFER         <- ->   (arrow-based transfer indicators)

Broad catch-all:
  68. LOCATION_PATH    <- \d+_\d+_.*_\d+_   (structured location path codes)
  69. OTHER            <- fallback when nothing above matched
"""


# ---------------------------------------------------------------------------
# Lightweight keyword detector: does the question involve checkpoint categories
# or shop areas? If not, the 65-entry regex cascade is irrelevant (~1,779 tok).
# ---------------------------------------------------------------------------
_CATEGORY_KEYWORDS_RE = re.compile(
    r"(?i)(kategori|category|shop|bodyshop|paintshop|assembly|lackier|"
    r"montage|rohbau|qualit|famos|nacharbeit|rework|einlauf|auslauf|"
    r"checkpoint.{0,20}(typ|art|klasse|classif)|bereich|area|"
    r"\bIS.TEST|FUNKTION|DYNAMISCH|\bFAT\b|\bFOT\b|\bEOL\b|"
    r"vormontage|fahrwerk|hochzeit|marriage|schraubstation|"
    r"CASE\s+WHEN.*RLIKE)"
)


def _needs_category_cascade(question: str) -> bool:
    """Return True if the question likely requires the category regex cascade."""
    return bool(_CATEGORY_KEYWORDS_RE.search(question))

# ---------------------------------------------------------------------------
# Rule-based (no-LLM) schema linker — keyword trigger patterns
#
# Three regex families cover the three main table groups. OrderModuleMapping
# is a bridge and gets added automatically when Orders joins Faults/ActualDates.
# Conservative threshold: only filter when >= 2 table docs can be dropped.
# ---------------------------------------------------------------------------
_LINK_ORDERS_RE = re.compile(
    r"(?i)(vin|fin|fahrzeug|vehicle|baureihe|production.number|\bpoi\b|auftrag|bestellung)"
)
_LINK_FAULTS_RE = re.compile(
    r"(?i)(fehler|fault|mangel|defekt|nacharbeit|rework|symptom|verursach|reparatur|workstep"
    r"|fehlertyp|fehlerort|priorität|faultpriority|fehleranzahl)"
)
_LINK_CHECKPOINT_RE = re.compile(
    r"(?i)(checkpoint|pruefpunkt|halle|bodyshop|paintshop|lackier|rohbau|montage|assembly"
    r"|location.progress|reihenfolge|fortschritt|bereich|durchlaufen"
    r"|\bstation|\berste|\bletzte|\bfirst|\blast|karosseriebau)"
)
_LINK_DATETIME_RE = re.compile(
    r"(?i)(zeitstempel|timestamp|\bwann\b|event.datetime|\bdatum\b|\bdate\b|uhrzeit)"
)


def _rule_link_tables(question: str) -> list[str] | None:
    """Keyword-based schema linking with zero LLM calls (~0 ms overhead).

    Maps the question to the minimal relevant table subset using regex rules.
    Conservative design — returns None (full schema) when:
      - no recognisable group keywords are found, OR
      - the selected subset would be >= 6 tables (drop < 2 docs, savings negligible).

    Companion tables (FaultTypesMD, FaultVehicleLocationsMD, ActualDatesMD)
    always follow their base tables.  Orders ALWAYS pulls in ActualDates,
    ActualDatesMD, and OrderModuleMapping — because VIN checkpoint flow,
    Assembly vehicle_type resolution (PN-Join), and timestamp context all
    depend on the bridge path Orders → OrderModuleMapping → ActualDates.

    Returns:
        List of logical table keys in canonical order, or None for full schema.
    """
    needs_orders      = bool(_LINK_ORDERS_RE.search(question))
    needs_faults      = bool(_LINK_FAULTS_RE.search(question))
    needs_checkpoints = bool(_LINK_CHECKPOINT_RE.search(question))
    needs_datetime    = bool(_LINK_DATETIME_RE.search(question))

    # event_datetime lives in ActualDates — pull the checkpoint group along.
    if needs_datetime:
        needs_checkpoints = True

    # No signal at all → full schema (safe default).
    if not (needs_orders or needs_faults or needs_checkpoints):
        return None

    selected: set[str] = set()

    if needs_faults:
        # Always include lookup companions so fault-text joins stay possible.
        # Also include Orders + OMM + ActualDates for Fault->VIN resolution
        # (Strategy A via OMM + Strategy C via PN-Fallback, Rule 2).
        selected.update({"Faults", "FaultTypesMD", "FaultVehicleLocationsMD",
                         "Orders", "OrderModuleMapping", "ActualDates", "ActualDatesMD"})

    if needs_checkpoints:
        # ActualDates plus its master-data companion.
        selected.update({"ActualDates", "ActualDatesMD"})

    if needs_orders:
        # Orders nearly always requires ActualDates: VIN checkpoint flow,
        # Assembly vehicle_type resolution (PN-Join, Rule 2/Pattern 23),
        # and timestamp context all depend on ActualDates + bridge table.
        selected.update({"Orders", "ActualDates", "ActualDatesMD", "OrderModuleMapping"})

    # Only apply subset when it drops >= 2 table docs; otherwise send full schema.
    if len(selected) > len(_TABLE_KEYS) - 2:
        return None

    return [k for k in _TABLE_KEYS if k in selected]


def build_schema_context(selected_keys: list[str] | None = None, *, include_cascade: bool = True) -> str:
    """Build the schema context, optionally filtered to the linked tables.

    Stage-1 output (schema linking) feeds this builder. Filtering happens at
    TABLE granularity only; terminology and pipeline docs are always included
    because they encode business rules (see header rationale, Distillery [58]).

    Args:
        selected_keys: Logical table keys to include. None or empty -> all 7.

    Returns:
        Combined schema string: table docs + join paths + terminology + docs.
    """
    keys = [k for k in _TABLE_KEYS if (not selected_keys) or k in selected_keys]
    if not keys:
        keys = list(_TABLE_KEYS)

    if len(keys) == len(_TABLE_KEYS):
        header = "=== AVAILABLE TABLES (7 normalised MRS data-products) ==="
    else:
        header = (
            f"=== AVAILABLE TABLES ({len(keys)} of 7 MRS data-products, "
            "pre-selected via schema linking for THIS question) ==="
        )

    numbered = "\n\n".join(
        f"{i}. {_TABLE_DOCS[k]}" for i, k in enumerate(keys, start=1)
    )
    return (
        f"\n{header}\n\n{numbered}\n"
        + _join_paths_for(keys)
        + _SQL_TERMINOLOGY
        + (_SQL_PIPELINE_DOCS_CORE.rstrip() + "\n\n"
           + _SQL_PIPELINE_DOCS_CASCADE
           if include_cascade
           else _SQL_PIPELINE_DOCS_CORE)
    )


def get_sql_schema_context() -> str:
    """Return the FULL SQL schema context string (all 7 tables).

    Retained for benchmark_metrics, which measures the unfiltered schema
    token cost. The SOTA pipeline itself uses build_schema_context() with
    the linked table subset.
    """
    return build_schema_context(None)


# =============================================================================
# Static prompt building blocks (rules / anti-patterns / strategies)
# =============================================================================

_RULES_BLOCK = """=== RULES (SQL output & syntax) ===
1. Output ONLY the raw SQL - no markdown fences, no explanation.
2. Use fully qualified table names (catalog.schema.table).
3. JOIN tables using the documented keys (poi, pmi, checkpoint, etc.).
4. Column names are case-sensitive - use exact names from the schema.
5. Always qualify column names with table alias (o.vin, NOT vin) - avoids
   AMBIGUOUS_REFERENCE in UNION/JOIN/subqueries.
6. Never use bare SELECT * (selecting all columns without aggregation) - it
   returns millions of rows and causes timeouts. COUNT(*) and other aggregate
   functions are perfectly fine and encouraged for counting.
7. NEVER add LIMIT unless the user explicitly asks for a top-N 
    result (e.g. "top 10", "die 5 häufigsten"). Always return the full dataset
    so users can see all records and count totals accurately. Do NOT add LIMIT
    to "protect" against large results — use aggregation or filtering instead.
8. MANDATORY AREA SCOPE: Only return data from Bodyshop, Paintshop, or Assembly.
   Always filter TRIM(hall) to the canonical hall lists (see PIPELINE DOCS Rule 12).
   Never include rows with NULL/unknown halls or unmapped areas.
9. CTE-FIRST for multi-strategy queries: When combining two or more join strategies
   via UNION (e.g. VIN/vehicle_type resolution from PIPELINE DOCS Rule 2), ALWAYS
    express the UNION inside a WITH...AS CTE, then SELECT from it:
      For DISTINCT vehicle_type queries (Assembly): UNION A + C
      For COUNT/coverage queries (Assembly): UNION A + B + C
      (Bodyshop/Paintshop: Strategy A alone suffices for both.)
      WITH resolved AS (
          <Strategy A query>
          UNION
          <Strategy B query>  -- only for count/coverage in Assembly
         UNION
         <Strategy C query>
     )
     SELECT DISTINCT ... FROM resolved ORDER BY ...
   NEVER inline a UNION as a subquery or bare FROM clause.
   Also use CTEs whenever an intermediate deduplication step is needed before the
   final aggregation or SELECT — it keeps query logic readable and avoids double-joins."""

# Candidate strategies (Stage 3, C3a). Diversity comes primarily from these
# prompt variants (MCS-SQL [55] "multiple prompts"); temperature is added on
# top where the endpoint supports it. Reasoning stays INTERNAL because the
# Layer-2 validator only accepts a bare SELECT/WITH statement.
_STRATEGY_BLOCKS: dict[str, str] = {
    "direct": (
        "=== STRATEGY ===\n"
        "Translate the question into one SQL query in a single direct pass.\n"
        "Structure the query using WITH...AS CTEs: put validation/filtering\n"
        "logic (EXISTS checks, area filters) in a CTE, then SELECT from it.\n"
        "Only trivial single-table queries (1-2 JOINs, no EXISTS) may skip CTEs."
    ),
    "decompose": (
        "=== STRATEGY (divide and conquer, CHASE-SQL style) ===\n"
        "Internally decompose the question into logical sub-steps. Express\n"
        "EACH sub-step as a named WITH...AS CTE:\n"
        "  - CTE 1: data validation (EXISTS checks against ActualDates)\n"
        "  - CTE 2: resolution / enrichment (VIN->PMI, hall->area, text joins)\n"
        "  - Final SELECT: aggregation or output from the CTEs\n"
        "When sub-steps combine via UNION, put the UNION inside a CTE.\n"
        "Output only the final SQL starting with WITH — no sub-step explanation."
    ),
    "plan": (
        "=== STRATEGY (execution-order planning, QDecomp style) ===\n"
        "Internally reason in SQL execution order: FROM/JOIN -> WHERE ->\n"
        "GROUP BY -> HAVING -> SELECT -> ORDER BY. Verify every\n"
        "join key and alias against the schema while planning.\n"
        "Structure the result using CTEs: heavy joins and validation go into\n"
        "named CTEs, the outer SELECT stays clean. Output only the final SQL."
    ),
}
_STRATEGY_ORDER = ["direct", "decompose", "plan"]


# =============================================================================
# Stage 2: Few-shot example pool + similarity selection (C2a, DAIL-SQL style)
# =============================================================================

_EXAMPLE_POOL: list[dict] = [
    # ---- Pattern 1: VIN → Checkpoint resolution (CTE, 4-join, ORDER BY progress) --
    {
        "question": "Stationsabfolge inkl. Halle und Zeitstempel fuer VIN W1KLF4HB0TA278667 ausgeben",
        "sql": f"""WITH vin_scans AS (
    -- CTE: resolve VIN -> PMI -> checkpoint scans
    SELECT DISTINCT
        ad.actual_checkpoint,
        ad.event_datetime,
        ad.plant
    FROM {TABLES['Orders']} o
    JOIN {TABLES['OrderModuleMapping']} omm
        ON o.poi = omm.poi AND o.plant = omm.plant
    JOIN {TABLES['ActualDates']} ad
        ON omm.pmi_node = ad.pmi AND omm.plant = ad.plant
    WHERE o.vin = 'W1KLF4HB0TA278667'
)
SELECT
    vs.actual_checkpoint,
    amd.checkpoint_description,
    amd.hall,
    vs.event_datetime,
    amd.location_progress
FROM vin_scans vs
JOIN {TABLES['ActualDatesMD']} amd
    ON vs.actual_checkpoint = amd.checkpoint
   AND vs.plant = amd.plant
ORDER BY amd.location_progress ASC""",
    },
    # ---- Pattern 2: VIN-level DISCOVERED fault count in Bodyshop (s_checkpoint) -
    # NOTE: Bodyshop is >99% id_type='BN'/'VD', so PN-Fallback adds nothing here.
    # Strategy A alone suffices for Bodyshop. See Pattern 27 for Assembly with PN-Fallback.
    {
        "question": "Top-10 Fahrzeuge nach Anzahl entdeckter Fehler (Symptom-Checkpoint) im Karosseriebau — sortiert absteigend",
        "sql": f"""WITH validated_bodyshop_faults AS (
    -- CTE: faults validated against real scans, filtered to Bodyshop halls
    SELECT f.workstep_id, f.pmi, f.plant
    FROM {TABLES['Faults']} f
    JOIN {TABLES['ActualDatesMD']} amd
        ON f.s_checkpoint = amd.checkpoint
       AND f.plant = amd.plant
    WHERE TRIM(amd.hall) IN ('2','3/2','5','7/1','7/4','8','9','15','15/2','15/3','30')
      AND EXISTS (
          SELECT 1 FROM {TABLES['ActualDates']} ad
          WHERE ad.pmi = f.pmi AND ad.plant = f.plant
            AND ad.actual_checkpoint = f.s_checkpoint
      )
),
vin_faults AS (
    -- CTE: resolve PMI -> VIN via Orders + OrderModuleMapping
    SELECT o.vin, vf.workstep_id
    FROM validated_bodyshop_faults vf
    JOIN {TABLES['OrderModuleMapping']} omm
        ON omm.pmi_node = vf.pmi AND omm.plant = vf.plant
    JOIN {TABLES['Orders']} o
        ON o.poi = omm.poi AND o.plant = omm.plant
)
SELECT vin, COUNT(DISTINCT workstep_id) AS fehler_anzahl
FROM vin_faults
GROUP BY vin
ORDER BY fehler_anzahl DESC
LIMIT 10""",
    },
    # ---- Pattern 3: VIN → discovered faults with text joins (s_* columns) -
    # Uses UNION Strategy A + C for complete coverage (this VIN has faults only via PN).
    {
        "question": "Zeige alle entdeckten Fehler (Symptome) fuer VIN W1KLF4HB0TA278667 mit lesbarem Fehlertyp und Fehlerort",
        "sql": f"""WITH vin_faults AS (
    -- Strategy A: VIN -> OMM -> Faults
    SELECT DISTINCT
        f.workstep_id, f.s_checkpoint, f.s_fault_type,
        f.s_fault_location, f.s_code_system_type, f.s_code_system,
        f.s_input_timestamp
    FROM {TABLES['Orders']} o
    JOIN {TABLES['OrderModuleMapping']} omm
        ON o.poi = omm.poi AND o.plant = omm.plant
    JOIN {TABLES['Faults']} f
        ON omm.pmi_node = f.pmi AND omm.plant = f.plant
    WHERE o.vin = 'W1KLF4HB0TA278667'
      AND EXISTS (
          SELECT 1 FROM {TABLES['ActualDates']} ad
          WHERE ad.pmi = f.pmi AND ad.plant = f.plant
            AND ad.actual_checkpoint = f.s_checkpoint
      )

    UNION

    -- Strategy C: VIN -> production_number -> Faults (PN-Fallback)
    SELECT DISTINCT
        f.workstep_id, f.s_checkpoint, f.s_fault_type,
        f.s_fault_location, f.s_code_system_type, f.s_code_system,
        f.s_input_timestamp
    FROM {TABLES['Orders']} o
    JOIN {TABLES['Faults']} f
        ON f.id_value = o.production_number
       AND f.plant = o.plant
       AND f.id_type = 'PN'
    WHERE o.vin = 'W1KLF4HB0TA278667'
      AND EXISTS (
          SELECT 1 FROM {TABLES['ActualDates']} ad
          WHERE ad.pmi = f.pmi AND ad.plant = f.plant
            AND ad.actual_checkpoint = f.s_checkpoint
      )
)
SELECT
    vf.workstep_id,
    vf.s_checkpoint AS erkannt_checkpoint,
    vf.s_fault_type,
    ft.text AS fehlertyp_text,
    vf.s_fault_location,
    fvl.text AS fehlerort_text,
    vf.s_input_timestamp
FROM vin_faults vf
LEFT JOIN {TABLES['FaultTypesMD']} ft
    ON vf.s_fault_type = ft.fault_type
   AND vf.s_code_system_type = ft.code_system
   AND ft.language = 'de'
LEFT JOIN {TABLES['FaultVehicleLocationsMD']} fvl
    ON vf.s_fault_location = fvl.vehicle_location
   AND vf.s_code_system = fvl.code_system
   AND fvl.language = 'de'""",
    },    # ---- Pattern 5: CAUSED faults per checkpoint (c_checkpoint, Ursache) --------
    {
        "question": "An welchen Checkpoints werden die meisten Fehler verursacht (Ursache-Checkpoint, sortiert nach Anzahl)?",
        "sql": f"""WITH caused_faults AS (
    -- CTE: faults with a known cause checkpoint, validated against real scans
    SELECT f.workstep_id, f.c_checkpoint, f.plant
    FROM {TABLES['Faults']} f
    WHERE f.c_checkpoint IS NOT NULL
      AND EXISTS (
        SELECT 1 FROM {TABLES['ActualDates']} ad
        WHERE ad.pmi = f.pmi AND ad.plant = f.plant
          AND ad.actual_checkpoint = f.c_checkpoint
    )
)
SELECT
    cf.c_checkpoint AS verursacht_checkpoint,
    amd.checkpoint_description,
    amd.hall,
    COUNT(DISTINCT cf.workstep_id) AS fehler_anzahl
FROM caused_faults cf
JOIN {TABLES['ActualDatesMD']} amd
    ON cf.c_checkpoint = amd.checkpoint
   AND cf.plant = amd.plant
GROUP BY cf.c_checkpoint, amd.checkpoint_description, amd.hall
ORDER BY fehler_anzahl DESC""",
    },    # ---- Pattern 14: DISCOVERED faults by shop area (s_checkpoint → hall CASE) -
    {
        "question": "Verteilung entdeckter Fehler (Symptom-Checkpoint) ueber die drei Produktionsbereiche (Rohbau, Lackiererei, Montage)",
        "sql": f"""SELECT
    CASE
        WHEN TRIM(amd.hall) IN ('2','3/2','5','7/1','7/4','8','9','15','15/2','15/3','30')
             THEN 'Rohbau'
        WHEN TRIM(amd.hall) IN ('34','34/4','44','44/7')
             THEN 'Lackiererei'
        WHEN TRIM(amd.hall) IN ('03/10','3/10','32','38','38/5','42','44/6','46','46/1','56','510','513','515','516','514')
             THEN 'Montage'
    END                              AS bereich,
    COUNT(DISTINCT f.workstep_id)    AS fehler_anzahl
FROM {TABLES['Faults']} f
JOIN {TABLES['ActualDatesMD']} amd
    ON f.s_checkpoint = amd.checkpoint
   AND f.plant        = amd.plant
WHERE EXISTS (
    SELECT 1 FROM {TABLES['ActualDates']} ad
    WHERE ad.pmi = f.pmi AND ad.plant = f.plant
      AND ad.actual_checkpoint = f.s_checkpoint
)
  AND TRIM(amd.hall) IN ('2','3/2','5','7/1','7/4','8','9','15','15/2','15/3','30',
                          '34','34/4','44','44/7',
                          '03/10','3/10','32','38','38/5','42','44/6','46','46/1','56','510','513','515','516','514')
GROUP BY 1
ORDER BY fehler_anzahl DESC""",
    },
    # ---- Pattern 15: VINs with REWORK faults (r_checkpoint, Nacharbeit) ---------
    # Uses UNION A+C for complete VIN coverage (PN-Fallback for faults without OMM).
    {
        "question": "Top-10 VINs mit den meisten nachgearbeiteten Fehlern (Rework-Checkpoint) auflisten",
        "sql": f"""WITH rework_faults AS (
    -- CTE: faults that have a rework checkpoint (r_plant IS NOT NULL)
    SELECT f.workstep_id, f.pmi, f.plant, f.r_checkpoint, f.id_type, f.id_value
    FROM {TABLES['Faults']} f
    WHERE f.r_plant IS NOT NULL
      AND f.r_checkpoint IS NOT NULL
      AND EXISTS (
          SELECT 1 FROM {TABLES['ActualDates']} ad
          WHERE ad.pmi = f.pmi AND ad.plant = f.plant
            AND ad.actual_checkpoint = f.r_checkpoint
      )
),
vin_rework AS (
    -- Strategy A: OMM-based VIN resolution
    SELECT o.vin, rf.workstep_id
    FROM rework_faults rf
    JOIN {TABLES['OrderModuleMapping']} omm
        ON omm.pmi_node = rf.pmi AND omm.plant = rf.plant
    JOIN {TABLES['Orders']} o
        ON o.poi = omm.poi AND o.plant = omm.plant
    WHERE o.vin IS NOT NULL

    UNION

    -- Strategy C: PN-Fallback for faults without OMM entry
    SELECT o.vin, rf.workstep_id
    FROM rework_faults rf
    JOIN {TABLES['Orders']} o
        ON o.production_number = rf.id_value
       AND o.plant = rf.plant
    WHERE rf.id_type = 'PN'
      AND o.vin IS NOT NULL
)
SELECT vin, COUNT(DISTINCT workstep_id) AS nacharbeit_anzahl
FROM vin_rework
GROUP BY vin
ORDER BY nacharbeit_anzahl DESC
LIMIT 10""",
    },    # ---- Pattern 18: Cause vs Discovery comparison (c_checkpoint vs s_checkpoint) -
    {
        "question": "Vergleich: An welchem Checkpoint wurde der Fehler verursacht (c_checkpoint) vs. entdeckt (s_checkpoint)?",
        "sql": f"""WITH validated_faults AS (
    -- CTE: faults with both cause and symptom checkpoints, validated
    SELECT f.workstep_id, f.c_checkpoint, f.s_checkpoint, f.plant
    FROM {TABLES['Faults']} f
    WHERE f.c_checkpoint IS NOT NULL
      AND f.s_checkpoint IS NOT NULL
      AND EXISTS (
          SELECT 1 FROM {TABLES['ActualDates']} ad
          WHERE ad.pmi = f.pmi AND ad.plant = f.plant
            AND ad.actual_checkpoint IN (f.s_checkpoint, f.c_checkpoint)
      )
)
SELECT
    vf.c_checkpoint          AS verursacht_checkpoint,
    vf.s_checkpoint          AS erkannt_checkpoint,
    COUNT(DISTINCT vf.workstep_id) AS anzahl
FROM validated_faults vf
GROUP BY vf.c_checkpoint, vf.s_checkpoint
ORDER BY anzahl DESC""",
    },
    # ---- Pattern 21: DISCOVERED faults in Paintshop (s_checkpoint → hall) -------
    # NOTE: Faults in the Lackiererei use id_type='BN' (Body Number).
    # They join to ActualDatesMD via s_checkpoint + plant.
    # f.id_value for BN faults is the body_number (Karosserienummer).
    # LIMITATION: There is NO join path from body_number to vehicle_type
    # (Orders has no body_number column) — vehicle_type queries for
    # Lackiererei faults are NOT answerable in this data model.
    {
        "question": "Paintshop-Qualitaetsuebersicht: Fehlervolumen und betroffene Karosserien je Lackiererei-Halle mit CTE-Validierung",
        "sql": f"""WITH validated_paint_faults AS (
    -- CTE: validated faults in Paintshop halls
    SELECT f.workstep_id, f.id_value, f.s_checkpoint, f.plant
    FROM {TABLES['Faults']} f
    JOIN {TABLES['ActualDatesMD']} amd
        ON f.s_checkpoint = amd.checkpoint
       AND f.plant = amd.plant
    WHERE TRIM(amd.hall) IN ('34','34/4','44','44/7')
      AND EXISTS (
          SELECT 1 FROM {TABLES['ActualDates']} ad
          WHERE ad.pmi = f.pmi AND ad.plant = f.plant
            AND ad.actual_checkpoint = f.s_checkpoint
      )
)
SELECT
    TRIM(amd.hall)                   AS halle,
    COUNT(DISTINCT vpf.workstep_id)  AS fehler_anzahl,
    COUNT(DISTINCT vpf.id_value)     AS betroffene_karosserien
FROM validated_paint_faults vpf
JOIN {TABLES['ActualDatesMD']} amd
    ON vpf.s_checkpoint = amd.checkpoint
   AND vpf.plant = amd.plant
GROUP BY TRIM(amd.hall)
ORDER BY fehler_anzahl DESC""",
    },
    # ---- Pattern 22: Structural query grounded in actual_dates (CTE + Rule 11) --
    # Shows that hall/checkpoint questions MUST be validated against ActualDates
    # to return only production-relevant results, not dormant master data.
    # NOTE: No LIMIT 1 — multiple checkpoints can share the same minimum
    # location_progress (ties). Return ALL of them; the first rows are the answer.
    {
        "question": "Wo startet die Lackiererei-Linie — welcher Checkpoint hat den kleinsten location_progress in Paintshop-Hallen?",
        "sql": f"""WITH active_paintshop_checkpoints AS (
    -- CTE: only checkpoints with real scan data in Paintshop halls
    SELECT
        TRIM(amd.hall) AS halle,
        amd.checkpoint,
        amd.checkpoint_description,
        amd.location_progress
    FROM {TABLES['ActualDatesMD']} amd
    WHERE TRIM(amd.hall) IN ('34','34/4','44','44/7')
      AND EXISTS (
          SELECT 1 FROM {TABLES['ActualDates']} ad
          WHERE ad.actual_checkpoint = amd.checkpoint
            AND ad.plant = amd.plant
      )
)
SELECT halle, checkpoint, checkpoint_description, location_progress
FROM active_paintshop_checkpoints
ORDER BY location_progress ASC""",
    },
    # ---- Pattern 22b: Superlative LAST — ORDER BY DESC, CTE style, no LIMIT 1 -
    # "letzte/last" → ORDER BY location_progress DESC (highest progress = latest).
    # No LIMIT 1 — multiple checkpoints can share the same maximum
    # location_progress (ties). Return ALL; the first rows are the answer.
    {
        "question": "Paintshop-Linienende: Checkpoints mit hoechstem Fortschrittswert und Hallenzuordnung in der Lackierstrecke",
        "sql": f"""WITH active_paintshop_checkpoints_desc AS (
    -- CTE: only checkpoints with real scan data in Paintshop halls
    SELECT
        TRIM(amd.hall) AS halle,
        amd.checkpoint,
        amd.checkpoint_description,
        amd.location_progress
    FROM {TABLES['ActualDatesMD']} amd
    WHERE TRIM(amd.hall) IN ('34','34/4','44','44/7')
      AND EXISTS (
          SELECT 1 FROM {TABLES['ActualDates']} ad
          WHERE ad.actual_checkpoint = amd.checkpoint
            AND ad.plant = amd.plant
      )
)
SELECT halle, checkpoint, checkpoint_description, location_progress
FROM active_paintshop_checkpoints_desc
ORDER BY location_progress DESC""",
    },
    # ---- Pattern 23: Assembly vehicle_type — 3-strategy UNION (Rule 2) ----------
    # Direct module mapping (Strategy A) covers only ~60% of Assembly PMIs.
    # ~40% of Assembly PMIs have NO entry in OrderModuleMapping and are resolved
    # ONLY via production number (id_type='PN', Strategy C).
    # Strategy A alone: 10 of 18 vehicle_types.  A+C UNION: all 18.
    # Strategy B (pmi_root join) adds no new vehicle_types beyond A.
    {
        "question": "Strategie-A+C-Validierung: Welche vehicle_types sind ueber beide Aufloesungswege in Montage-Hallen erreichbar?",
        "sql": f"""WITH vin_resolved AS (
    -- Strategy A: direct module mapping (pmi_node)
    SELECT DISTINCT ad.pmi, ad.plant, o.vehicle_type
    FROM {TABLES['ActualDates']} ad
    JOIN {TABLES['ActualDatesMD']} amd
         ON amd.checkpoint = ad.actual_checkpoint AND amd.plant = ad.plant
    JOIN {TABLES['OrderModuleMapping']} omm
         ON omm.pmi_node = ad.pmi AND omm.plant = ad.plant
    JOIN {TABLES['Orders']} o ON o.poi = omm.poi
    WHERE TRIM(amd.hall) IN ('03/10','3/10','32','38','38/5','42','44/6','46','46/1','56','510','513','515','516','514')
      AND o.vehicle_type IS NOT NULL

    UNION

    -- Strategy C: production number fallback (id_type='PN')
    -- Resolves 8 Assembly vehicle_types not reachable via OrderModuleMapping
    SELECT DISTINCT ad.pmi, ad.plant, o.vehicle_type
    FROM {TABLES['ActualDates']} ad
    JOIN {TABLES['ActualDatesMD']} amd
         ON amd.checkpoint = ad.actual_checkpoint AND amd.plant = ad.plant
    JOIN {TABLES['Orders']} o
         ON o.production_number = ad.id_value
        AND o.plant             = ad.plant
        AND ad.id_type          = 'PN'
    WHERE TRIM(amd.hall) IN ('03/10','3/10','32','38','38/5','42','44/6','46','46/1','56','510','513','515','516','514')
      AND o.vehicle_type IS NOT NULL
)
SELECT DISTINCT vehicle_type AS fahrzeugmodell
FROM vin_resolved
ORDER BY fahrzeugmodell""",
    },
    # ---- Pattern 24: Rohbau vehicle_type — Strategy A only (Rule 2) -----------
    # Bodyshop id_types: BN (52%), VD (47%), PN (0.01%).
    # Strategy C (PN join) adds no new vehicle_types for Bodyshop — omit it.
    # Strategy A covers all 10 meaningful vehicle_types.
    # NOTE: 'W 223' (mid-string space) is a dirty-data typo for 'W223' in some
    # Orders rows. Use REPLACE(o.vehicle_type,' ','') if exact deduplication
    # is needed; otherwise both appear as separate rows.
    {
        "question": "Strategie-A-Abdeckung im Rohbau: OrderModuleMapping-basierte vehicle_type-Aufloesung fuer Karosseriebau-Checkpoints",
        "sql": f"""SELECT DISTINCT o.vehicle_type AS fahrzeugmodell
FROM {TABLES['ActualDates']} ad
JOIN {TABLES['ActualDatesMD']} amd
     ON amd.checkpoint = ad.actual_checkpoint AND amd.plant = ad.plant
JOIN {TABLES['OrderModuleMapping']} omm
     ON omm.pmi_node = ad.pmi AND omm.plant = ad.plant
JOIN {TABLES['Orders']} o ON o.poi = omm.poi
WHERE TRIM(amd.hall) IN ('2','3/2','5','7/1','7/4','8','9','15','15/2','15/3','30')
  AND o.vehicle_type IS NOT NULL
ORDER BY fahrzeugmodell""",
    },
    # ---- Pattern 25: Lackiererei vehicle_type — BN limitation (Rule 21) ------
    # Paintshop id_types: BN (99.99%), PN (< 0.01%).
    # BN (Body Number/Karosserienummer) has NO join path to vehicle_type —
    # Orders has no body_number column (Rule 21). Strategy A via OrderModuleMapping
    # is the only viable path; coverage is structurally partial for Lackiererei.
    # Do NOT attempt Strategy C (PN join) — only 3 PN events exist, adds nothing.
    {
        "question": "Welche vehicle_types sind ueber OrderModuleMapping in der Lackiererei auflösbar?",
        "sql": f"""-- Lackiererei vehicle_type resolution: Strategy A only.
-- WARNING: id_type='BN' (Karosserienummer) dominates (99.99%) but has
-- NO join path to vehicle_type in this data model (Rule 21). Results
-- reflect only the subset of PMIs present in OrderModuleMapping.
SELECT DISTINCT o.vehicle_type AS fahrzeugmodell
FROM {TABLES['ActualDates']} ad
JOIN {TABLES['ActualDatesMD']} amd
     ON amd.checkpoint = ad.actual_checkpoint AND amd.plant = ad.plant
JOIN {TABLES['OrderModuleMapping']} omm
     ON omm.pmi_node = ad.pmi AND omm.plant = ad.plant
JOIN {TABLES['Orders']} o ON o.poi = omm.poi
WHERE TRIM(amd.hall) IN ('34','34/4','44','44/7')
  AND o.vehicle_type IS NOT NULL
ORDER BY fahrzeugmodell""",
    },
    # ---- Pattern 26: Assembly VIN count — 3-strategy UNION A+B+C (Rule 2) ----
    # For COUNT/coverage queries, Strategy B (pmi_root) adds +27.7% PMIs and
    # +10.5% VINs in Assembly. Combined with C, this is the most complete path.
    # This pattern serves as the LLM's executable template for count queries.
    {
        "question": "Wie viele unterschiedliche Fahrzeuge (VINs) durchliefen Montage-Checkpoints?",
        "sql": f"""WITH vin_resolved AS (
    -- Strategy A: direct module mapping (pmi_node)
    SELECT DISTINCT o.vin, ad.plant
    FROM {TABLES['ActualDates']} ad
    JOIN {TABLES['ActualDatesMD']} amd
         ON amd.checkpoint = ad.actual_checkpoint AND amd.plant = ad.plant
    JOIN {TABLES['OrderModuleMapping']} omm
         ON omm.pmi_node = ad.pmi AND omm.plant = ad.plant
    JOIN {TABLES['Orders']} o ON o.poi = omm.poi
    WHERE TRIM(amd.hall) IN ('03/10','3/10','32','38','38/5','42','44/6','46','46/1','56','510','513','515','516','514')

    UNION

    -- Strategy B: parent traversal (pmi_root) — +10.5% VINs in Assembly
    SELECT DISTINCT o.vin, ad.plant
    FROM {TABLES['ActualDates']} ad
    JOIN {TABLES['ActualDatesMD']} amd
         ON amd.checkpoint = ad.actual_checkpoint AND amd.plant = ad.plant
    JOIN {TABLES['OrderModuleMapping']} omm
         ON omm.pmi_root = ad.pmi AND omm.plant = ad.plant
    JOIN {TABLES['Orders']} o ON o.poi = omm.poi
    WHERE TRIM(amd.hall) IN ('03/10','3/10','32','38','38/5','42','44/6','46','46/1','56','510','513','515','516','514')

    UNION

    -- Strategy C: production number fallback (id_type='PN')
    SELECT DISTINCT o.vin, ad.plant
    FROM {TABLES['ActualDates']} ad
    JOIN {TABLES['ActualDatesMD']} amd
         ON amd.checkpoint = ad.actual_checkpoint AND amd.plant = ad.plant
    JOIN {TABLES['Orders']} o
         ON o.production_number = ad.id_value
        AND o.plant             = ad.plant
        AND ad.id_type          = 'PN'
    WHERE TRIM(amd.hall) IN ('03/10','3/10','32','38','38/5','42','44/6','46','46/1','56','510','513','515','516','514')
)
SELECT COUNT(*) AS anzahl_fahrzeuge
FROM vin_resolved""",
    },
    # ---- Pattern 27: Fault->VIN with PN-Fallback (Assembly, Rule 2 Strategy A+C) -
    # CRITICAL: ~34% of validated Fault PMIs have NO OMM entry (id_type='PN').
    # Strategy A alone gives only 60% VIN coverage for faults.
    # UNION with Strategy C (f.id_value = o.production_number) adds +21k faults (88% total).
    # Use this pattern for ALL Fault->VIN queries that are NOT area-restricted to Bodyshop.
    {
        "question": "Top-10 Fahrzeuge nach Gesamtzahl entdeckter Fehler (alle Bereiche) mit PN-Fallback fuer vollstaendige Abdeckung",
        "sql": f"""WITH validated_faults AS (
    -- CTE: all discovered faults validated against real scans
    SELECT f.workstep_id, f.pmi, f.plant, f.id_type, f.id_value
    FROM {TABLES['Faults']} f
    WHERE EXISTS (
        SELECT 1 FROM {TABLES['ActualDates']} ad
        WHERE ad.pmi = f.pmi AND ad.plant = f.plant
          AND ad.actual_checkpoint = f.s_checkpoint
    )
),
vin_faults AS (
    -- Strategy A: OMM-based VIN resolution (~60% coverage)
    SELECT DISTINCT o.vin, vf.workstep_id
    FROM validated_faults vf
    JOIN {TABLES['OrderModuleMapping']} omm
        ON omm.pmi_node = vf.pmi AND omm.plant = vf.plant
    JOIN {TABLES['Orders']} o
        ON o.poi = omm.poi AND o.plant = omm.plant
    WHERE o.vin IS NOT NULL

    UNION

    -- Strategy C: PN-Fallback for faults without OMM entry (+28pp coverage)
    SELECT DISTINCT o.vin, vf.workstep_id
    FROM validated_faults vf
    JOIN {TABLES['Orders']} o
        ON o.production_number = vf.id_value
       AND o.plant = vf.plant
    WHERE vf.id_type = 'PN'
      AND o.vin IS NOT NULL
)
SELECT vin, COUNT(DISTINCT workstep_id) AS fehler_anzahl
FROM vin_faults
GROUP BY vin
ORDER BY fehler_anzahl DESC
LIMIT 10""",
    },
]


_STOPWORDS = {
    "der", "die", "das", "den", "dem", "ein", "eine", "einen", "und", "oder",
    "fuer", "für", "mit", "von", "im", "in", "am", "an", "auf", "zu", "zur",
    "zum", "ist", "sind", "hat", "haben", "wie", "was", "welche", "welcher",
    "welches", "zeige", "zeig", "alle", "aller", "bitte", "mir", "pro",
    "the", "a", "an", "of", "to", "for", "on", "and", "or", "show", "all",
    "me", "per", "with",
}

_VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b", re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d+([.,/]\d+)*\b")


def _mask_question(question: str) -> set[str]:
    """Mask literals and return the skeleton token set of a question.

    DAIL-SQL [8] shows that masking domain literals before similarity
    matching selects structurally similar demonstrations instead of
    literally similar ones. VINs and numbers are replaced by placeholders,
    stopwords are dropped, remaining tokens are lowercased.
    Used as Jaccard fallback when the Carina endpoint is unavailable.
    """
    q = _VIN_RE.sub(" <vin> ", question)
    q = _NUM_RE.sub(" <num> ", q)
    tokens = re.split(r"[^a-zA-Z0-9<>äöüÄÖÜß]+", q.lower())
    return {t for t in tokens if t and t not in _STOPWORDS}


# ---------------------------------------------------------------------------
# Carina embedding cache for the few-shot pool
#
# Pool embeddings are built ONCE on first call and reused for every
# subsequent select_few_shot_examples() invocation. The cache holds a
# flat list of float[] vectors, index-aligned with _EXAMPLE_POOL.
# None signals that the Carina endpoint was unreachable — callers fall
# back to Jaccard similarity automatically.
# ---------------------------------------------------------------------------
_POOL_EMB_CACHE: list[list[float]] | None = None
_POOL_EMB_ATTEMPTED: bool = False   # avoid re-attempting after a failed call


def _get_pool_embeddings() -> list[list[float]] | None:
    """Return Carina embeddings for every entry in _EXAMPLE_POOL.

    Results are cached module-globally after the first successful call
    (a single batch request for all pool questions). Returns None when
    the Carina endpoint is unavailable so the caller can fall back to
    Jaccard without raising an exception.
    """
    global _POOL_EMB_CACHE, _POOL_EMB_ATTEMPTED
    if _POOL_EMB_CACHE is not None:
        return _POOL_EMB_CACHE
    if _POOL_EMB_ATTEMPTED:
        return None   # previous attempt failed; don't retry on every call
    _POOL_EMB_ATTEMPTED = True
    try:
        from graph_rag import _embed_texts  # local import avoids circular risk
        questions = [ex["question"] for ex in _EXAMPLE_POOL]
        _POOL_EMB_CACHE = _embed_texts(questions)
        return _POOL_EMB_CACHE
    except Exception:
        return None


def select_few_shot_examples(question: str, k: int = _N_FEW_SHOTS) -> list[dict]:
    """Select the k most similar examples from the curated pool.

    Primary path: Carina 3.0 cosine similarity on embedded question text
    (DAIL-SQL [8] embedding retrieval). Pool embeddings are built once
    at first call and cached; only the incoming question requires a live
    Carina call per pipeline run.

    Fallback path: Jaccard overlap of masked token sets — lightweight,
    dependency-free, used when the Carina endpoint is unavailable.

    Deterministic tie-break by pool index keeps results reproducible.
    """
    if k <= 0:
        return []

    # --- Primary: Carina cosine-similarity retrieval --------------------------
    try:
        pool_embs = _get_pool_embeddings()
        if pool_embs is not None:
            from graph_rag import _embed_texts, _cosine_sim
            q_emb = _embed_texts([question])[0]
            scored = [
                (_cosine_sim(q_emb, e_emb), idx, ex)
                for idx, (ex, e_emb) in enumerate(zip(_EXAMPLE_POOL, pool_embs))
            ]
            scored.sort(key=lambda t: (-t[0], t[1]))
            return [ex for _, _, ex in scored[:k]]
    except Exception:
        pass  # fall through to Jaccard

    # --- Fallback: Jaccard on masked token sets --------------------------------
    q_tokens = _mask_question(question)
    scored_j = []
    for idx, ex in enumerate(_EXAMPLE_POOL):
        e_tokens = _mask_question(ex["question"])
        union = q_tokens | e_tokens
        score = (len(q_tokens & e_tokens) / len(union)) if union else 0.0
        scored_j.append((score, idx, ex))
    scored_j.sort(key=lambda t: (-t[0], t[1]))
    return [ex for _, _, ex in scored_j[:k]]


def _render_examples_block(examples: list[dict]) -> str:
    """Render selected few-shots into the system-prompt EXAMPLES section."""
    if not examples:
        return ""
    parts = ["=== EXAMPLES (selected by question similarity) ==="]
    for i, ex in enumerate(examples, start=1):
        parts.append(f'Example {i}:\nUser: "{ex["question"]}"\n\n{ex["sql"]}')
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Pre-warm pool embeddings at module load
#
# Calling _get_pool_embeddings() here embeds all 20 pool questions in ONE
# Carina batch request during app startup. Every subsequent pipeline run
# then hits the module-global cache instantly instead of paying the ~200 ms
# cold-start penalty on the first interactive query.
# Errors are silently suppressed; the pipeline falls back to Jaccard.
# ---------------------------------------------------------------------------
_get_pool_embeddings()


# =============================================================================
# Low-level LLM call helper
# =============================================================================

def _strip_sql_fences(text: str) -> str:
    """Strip markdown code fences some models emit despite instructions."""
    sql = (text or "").strip()
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1] if "\n" in sql else sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]
    return sql.strip()


def _normalize_sql(sql: str) -> str:
    """Canonicalise SQL for candidate de-duplication and agreement checks."""
    return re.sub(r"\s+", " ", sql or "").strip().rstrip(";").lower()


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    cfg: dict,
    temperature: float | None,
    timeout_s: float,
) -> tuple[str, float, int, int]:
    """Single LLM call via the shared daemon-thread client.

    Returns (raw_text, elapsed_s, api_prompt_tokens, api_completion_tokens).
    Temperature is only passed when the endpoint supports it
    (cfg["supports_temperature"] = False disables it for reasoning models).
    """
    query_kwargs: dict = {}
    if temperature is not None and cfg.get("supports_temperature", True):
        query_kwargs["temperature"] = temperature

    t0 = time.perf_counter()
    response = call_llm_with_timeout(
        endpoint=cfg.get("endpoint", LLM_ENDPOINT),
        messages=[
            ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=ChatMessageRole.USER,   content=user_prompt),
        ],
        query_kwargs=query_kwargs,
        timeout_s=timeout_s,
    )
    elapsed = time.perf_counter() - t0

    text  = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    p_tok = getattr(usage, "prompt_tokens", 0) or 0
    c_tok = getattr(usage, "completion_tokens", 0) or 0
    return text, elapsed, p_tok, c_tok


# =============================================================================
# Stage 3: Candidate generation (C3a strategies + C3b self-consistency)
# =============================================================================

def _build_generation_prompt(
    schema_context: str,
    table_list: str,
    examples_block: str,
    strategy_key: str,
) -> str:
    """Assemble one candidate's system prompt (schema + rules + shots + strategy)."""
    parts = [
        "You are a Databricks SQL expert. Translate the user's question into a "
        "single efficient Spark SQL query.",
        schema_context,
        _RULES_BLOCK,
    ]
    if examples_block:
        parts.append(examples_block)
    parts.append(_STRATEGY_BLOCKS[strategy_key])
    parts.append(f"Tables available for this query:\n{table_list}")
    return "\n\n".join(parts) + "\n" + sandwich_suffix_sql()


def generate_sql_candidates(
    clean_question: str,
    selected_keys: list[str],
    selected_tables: list[str],
    model_config: dict | None = None,
) -> tuple[list[dict], str, float, dict]:
    """Generate up to n diverse, Layer-2-validated SQL candidates.

    Diversity sources (MCS-SQL [55], CHASE-SQL [77]):
      - strategy prompt variants (direct / decompose / plan, cycled)
      - temperature 0.0 for candidate 1, 0.7 diversity temp afterwards
        (only if the endpoint supports temperature)

    Self-consistency short-circuit [126]: if candidate 2 normalises to the
    same SQL as candidate 1, generation stops early - strong agreement makes
    further sampling redundant and saves cost (Survey VI.B).

    Returns:
        (candidates, schema_context, llm_elapsed_total, tokens_agg)
        candidates: list of dicts with keys
            sql, strategy, temperature, valid (Layer-2), guard_reason
        tokens_agg: summed tiktoken parts over ALL generation calls:
            system, schema, user, output, api_prompt, api_completion, calls
    """
    cfg = model_config or {"endpoint": LLM_ENDPOINT, "supports_temperature": True}
    n         = max(1, int(cfg.get("n_candidates", _DEFAULT_N_CANDIDATES)))
    cand_temp = _CANDIDATE_TEMP
    n_shots   = _N_FEW_SHOTS
    timeout   = cfg.get("llm_timeout_s", _LLM_TIMEOUT_S)

    needs_cascade = _needs_category_cascade(clean_question)
    schema_context = build_schema_context(selected_keys, include_cascade=needs_cascade)
    schema_tokens  = estimate_token_count(schema_context)
    examples_block = _render_examples_block(
        select_few_shot_examples(clean_question, n_shots)
    )
    table_list = "\n".join(f"  - {t}" for t in selected_tables)
    user_tokens = estimate_token_count(clean_question)

    candidates: list[dict] = []
    seen_norm: set[str] = set()
    tokens_agg = {
        "system": 0, "schema": 0, "user": 0, "output": 0,
        "api_prompt": 0, "api_completion": 0, "calls": 0,
    }

    # Build all prompts upfront — they differ only in strategy + temperature.
    tasks: list[tuple[str, str, float]] = []
    for i in range(n):
        strategy = _STRATEGY_ORDER[i % len(_STRATEGY_ORDER)]
        temperature = 0.0 if i == 0 else cand_temp
        system_prompt = _build_generation_prompt(
            schema_context, table_list, examples_block, strategy
        )
        tasks.append((system_prompt, strategy, temperature))

    # --- Parallel LLM calls (thread-safe: each call uses a fresh client) ---
    def _gen_one(task: tuple[str, str, float]) -> dict:
        sys_prompt, strat, temp = task
        try:
            raw, elapsed, p_tok, c_tok = _call_llm(
                sys_prompt, clean_question, cfg, temp, timeout
            )
            return {"raw": raw, "elapsed": elapsed, "p_tok": p_tok,
                    "c_tok": c_tok, "strategy": strat, "temperature": temp,
                    "sys_prompt": sys_prompt, "error": None}
        except Exception as exc:
            return {"raw": "", "elapsed": 0.0, "p_tok": 0, "c_tok": 0,
                    "strategy": strat, "temperature": temp,
                    "sys_prompt": "", "error": str(exc)}

    with ThreadPoolExecutor(max_workers=min(n, 4)) as executor:
        results = list(executor.map(_gen_one, tasks))

    # --- Process results sequentially (validation + dedup, order-stable) ---
    llm_elapsed_total = max((r["elapsed"] for r in results), default=0.0)

    for r in results:
        if r["error"]:
            candidates.append({
                "sql": "", "strategy": r["strategy"], "temperature": r["temperature"],
                "valid": False, "guard_reason": f"LLM-Call fehlgeschlagen: {r['error']}",
            })
            continue

        tokens_agg["system"]  += estimate_token_count(r["sys_prompt"])
        tokens_agg["schema"]  += schema_tokens
        tokens_agg["user"]    += user_tokens
        tokens_agg["api_prompt"]     += r["p_tok"]
        tokens_agg["api_completion"] += r["c_tok"]
        tokens_agg["calls"]   += 1

        sql = _strip_sql_fences(r["raw"])
        tokens_agg["output"] += estimate_token_count(sql)

        # Layer 2 guard PER candidate.
        is_safe, unsafe_reason = validate_sql_output(sql)
        norm = _normalize_sql(sql)
        if norm not in seen_norm:
            seen_norm.add(norm)
            candidates.append({
                "sql": sql, "strategy": r["strategy"], "temperature": r["temperature"],
                "valid": bool(is_safe),
                "guard_reason": None if is_safe else unsafe_reason,
            })

    return candidates, schema_context, llm_elapsed_total, tokens_agg


# =============================================================================
# Pipeline step: SQL execution on Databricks SQL Warehouse (unchanged core)
# =============================================================================

def execute_sql_query(
    sql: str,
    warehouse_id: str,
    timeout_s: int = _DEFAULT_SQL_TIMEOUT_S,
) -> tuple[pd.DataFrame, float]:
    """Execute a Spark SQL statement on a Databricks SQL Warehouse.

    Clamps ``timeout_s`` to the API-allowed range [5, 50] and adds a 15 s
    Python-level buffer for HTTP overhead (see llm_client.call_sql_with_timeout).

    Returns:
        (df, elapsed): result DataFrame and wall-clock seconds of this call.

    Raises:
        ValueError:   Statement timed out (state RUNNING) or failed.
        TimeoutError: Python-level thread timeout exceeded.
    """
    from databricks.sdk.service.sql import StatementState

    timeout_s      = min(max(timeout_s, _MIN_SQL_TIMEOUT_S), _MAX_SQL_TIMEOUT_S)
    python_timeout = timeout_s + 15  # 15 s buffer for HTTP overhead

    t0 = time.perf_counter()
    response = call_sql_with_timeout(
        sql=sql,
        warehouse_id=warehouse_id,
        wait_timeout_s=timeout_s,
        python_timeout_s=python_timeout,
    )
    elapsed = time.perf_counter() - t0

    state = response.status.state
    if state == StatementState.RUNNING:
        raise ValueError(
            f"SQL-Abfrage hat das Timeout ueberschritten (>{timeout_s}s). "
            "Versuche eine spezifischere Abfrage oder fuege WHERE-Bedingungen hinzu."
        )
    if state != StatementState.SUCCEEDED:
        err_msg = response.status.error.message if response.status.error else ""
        raise ValueError(f"SQL fehlgeschlagen [{state}]: {err_msg}")

    cols = [c.name for c in (response.manifest.schema.columns or [])]
    rows = response.result.data_array or []
    return pd.DataFrame(rows, columns=cols), elapsed


# =============================================================================
# Stage 4: Execution-guided selection (C4b, PET-SQL / MCS-SQL style)
# =============================================================================

def _result_fingerprint(df: pd.DataFrame) -> str:
    """Order-insensitive fingerprint of a result set for cross-consistency.

    Two candidates 'agree' when column set, row count and (sorted, capped)
    row content match - the cross-consistency criterion of PET-SQL [52].
    Rows are sorted so that missing ORDER BY clauses do not break agreement.
    """
    cols = tuple(df.columns)
    rows = sorted(str(tuple(r)) for r in df.itertuples(index=False, name=None))
    payload = repr((cols, len(df), rows[:200]))
    return hashlib.md5(payload.encode("utf-8", errors="replace")).hexdigest()


def select_by_execution(
    candidates: list[dict],
    warehouse_id: str,
    timeout_s: int,
) -> tuple[int | None, pd.DataFrame | None, float]:
    """Execute all valid candidates IN PARALLEL and majority-vote on result fingerprints.

    Candidates are submitted concurrently to the SQL Warehouse via a
    ThreadPoolExecutor (max 4 workers). Because call_sql_with_timeout
    creates a fresh WorkspaceClient per thread, there is no shared
    connection state and no pool-exhaustion risk.

    Wall-clock latency is measured around the parallel executor block
    (actual elapsed time, not the sum of individual per-thread durations).
    For n=2 candidates running in parallel this roughly halves the
    reported SQL phase compared to sequential execution.

    Mutates each candidate dict in place with exec_status / exec_error /
    n_rows. All mutations happen sequentially AFTER the pool finishes to
    avoid any write-race on the shared list.

    Tie-break: the earliest candidate (deterministic, temp 0.0 first)
    among the largest agreement group wins, mirroring MCS-SQL's [55]
    preference ordering.

    Returns:
        (winner_index, winner_df, exec_elapsed_total).
        winner_index is None when every execution failed.
    """
    # --- Identify valid candidates (sequential, order preserved) --------------
    valid_indices = []
    for idx, cand in enumerate(candidates):
        if not cand.get("valid"):
            cand["exec_status"] = "skipped (guard)"
        else:
            valid_indices.append(idx)

    if not valid_indices:
        return None, None, 0.0

    # --- Parallel execution ---------------------------------------------------
    # Each worker returns (idx, df, elapsed, exc) — no shared mutable state.
    def _exec_one(
        idx: int,
    ) -> tuple[int, pd.DataFrame | None, float, Exception | None]:
        try:
            df, elapsed = execute_sql_query(
                candidates[idx]["sql"], warehouse_id, timeout_s=timeout_s
            )
            return idx, df, elapsed, None
        except Exception as exc:  # noqa: BLE001
            return idx, None, 0.0, exc

    n_workers = min(len(valid_indices), 4)
    t_exec_wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        raw_results: list[tuple[int, pd.DataFrame | None, float, Exception | None]] = (
            list(pool.map(_exec_one, valid_indices))
        )
    # Wall-clock time for the entire parallel execution block.
    exec_total = time.perf_counter() - t_exec_wall

    # --- Collect results sequentially (no race conditions) --------------------
    successes: list[tuple[int, str, pd.DataFrame]] = []

    for idx, df, elapsed, exc in raw_results:
        cand = candidates[idx]
        if exc is not None:
            cand["exec_status"] = "error"
            cand["exec_error"]  = str(exc)
        else:
            cand["exec_status"]  = "ok"
            cand["n_rows"]       = len(df)  # type: ignore[arg-type]
            successes.append((idx, _result_fingerprint(df), df))  # type: ignore[arg-type]

    if not successes:
        return None, None, exec_total

    # --- Majority-vote on result fingerprints ---------------------------------
    # Sort successes by idx so the tie-break (earliest candidate wins) is
    # deterministic regardless of thread completion order.
    successes.sort(key=lambda t: t[0])
    counts = Counter(fp for _, fp, _ in successes)
    best_count = max(counts.values())
    for idx, fp, df in successes:  # earliest candidate in the winning group
        if counts[fp] == best_count:
            return idx, df, exec_total
    return successes[0][0], successes[0][2], exec_total  # unreachable safety net


# =============================================================================
# Stage 5: Feedback-driven repair (C4a, Self-Debugging / MAC-SQL Refiner)
# =============================================================================

def repair_sql_with_feedback(
    clean_question: str,
    schema_context: str,
    failed_sql: str,
    error_message: str,
    warehouse_id: str,
    timeout_s: int,
    model_config: dict | None = None,
) -> tuple[str | None, pd.DataFrame | None, int, float, float, dict]:
    """Iteratively repair a failing query using warehouse error feedback.

    Implements the feedback-driven regeneration loop of Self-Debugging [80]
    and MAC-SQL's Refiner agent [48], with a hard round cap to prevent the
    infinite-loop failure mode DESEM [63] warns about. Every repaired query
    passes the Layer-2 guard before touching the warehouse.

    Returns:
        (sql, df, rounds_used, llm_elapsed, exec_elapsed, tokens_agg)
        sql/df are None when all rounds failed; tokens_agg sums tiktoken
        parts (system/schema/user/output/api_*) over the repair calls.
    """
    cfg = model_config or {"endpoint": LLM_ENDPOINT, "supports_temperature": True}
    max_rounds = _MAX_REPAIR_ROUNDS
    timeout    = cfg.get("llm_timeout_s", _LLM_TIMEOUT_S)
    schema_tok = estimate_token_count(schema_context)

    tokens_agg = {
        "system": 0, "schema": 0, "user": 0, "output": 0,
        "api_prompt": 0, "api_completion": 0, "calls": 0,
    }
    llm_elapsed = 0.0
    exec_elapsed = 0.0
    current_sql, current_err = failed_sql, error_message

    system_prompt = "\n\n".join([
        "You are a Databricks SQL expert acting as a query REPAIR module. "
        "A previously generated Spark SQL query failed on the warehouse. "
        "Fix it. Output ONLY the corrected raw SQL - no fences, no explanation. "
        "Keep the query a single SELECT/WITH statement.",
        schema_context,
        _RULES_BLOCK,
    ]) + "\n" + sandwich_suffix_sql()
    system_tok = estimate_token_count(system_prompt)

    for round_no in range(1, max_rounds + 1):
        user_prompt = (
            f"Original question:\n{clean_question}\n\n"
            f"Failing SQL:\n{current_sql}\n\n"
            f"Warehouse error message:\n{current_err}\n\n"
            "Return the corrected SQL only."
        )
        try:
            raw, elapsed, p_tok, c_tok = _call_llm(
                system_prompt, user_prompt, cfg, temperature=0.0, timeout_s=timeout
            )
        except Exception as exc:
            current_err = f"Repair-LLM-Call fehlgeschlagen: {exc}"
            break

        llm_elapsed += elapsed
        tokens_agg["system"] += system_tok
        tokens_agg["schema"] += schema_tok
        tokens_agg["user"]   += estimate_token_count(user_prompt)
        tokens_agg["api_prompt"]     += p_tok
        tokens_agg["api_completion"] += c_tok
        tokens_agg["calls"] += 1

        fixed_sql = _strip_sql_fences(raw)
        tokens_agg["output"] += estimate_token_count(fixed_sql)

        is_safe, unsafe_reason = validate_sql_output(fixed_sql)
        if not is_safe:
            current_sql, current_err = fixed_sql, f"Guard-Ablehnung: {unsafe_reason}"
            continue

        try:
            df, elapsed_exec = execute_sql_query(fixed_sql, warehouse_id, timeout_s=timeout_s)
            exec_elapsed += elapsed_exec
            return fixed_sql, df, round_no, llm_elapsed, exec_elapsed, tokens_agg
        except Exception as exc:
            current_sql, current_err = fixed_sql, str(exc)

    return None, None, max_rounds, llm_elapsed, exec_elapsed, tokens_agg


# =============================================================================
# Legacy single-candidate generator (C0-style), kept for benchmark_metrics
# =============================================================================

def generate_sql_query(
    user_prompt: str,
    selected_tables: list[str],
    model_config: dict | None = None,
    warehouse_id: str | None = None,
) -> tuple[str, float, int, int, int, dict]:
    """Translate a question into ONE validated Spark SQL query (legacy path).

    This is the previous single-call pipeline step (survey category C0 with
    one similarity-selected demonstration). It is retained UNCHANGED in
    signature and return shape because benchmark runners import it directly.
    The state-of-the-art multi-stage flow lives in run_sql_pipeline().

    Returns the historical 6-tuple:
        (sql, elapsed, api_prompt_tokens, api_completion_tokens,
         system_prompt_tokens, token_breakdown)
    """
    cfg     = model_config or {"endpoint": LLM_ENDPOINT, "supports_temperature": True}
    timeout = cfg.get("llm_timeout_s", _LLM_TIMEOUT_S)

    # Layer 1: input sanitisation.
    clean_prompt, is_blocked, block_reason = sanitize_user_input(user_prompt)
    if is_blocked:
        safe_sql = f"SELECT '{block_reason}' AS hinweis"
        return (safe_sql, 0.0, 0, 0, 0, {
            "schema_context_tokens":  0, "instructions_tokens":  0,
            "user_message_tokens":    0, "total_prompt_tokens":  0,
            "output_tokens":          0, "api_prompt_tokens":    0,
            "api_completion_tokens":  0, "injection_blocked":    True,
        })

    schema_context = build_schema_context(None)  # full schema, as before
    examples_block = _render_examples_block(
        select_few_shot_examples(clean_prompt, 1)
    )
    table_list = "\n".join(f"  - {t}" for t in selected_tables)
    system_prompt = _build_generation_prompt(
        schema_context, table_list, examples_block, "direct"
    )

    schema_context_tokens = estimate_token_count(schema_context)
    system_prompt_tokens  = estimate_token_count(system_prompt)
    instructions_tokens   = system_prompt_tokens - schema_context_tokens
    user_message_tokens   = estimate_token_count(clean_prompt)

    raw, elapsed, prompt_tokens, completion_tokens = _call_llm(
        system_prompt, clean_prompt, cfg, temperature=0.0, timeout_s=timeout
    )
    sql = _strip_sql_fences(raw)

    # Layer 2: output validation.
    is_safe, unsafe_reason = validate_sql_output(sql)
    if not is_safe:
        sql = f"SELECT '{unsafe_reason}' AS hinweis"

    token_breakdown = {
        "schema_context_tokens": schema_context_tokens,
        "instructions_tokens":   instructions_tokens,
        "user_message_tokens":   user_message_tokens,
        "total_prompt_tokens":   system_prompt_tokens + user_message_tokens,
        "output_tokens":         estimate_token_count(sql),
        "api_prompt_tokens":     prompt_tokens,
        "api_completion_tokens": completion_tokens,
        "injection_blocked":     False,
    }
    return sql, elapsed, prompt_tokens, completion_tokens, system_prompt_tokens, token_breakdown


# =============================================================================
# Pipeline orchestration (headless entry point) - the SOTA flow
# =============================================================================

def run_sql_pipeline(
    question: str,
    selected_tables: list[str],
    warehouse_id: str,
    model_config: dict | None = None,
) -> dict:
    """Run the full state-of-the-art SQL pipeline (see module header).

    Headless entry point used by the Streamlit panel and benchmark runners;
    calls no Streamlit functions and is safe to run in background threads.

    Flow:
        0. sanitize_user_input                       (Layer 1)
        1. _rule_link_tables                         (C1 keyword-based, zero LLM)
        2./3. generate_sql_candidates                (C2a few-shots + C3)
        4. select_by_execution                       (C4b, optional)
        5. repair_sql_with_feedback                  (C4a, on total failure)

    Returns a flat result dict. All legacy keys are preserved; token values
    are SUMS over every LLM call of this question (see header TOKEN-METRIKEN).
    New keys: schema_linked_tables, candidates, selected_candidate_index,
    repair_rounds, n_llm_calls.
    """
    total_start = time.perf_counter()
    result: dict = {
        "question": question,
        "error": None,
        "schema_linked_tables": [],
        "candidates": [],
        "selected_candidate_index": None,
        "repair_rounds": 0,
    }

    cfg          = model_config or {}
    timeout_s    = cfg.get("sql_timeout_s", _DEFAULT_SQL_TIMEOUT_S)
    n_cands      = max(1, int(cfg.get("n_candidates", _DEFAULT_N_CANDIDATES)))
    do_selection = n_cands > 1  # voting only makes sense with multiple candidates

    def _zero_breakdown(blocked: bool) -> dict:
        return {
            "schema_context_tokens": 0, "instructions_tokens": 0,
            "user_message_tokens": 0, "total_prompt_tokens": 0,
            "output_tokens": 0, "api_prompt_tokens": 0,
            "api_completion_tokens": 0, "injection_blocked": blocked,
            "n_llm_calls": 0,
            "linking_prompt_tokens": 0, "linking_output_tokens": 0,
            "generation_prompt_tokens": 0, "generation_output_tokens": 0,
            "repair_prompt_tokens": 0, "repair_output_tokens": 0,
            "schema_context_tokens_single": 0,
        }

    try:
        # ---- Stage 0: Layer-1 guard (once, up front) ----------------------
        clean_question, is_blocked, block_reason = sanitize_user_input(question)
        if is_blocked:
            result["generated_sql"] = f"SELECT '{block_reason}' AS hinweis"
            result["results_df"]    = pd.DataFrame({"hinweis": [block_reason]})
            result["sql_gen_time"]  = 0.0
            result["sql_exec_time"] = 0.0
            tb = _zero_breakdown(blocked=True)
            result["token_breakdown"]            = tb
            result["schema_context_size_tokens"] = 0
            result["prompt_token_count"] = result["prompt_tokens"] = 0
            result["query_token_count"]  = result["completion_tokens"] = 0
            result["total_elapsed"] = time.perf_counter() - total_start
            return result

        # ---- Stage 1: rule-based schema linking (no LLM, ~0 ms) ---------------
        # Keyword rules select the minimal table subset; any ambiguous question
        # falls back to full schema (recall over precision, Distillery [58]).
        _rule_linked = _rule_link_tables(clean_question)
        linked_keys  = _rule_linked if _rule_linked else list(_TABLE_KEYS)
        linked_fq    = [TABLES[k] for k in linked_keys]
        result["schema_linked_tables"] = linked_keys
        link_elapsed = 0.0
        link_tokens  = {"system": 0, "user": 0, "output": 0,
                        "api_prompt": 0, "api_completion": 0}

        # ---- Stages 2+3: few-shots + multi-candidate generation -----------
        candidates, schema_context, gen_elapsed, gen_tokens = generate_sql_candidates(
            clean_question, linked_keys, linked_fq, cfg
        )
        result["candidates"] = candidates
        valid_candidates = [c for c in candidates if c.get("valid")]

        if not valid_candidates:
            # Every candidate was rejected by the Layer-2 guard: return the
            # historical safe stub instead of touching the warehouse.
            reason = next(
                (c.get("guard_reason") for c in candidates if c.get("guard_reason")),
                "Ausgabe durch Guard blockiert",
            )
            result["generated_sql"] = f"SELECT '{reason}' AS hinweis"
            result["results_df"]    = pd.DataFrame({"hinweis": [reason]})
            result["sql_gen_time"]  = link_elapsed + gen_elapsed
            result["sql_exec_time"] = 0.0
            df_winner = None
        else:
            # ---- Stage 4: execution-guided selection ----------------------
            if do_selection:
                winner_idx, df_winner, exec_elapsed = select_by_execution(
                    candidates, warehouse_id, timeout_s
                )
            else:
                # Cost-saving mode: execute only the deterministic candidate.
                winner_idx, df_winner, exec_elapsed = None, None, 0.0
                first_idx = candidates.index(valid_candidates[0])
                cand = candidates[first_idx]
                try:
                    df_winner, elapsed_exec = execute_sql_query(
                        cand["sql"], warehouse_id, timeout_s=timeout_s
                    )
                    exec_elapsed += elapsed_exec
                    cand["exec_status"] = "ok"
                    cand["n_rows"] = len(df_winner)
                    winner_idx = first_idx
                except Exception as exc:
                    cand["exec_status"] = "error"
                    cand["exec_error"]  = str(exc)

            repair_tokens = {"system": 0, "schema": 0, "user": 0, "output": 0,
                             "api_prompt": 0, "api_completion": 0, "calls": 0}
            repair_llm_elapsed = 0.0

            # ---- Stage 5: feedback-driven repair on total failure ----------
            if winner_idx is None:
                seed = next(
                    (c for c in candidates if c.get("exec_status") == "error"),
                    valid_candidates[0],
                )
                fixed_sql, df_fixed, rounds, r_llm, r_exec, repair_tokens = (
                    repair_sql_with_feedback(
                        clean_question, schema_context,
                        seed.get("sql", ""), seed.get("exec_error", "unbekannter Fehler"),
                        warehouse_id, timeout_s, cfg,
                    )
                )
                result["repair_rounds"] = rounds
                repair_llm_elapsed = r_llm
                exec_elapsed += r_exec
                if fixed_sql is not None:
                    result["generated_sql"] = fixed_sql
                    df_winner = df_fixed
                else:
                    # Surface the last error like the old pipeline did.
                    result["generated_sql"] = seed.get("sql", "")
                    result["error"] = seed.get(
                        "exec_error", "Alle Kandidaten und Reparaturen fehlgeschlagen."
                    )
            else:
                candidates[winner_idx]["selected"] = True
                result["selected_candidate_index"] = winner_idx
                result["generated_sql"] = candidates[winner_idx]["sql"]

                # ---- Stage 5a: empty-result repair (heuristic) -----------
                # A syntactically correct query that returns 0 rows with a
                # WHERE clause is likely semantically wrong (incorrect filter
                # value, wrong join path, bad id_type assumption). Feed a
                # semantic hint back to the LLM and attempt one repair cycle.
                # df_winner is only replaced when the repaired query returns
                # at least one row; an empty or failed repair is discarded.
                _winner_sql = candidates[winner_idx]["sql"]
                if (
                    df_winner is not None
                    and len(df_winner) == 0
                    and re.search(r"\bWHERE\b", _winner_sql, re.IGNORECASE)
                ):
                    _empty_hint = (
                        "The query executed without errors but returned 0 rows. "
                        "The WHERE clause filters or JOIN conditions are likely "
                        "incorrect for this question. Revise the query so it "
                        "returns matching data."
                    )
                    _fixed, _df_fx, _rnd, _r_llm, _r_exec, _rt = (
                        repair_sql_with_feedback(
                            clean_question, schema_context,
                            _winner_sql, _empty_hint,
                            warehouse_id, timeout_s, cfg,
                        )
                    )
                    result["repair_rounds"] = _rnd
                    repair_llm_elapsed     += _r_llm
                    exec_elapsed           += _r_exec
                    for _k, _v in _rt.items():
                        repair_tokens[_k] = repair_tokens.get(_k, 0) + _v
                    if _fixed is not None and _df_fx is not None and len(_df_fx) > 0:
                        result["generated_sql"] = _fixed
                        df_winner               = _df_fx

            result["results_df"]    = df_winner
            result["sql_gen_time"]  = link_elapsed + gen_elapsed + repair_llm_elapsed
            result["sql_exec_time"] = exec_elapsed

            gen_tokens = {
                k: gen_tokens.get(k, 0) + repair_tokens.get(k, 0)
                for k in set(gen_tokens) | set(repair_tokens)
            }
            result["_repair_tokens"] = repair_tokens  # internal, for breakdown

        # ---- Aggregated token accounting (additive over all calls) --------
        repair_tokens = result.pop("_repair_tokens", {"system": 0, "output": 0,
                                                      "user": 0, "schema": 0,
                                                      "calls": 0,
                                                      "api_prompt": 0,
                                                      "api_completion": 0})
        schema_sum = gen_tokens.get("schema", 0)
        system_sum = gen_tokens.get("system", 0) + link_tokens.get("system", 0)
        user_sum   = gen_tokens.get("user", 0)   + link_tokens.get("user", 0)
        out_sum    = gen_tokens.get("output", 0) + link_tokens.get("output", 0)
        total_prompt = system_sum + user_sum

        token_breakdown = {
            "schema_context_tokens":  schema_sum,
            "instructions_tokens":    total_prompt - schema_sum - user_sum,
            "user_message_tokens":    user_sum,
            "total_prompt_tokens":    total_prompt,
            "output_tokens":          out_sum,
            "api_prompt_tokens":      gen_tokens.get("api_prompt", 0) + link_tokens.get("api_prompt", 0),
            "api_completion_tokens":  gen_tokens.get("api_completion", 0) + link_tokens.get("api_completion", 0),
            "injection_blocked":      False,
            "n_llm_calls":            gen_tokens.get("calls", 0),
            "linking_prompt_tokens":     link_tokens.get("system", 0) + link_tokens.get("user", 0),
            "linking_output_tokens":     link_tokens.get("output", 0),
            "generation_prompt_tokens":  gen_tokens.get("system", 0) - repair_tokens.get("system", 0)
                                         + gen_tokens.get("user", 0) - repair_tokens.get("user", 0),
            "generation_output_tokens":  gen_tokens.get("output", 0) - repair_tokens.get("output", 0),
            "repair_prompt_tokens":      repair_tokens.get("system", 0) + repair_tokens.get("user", 0),
            "repair_output_tokens":      repair_tokens.get("output", 0),
            "schema_context_tokens_single": estimate_token_count(schema_context),
        }
        result["token_breakdown"]            = token_breakdown
        result["schema_context_size_tokens"] = token_breakdown["schema_context_tokens"]
        result["prompt_token_count"] = result["prompt_tokens"] = token_breakdown["total_prompt_tokens"]
        result["query_token_count"]  = result["completion_tokens"] = token_breakdown["output_tokens"]

    except Exception as e:
        result["error"] = str(e)
        result.setdefault("sql_gen_time", 0.0)
        result.setdefault("sql_exec_time", 0.0)
        result.setdefault("token_breakdown", _zero_breakdown(blocked=False))
        result.setdefault("schema_context_size_tokens", 0)
        result.setdefault("prompt_token_count", 0)
        result.setdefault("query_token_count", 0)
        result.setdefault("prompt_tokens", 0)
        result.setdefault("completion_tokens", 0)

    result["total_elapsed"] = time.perf_counter() - total_start
    return result


# =============================================================================
# Display helpers
# (Streamlit-only - never called from headless benchmark runners)
# =============================================================================

def _display_token_breakdown(res: dict) -> None:
    """Render the unified token breakdown expander for the SQL panel.

    Delegates to ui_components.render_token_breakdown (same layout as the
    SPARQL panel). Values are aggregated over ALL LLM calls of the question
    (linking + candidates + repairs); the additivity footer therefore checks
    schema + instructions + user == total over the summed values.
    """
    tb = res.get("token_breakdown")
    if not tb:
        return
    render_token_breakdown(tb)



def _display_sql_results(res: dict) -> None:
    """Render the SQL pipeline results (query, tokens, stages, data table)."""
    # Show the generated SQL FIRST (even on error/timeout).
    sql = res.get("generated_sql", "")
    if sql:
        with st.expander("Generierte SQL-Abfrage", expanded=True):
            st.code(sql, language="sql")

    _display_token_breakdown(res)

    if res.get("error"):
        st.error("Fehler bei der Verarbeitung:")
        st.code(res["error"])
        return

    df = res.get("results_df")
    if df is not None:
        if not df.empty:
            st.success(f"{len(df)} Ergebnisse gefunden.")
            st.dataframe(df, use_container_width=True, height=280)
        else:
            st.info("Die Abfrage lieferte keine Ergebnisse.")



# =============================================================================
# Panel entry point
# =============================================================================

def render_sql_panel() -> None:
    """Render the SQL panel as a Streamlit fragment child.

    Single Streamlit-facing entry point; must run inside a @st.fragment
    context so st.rerun(scope="fragment") works without re-triggering the
    ontology load or resetting page-level session state.

    UI structure:
        1. Description text
        2. Execution-time card (if a previous result exists in session state)
        3. Warehouse ID input (auto-filled from DATABRICKS_WAREHOUSE_ID)
        4. Question text input
        5. "Fragen" button - triggers pipeline, stores result, reruns fragment
        6. Result display (SQL + tokens + pipeline stages + DataFrame)
    """
    st.markdown(
        "_Stelle direkte Fragen zu den Produktionsdaten. "
        "KI generiert SQL und fuehrt es aus._"
    )

    if _SESSION_KEY in st.session_state:
        render_large_timer(
            st.session_state[_SESSION_KEY],
            overhead_ms=st.session_state.get(_OVERHEAD_KEY, 0.0),
            llm_ms=st.session_state.get(_LLM_MS_KEY, 0.0),
            exec_ms=st.session_state.get(_EXEC_MS_KEY, 0.0),
            accent_color=SQL_ACCENT_COLOR,
            bg_from=SQL_BG_FROM,
            bg_to=SQL_BG_TO,
        )

    wh_id = os.environ.get(
        "DATABRICKS_WAREHOUSE_ID",
        st.session_state.get("_sql_wh_id", ""),
    )
    if not wh_id:
        if "_sql_wh_id" not in st.session_state:
            st.session_state["_sql_wh_id"] = ""
        wh_id = st.text_input(
            "SQL Warehouse ID",
            value=st.session_state["_sql_wh_id"],
            placeholder="z.B. abc123def456",
            help="Wird automatisch aus DATABRICKS_WAREHOUSE_ID gelesen.",
            key="_wh_input",
        )
        st.session_state["_sql_wh_id"] = wh_id

    sql_question = st.text_input(
        "Frage an die Datenprodukte",
        placeholder="z.B. Welche 10 VINs haben die meisten Fehler im Bodyshop?",
        key="sql_question_input",
    )

    if st.button("Fragen", type="primary", key="sql_ask_btn", use_container_width=True):
        if not wh_id:
            st.error("DATABRICKS_WAREHOUSE_ID nicht gesetzt.")
        elif not sql_question.strip():
            st.warning("Bitte eine Frage eingeben.")
        else:
            with st.spinner("SQL-Pipeline wird ausgefuehrt..."):
                res = run_sql_pipeline(sql_question, ALL_SQL_TABLES, wh_id)

            st.session_state[_RESULTS_KEY] = res
            st.session_state[_SESSION_KEY] = res["total_elapsed"]
            _gen_ms  = res.get("sql_gen_time",  0.0) * 1000.0
            _exec_ms = res.get("sql_exec_time", 0.0) * 1000.0
            st.session_state[_LLM_MS_KEY]  = _gen_ms
            st.session_state[_EXEC_MS_KEY] = _exec_ms
            st.session_state[_OVERHEAD_KEY] = max(
                0.0, res["total_elapsed"] * 1000.0 - _gen_ms - _exec_ms
            )
            st.rerun(scope="fragment")

    if _RESULTS_KEY in st.session_state:
        _display_sql_results(st.session_state[_RESULTS_KEY])