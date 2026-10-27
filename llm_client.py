# =============================================================================
# llm_client.py — Timeout-geschützte SDK-Aufrufe (LLM + SQL Warehouse)
#
# Zentrale Infrastruktur für alle Databricks-SDK-Aufrufe, die einen
# Wall-Clock-Timeout benötigen.  Eliminiert die Code-Duplikation zwischen
# sparql_generator.py und sql_panel.py.
#
# ─── ARCHITEKTUR ─────────────────────────────────────────────────────────────
#
#   Jeder Aufruf erzeugt einen FRISCHEN WorkspaceClient() in einem
#   Daemon-Thread.  Dadurch sind:
#     • Client-Erstellung + OAuth-Token-Refresh + HTTP-Connect alle
#       vom Timeout abgedeckt.
#     • Kein geteilter Connection-Pool-State über 65+ schnelle Requests.
#     • Thread-Tod schließt automatisch die HTTP-Verbindung.
#     • gc.collect() nach jedem Call gibt File-Descriptors sofort frei.
#
#   Vorherige Ansätze, die fehlschlugen:
#     • Singleton: Hängt bei Q25 (stale Token/Pool).
#     • Recycling alle N Calls: Hängt bei Q23 (Refresh hängt).
#     • http_timeout_seconds kwarg: Bricht OAuth-Auth in Databricks Apps.
#
# ─── DESIGN-PRINZIPIEN ───────────────────────────────────────────────────────
#
#   • Kein Streamlit-Import — sicher für Daemon-Threads und Unit-Tests.
#   • Kein LLM-Prompt-Wissen — reine Transportschicht.
#   • Typ-Annotations + ausführliche Docstrings für IDE-Unterstützung.
#   • Explizite Exceptions (TimeoutError) statt stiller Fehler.
#
# ─── EXPORTS ─────────────────────────────────────────────────────────────────
#
#   • call_llm_with_timeout(endpoint, messages, query_kwargs, timeout_s)
#       → SDK-Response-Objekt (serving_endpoints.query-Ergebnis)
#   • call_sql_with_timeout(sql, warehouse_id, wait_timeout_s, python_timeout_s)
#       → SDK-Response-Objekt (statement_execution.execute_statement-Ergebnis)
# =============================================================================

from __future__ import annotations

import gc
import threading

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import Disposition, Format


# =============================================================================
# LLM-Aufruf (Serving Endpoint)
# =============================================================================

def call_llm_with_timeout(
    endpoint: str,
    messages: list,
    query_kwargs: dict,
    timeout_s: float,
):
    """Ruft einen Databricks Serving Endpoint in einem Daemon-Thread auf.

    Erzeugt einen frischen ``WorkspaceClient`` im Worker-Thread, sodass
    Client-Erstellung, Authentifizierung und HTTP-Request alle vom
    Timeout abgedeckt sind.

    Args:
        endpoint:     Name des Databricks Model Serving Endpoints
                      (z.B. 'MO360_Data_Platform_Nexus_GPT_4o_Chat_Model_Endpoint').
        messages:     Liste von ``ChatMessage``-Objekten (System-Prompt + User-Turn).
        query_kwargs: Zusätzliche kwargs für ``serving_endpoints.query()``
                      (z.B. ``{"temperature": 0.0}``).  Wird direkt durchgereicht.
        timeout_s:    Wall-Clock-Budget in Sekunden.  Nach Ablauf wird der Thread
                      verlassen (nicht getötet — Python-Threads sind nicht
                      unterbrechbar); das Daemon-Flag verhindert Blockierung
                      beim Prozess-Shutdown.

    Returns:
        Das rohe Response-Objekt von ``serving_endpoints.query()``.

    Raises:
        TimeoutError: Worker-Thread ist nach ``timeout_s`` noch aktiv.
        Exception:    Jede im Worker-Thread geworfene Exception wird re-raised.
    """
    result_holder: list = [None]
    error_holder: list = [None]

    def _run() -> None:
        try:
            w = WorkspaceClient()
            result_holder[0] = w.serving_endpoints.query(
                name=endpoint, messages=messages, **query_kwargs
            )
        except Exception as exc:
            error_holder[0] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)

    if worker.is_alive():
        gc.collect()
        raise TimeoutError(
            f"LLM-Aufruf an '{endpoint}' überschritt das Zeitlimit von {timeout_s}s. "
            "Mögliche Ursache: Endpoint überlastet oder Rate-Limiting nach vielen "
            "aufeinanderfolgenden Anfragen."
        )
    if error_holder[0] is not None:
        raise error_holder[0]

    gc.collect()
    return result_holder[0]


# =============================================================================
# SQL-Ausführung (Statement Execution API)
# =============================================================================

def call_sql_with_timeout(
    sql: str,
    warehouse_id: str,
    wait_timeout_s: int,
    python_timeout_s: float,
):
    """Führt ein SQL-Statement über die Statement Execution API in einem Daemon-Thread aus.

    Nutzt ``Disposition.INLINE`` — Ergebnisse werden direkt im Response-Body
    zurückgegeben (kein separater Fetch nötig).  Der ``wait_timeout``-Parameter
    wird an die API weitergegeben (muss 5 ≤ n ≤ 50 erfüllen); das Python-Level
    ``python_timeout_s`` setzt eine harte Obergrenze über den API-Wait hinaus
    (deckt HTTP-Overhead ab).

    Args:
        sql:              Validiertes Spark SQL Statement zur Ausführung.
        warehouse_id:     Databricks SQL Warehouse ID.
        wait_timeout_s:   Warehouse-seitiger Wait-Timeout in Sekunden (5–50,
                          API-Constraint).  Das Warehouse gibt das Ergebnis
                          synchron zurück, wenn die Query in diesem Fenster
                          endet; sonst Status RUNNING.
        python_timeout_s: Python Thread-Join-Timeout.  Sollte
                          ``wait_timeout_s + 15`` sein für HTTP-Overhead.

    Returns:
        Das rohe Response-Objekt von ``statement_execution.execute_statement()``.

    Raises:
        TimeoutError: Worker-Thread ist nach ``python_timeout_s`` noch aktiv.
        Exception:    Jede im Worker-Thread geworfene Exception wird re-raised.
    """
    result_holder: list = [None]
    error_holder: list = [None]

    def _run() -> None:
        try:
            w = WorkspaceClient()
            result_holder[0] = w.statement_execution.execute_statement(
                statement=sql,
                warehouse_id=warehouse_id,
                disposition=Disposition.INLINE,
                format=Format.JSON_ARRAY,
                wait_timeout=f"{wait_timeout_s}s",
            )
        except Exception as exc:
            error_holder[0] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=python_timeout_s)

    if worker.is_alive():
        gc.collect()
        raise TimeoutError(
            f"SQL-Ausführung überschritt das Zeitlimit von {python_timeout_s}s. "
            f"Warehouse wait_timeout: {wait_timeout_s}s. "
            "Mögliche Ursache: komplexe Query oder Warehouse überlastet."
        )
    if error_holder[0] is not None:
        raise error_holder[0]

    gc.collect()
    return result_holder[0]
