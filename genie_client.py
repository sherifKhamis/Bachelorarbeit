# =============================================================================
# genie_client.py — Genie Space API client for benchmark integration
#
# Provides a timeout-protected function to query Databricks Genie Spaces
# via the REST API (start-conversation → poll until COMPLETED/FAILED).
#
# Architecture:
#   Uses a fresh WorkspaceClient() per call in a daemon thread (same pattern
#   as llm_client.py) to avoid connection-pool exhaustion during batch runs.
#   Auth is handled via w.api_client.do() which manages OAuth automatically.
#
# Return dict (compatible with benchmark_metrics build_run_record):
#   {
#     "question":       str,
#     "total_elapsed":  float (seconds),
#     "status":         "COMPLETED" | "FAILED" | "TIMEOUT" | ...,
#     "error":          str | None,
#     "generated_sql":  str | None,
#     "description":    str | None,
#     "row_count":      int | None,
#   }
#
# Exports:
#   • run_genie_space(question, space_id, timeout_s=120) → dict
# =============================================================================

from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Optional

_log = logging.getLogger(__name__)


def run_genie_space(
    question: str,
    space_id: str,
    timeout_s: float = 120,
) -> dict:
    """Query a Genie Space and return timing + quality metrics.

    Starts a new conversation, polls until completion or timeout,
    and extracts the generated SQL + result metadata.

    Args:
        question:   Natural language question to send.
        space_id:   Genie Space ID (UUID hex string).
        timeout_s:  Wall-clock timeout in seconds.

    Returns:
        Dict with keys: question, total_elapsed, status, error,
        generated_sql, description, row_count.
    """
    result_holder: list[Optional[dict]] = [None]

    def _run() -> None:
        t0 = time.perf_counter()
        try:
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient()
            # Use SDK api_client for authenticated HTTP (handles OAuth in Apps)
            api = w.api_client
            t_init = time.perf_counter()  # Phase 1: SDK init done

            # ── Step 1: Start conversation ────────────────────────────────
            data = api.do(
                "POST",
                f"/api/2.0/genie/spaces/{space_id}/start-conversation",
                body={"content": question},
            )

            conversation_id = data["conversation"]["id"]
            message_id = data["message"]["id"]
            t_start = time.perf_counter()  # Phase 2: start-conversation POST done

            # ── Step 2: Poll until terminal state ─────────────────────────
            poll_path = (
                f"/api/2.0/genie/spaces/{space_id}"
                f"/conversations/{conversation_id}/messages/{message_id}"
            )

            terminal_states = {"COMPLETED", "FAILED", "CANCELLED"}
            status = data["message"].get("status", "IN_PROGRESS")
            message_data = data["message"]

            while status not in terminal_states:
                if (time.perf_counter() - t0) > (timeout_s - 2):
                    # About to exceed timeout, bail out
                    status = "TIMEOUT"
                    break
                time.sleep(1.5)
                message_data = api.do("GET", poll_path)
                status = message_data.get("status", "UNKNOWN")
            t_poll = time.perf_counter()  # Phase 3: polling done

            # ── Step 3: Extract results ───────────────────────────────────
            error_msg = None
            generated_sql = None
            description = None
            row_count = None

            result_preview = None  # list[dict] — first 20 rows as records

            if status == "TIMEOUT":
                error_msg = f"Genie Space Timeout ({timeout_s}s überschritten)"
            elif status == "FAILED":
                err_obj = message_data.get("error")
                error_msg = str(err_obj) if err_obj else "Genie Space Fehler (unbekannt)"
            elif status == "COMPLETED":
                attachments = message_data.get("attachments") or []
                query_attachment_id = None

                for att_idx, att in enumerate(attachments):
                    if att.get("type") == "QUERY" or "query" in att:
                        query_obj = att.get("query") or att
                        generated_sql = query_obj.get("query") or query_obj.get("content", "")
                        description = query_obj.get("description", "")
                        # The Genie API uses 'attachment_id', NOT 'id'.
                        query_attachment_id = att.get("attachment_id") or att.get("id") or None
                        # Check inline query_result (inside query obj)
                        qr = query_obj.get("query_result")
                        if isinstance(qr, dict):
                            row_count = row_count or qr.get("row_count")
                            # Try to extract inline data (some Genie versions embed it here)
                            if not result_preview:
                                _cols = qr.get("columns") or []
                                _rows = qr.get("data_array") or []
                                if _cols and _rows:
                                    _names = [
                                        c.get("name") or c.get("column_name") or f"col_{i}"
                                        for i, c in enumerate(_cols)
                                    ]
                                    result_preview = [
                                        dict(zip(_names, r)) for r in _rows
                                    ]
                    # Check query_result at attachment level
                    if "query_result" in att:
                        qr = att["query_result"]
                        if isinstance(qr, dict):
                            row_count = row_count or qr.get("row_count")

                # Fetch result data via Genie query-result API
                # Correct path: /attachments/{attachment_id}/query-result
                if query_attachment_id and generated_sql:
                    try:
                        qr_path = (
                            f"/api/2.0/genie/spaces/{space_id}"
                            f"/conversations/{conversation_id}"
                            f"/messages/{message_id}"
                            f"/attachments/{query_attachment_id}/query-result"
                        )
                        qr_data = api.do("GET", qr_path)
                        if isinstance(qr_data, dict):
                            # Extract columns and rows from statement result
                            stmt = qr_data.get("statement_response") or qr_data
                            manifest = stmt.get("manifest") or {}
                            result_obj = stmt.get("result") or qr_data.get("result") or {}
                            columns = manifest.get("schema", {}).get("columns") or manifest.get("columns") or qr_data.get("columns") or []
                            data_array = result_obj.get("data_array") or result_obj.get("data_typed_array") or qr_data.get("data_array") or []
                            row_count = row_count or manifest.get("total_row_count") or qr_data.get("row_count")
                            if columns and data_array:
                                col_names = [
                                    c.get("name") or c.get("column_name") or f"col_{i}"
                                    for i, c in enumerate(columns)
                                ]
                                result_preview = [
                                    dict(zip(col_names, row))
                                    for row in data_array
                                ]
                    except Exception:
                        # Silently drop preview — don't contaminate result
                        pass

                # Fallback: check top-level attachments structure
                if not generated_sql and attachments:
                    for att in attachments:
                        content = att.get("text", {}).get("content", "")
                        if content and not description:
                            description = content

            t_fetch  = time.perf_counter()  # Phase 4: result fetch done
            elapsed  = t_fetch - t0
            latency_breakdown = {
                "init_s":    t_init  - t0,
                "start_s":   t_start - t_init,
                "polling_s": t_poll  - t_start,
                "fetch_s":   t_fetch - t_poll,
            }

            # Mark as FAILED if COMPLETED but no result table returned
            if status == "COMPLETED" and not result_preview:
                status = "FAILED"
                error_msg = "Keine Ergebnistabelle zurückgegeben"

            result_holder[0] = {
                "question":          question,
                "total_elapsed":     elapsed,
                "status":            status,
                "error":             error_msg,
                "generated_sql":     generated_sql,
                "description":       description,
                "row_count":         row_count,
                "result_preview":    result_preview,
                "latency_breakdown": latency_breakdown,
            }

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result_holder[0] = {
                "question": question,
                "total_elapsed": elapsed,
                "status": "ERROR",
                "error": str(exc),
                "generated_sql": None,
                "description": None,
                "row_count": None,
            }
        finally:
            gc.collect()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s + 5)  # small buffer beyond internal timeout

    if worker.is_alive() or result_holder[0] is None:
        return {
            "question": question,
            "total_elapsed": timeout_s,
            "status": "TIMEOUT",
            "error": f"Thread-Timeout ({timeout_s}s überschritten)",
            "generated_sql": None,
            "description": None,
            "row_count": None,
        }

    return result_holder[0]
