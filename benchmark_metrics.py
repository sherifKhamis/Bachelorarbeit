# =============================================================================
# benchmark_metrics.py — Metric computation for the LLM Parameterstudie
#
# This module provides the classification and scoring logic for benchmark runs.
# It is intentionally free of Streamlit imports — pure Python logic that can
# be tested independently.
#
# Key components:
#
#   Enums:
#     • QuestionType  — unbeantwortbar | faktisch | relational | aggregativ |
#                       multi-hop | stress-test
#     • ErrorType     — none | syntax_error | unknown_predicate | unknown_column |
#                       missing_join | wrong_filter | empty_result | timeout | other
#
#   Classifiers:
#     • classify_error_type(error_str, result_df, panel)
#       Keyword-based heuristic mapping error messages to ErrorType values.
#       Panel-aware: "unknown_predicate" only for ONT, "unknown_column" for SQL.
#
#     • classify_question_type(question_text, generated_query)
#       Priority chain: unbeantwortbar (keywords) > multi-hop (join>=2 or path>=3) >
#       aggregativ (COUNT/SUM/AVG) > relational (JOINs or path>=2) > faktisch.
#       Used ONLY for user-created questions (is_fixture=False).
#       Uses _UNANSWERABLE_KEYWORDS regex for unanswerable detection.
#
#     • question_type_from_id(question_id)
#       Maps a fixture Q-number to its corpus-defined QuestionType using the
#       Q-number ranges from corpus.py. Used when is_fixture=True.
#
#   Complexity extraction:
#     • extract_join_depth(sql_query)     — counts JOIN clauses in SQL
#     • extract_triple_path_depth(sparql) — counts distinct triple patterns
#
#   Record builder:
#     • build_run_record(run_id, question_id, question_text, ont_result,
#                        sql_result, schema_context_ont, schema_context_sql,
#                        model_name, is_fixture)
#       Assembles a flat dict with all metrics for one (question x model) pair.
#       When is_fixture=True, question_type is derived from the Q-number range
#       (no auto-classification). join_depth and triple_path_depth are always
#       computed from the generated queries.
#       Includes: timing, tokens, error classification, complexity, query text.
# =============================================================================

from __future__ import annotations

import re
from collections import defaultdict
from enum import Enum
from typing import Optional

import pandas as pd


class QuestionType(str, Enum):
    UNBEANTWORTBAR = "unbeantwortbar"
    FAKTISCH       = "faktisch"
    RELATIONAL     = "relational"
    AGGREGATIV     = "aggregativ"
    MULTI_HOP      = "multi-hop"
    STRESS_TEST    = "stress-test"


class ErrorType(str, Enum):
    NONE              = "none"
    SYNTAX_ERROR      = "syntax_error"
    UNKNOWN_PREDICATE = "unknown_predicate"
    UNKNOWN_COLUMN    = "unknown_column"
    MISSING_JOIN      = "missing_join"
    WRONG_FILTER      = "wrong_filter"
    EMPTY_RESULT      = "empty_result"
    TIMEOUT           = "timeout"
    OTHER             = "other"


def classify_error_type(
    error_str: Optional[str],
    result_df=None,
    panel: str = "ont",
) -> str:
    """Classify an error message into one of the ErrorType categories."""
    if not error_str:
        if isinstance(result_df, pd.DataFrame) and len(result_df) == 0:
            return ErrorType.EMPTY_RESULT.value
        return ErrorType.NONE.value

    err = error_str.lower()

    if any(k in err for k in (
        "timeout", "time limit", "zeitlimit", "uberschritt",
        "timed out", "abgebrochen",
    )):
        return ErrorType.TIMEOUT.value

    if panel == "ont" and any(k in err for k in (
        "not defined", "unknown prefix", "undefined", "prefix not found",
        "undeclared prefix", "not found in",
    )):
        return ErrorType.UNKNOWN_PREDICATE.value

    if panel == "sql" and any(k in err for k in (
        "unresolved_column", "unresolved_routine", "unresolved column",
        "cannot resolve", "no such column", "no such struct field",
        "column_not_found", "table_or_view_not_found",
        "no viable alternative",
    )):
        return ErrorType.UNKNOWN_COLUMN.value

    if any(k in err for k in (
        "syntax", "parse", "unexpected token", "unexpected character",
        "sparql parse", "parse_syntax_error",
        "analysisexception", "analysis_exception",
        "invalid_identifier", "invalid identifier",
        "mismatched input", "extraneous input",
        "invalid query", "invalid sparql", "not a valid",
        "expected ", "unexpected end", "unclosed",
        "malformed", "illegal", "invalid expression",
        "unrecognized", "cannot be resolved",
        "blocked query form", "blocked sql operation",
        "blocked keyword", "unknown sql form",
        "only select", "cte without final select",
        "sparql update form",
    )):
        return ErrorType.SYNTAX_ERROR.value

    if any(k in err for k in (
        "join", "cartesian", "ambiguous reference",
        "detected cartesian product",
    )):
        return ErrorType.MISSING_JOIN.value

    if any(k in err for k in (
        "no results", "0 rows", "empty result",
    )):
        return ErrorType.WRONG_FILTER.value

    if panel == "sql" and "fehlgeschlagen" in err:
        return ErrorType.SYNTAX_ERROR.value

    if panel == "ont" and any(k in err for k in (
        "oxigraph", "rdflib", "query error",
    )):
        return ErrorType.SYNTAX_ERROR.value

    return ErrorType.OTHER.value


# =============================================================================
# Question type auto-classification
# =============================================================================

_UNANSWERABLE_KEYWORDS = re.compile(
    r"\b("
    r"preis|kosten|kostet|stuckpreis|herstellkosten|budget|umsatz|"
    r"gewinn|marge|rabatt|erlös|deckungsbeitrag|"
    r"mitarbeiter|personal|belegschaft|angestellte|schicht|schichtplan|"
    r"schulung|zertifikat|qualifikation|vorgesetzter|teamleiter|"
    r"farbe|farbcode|lackiert|lackierung|oberflächengute|glanzgrad|"
    r"zulieferer|lieferant|supplier|bauteil|gewicht|material|werkstoff|"
    r"lieferung|transport|versand|logistik|bestellung|auftrag|"
    r"maschine|roboter|werkzeug|anlage|maschinennummer|seriennummer|"
    r"wartung|instandhaltung|stillstand|reparatur|ersatzteil|"
    r"temperatur|luftfeuchtigkeit|strom|energie|verbrauch|"
    r"emission|co2|abfall|nachhaltigkeit|"
    r"kalender|zeitplan|geplant|termin|taktzeit|zykluszeit|durchlaufzeit|"
    r"kunde|kundin|verkauf|handler|garantie|reklamation|beschwerde|"
    r"price|cost|employee|staff|shift|color|colour|paint\s*color|"
    r"supplier|vendor|schedule|planned|customer|warranty|claim|"
    r"weight|material|machine|robot|tool|maintenance|downtime|"
    r"energy|consumption|delivery|shipment|order"
    r")\b",
    re.IGNORECASE,
)

_AGGREGATION_KEYWORDS = re.compile(
    r"\b(wie\s*viele|anzahl|durchschnitt|summe|gesamt|maximum|minimum|"
    r"haufigste|meisten|wenigsten|anteil|prozent|rate|"
    r"hochste|hochsten|niedrigste|niedrigsten|"
    r"quote|fehlerquote|verteilung|statistik|ranking|"
    r"haufig\w*|selten\w*|top\s*\d+|"
    r"count|sum|average|avg|total|max|min|most|least|ratio|percentage|"
    r"distribution|frequency|highest|lowest|statistics)\b",
    re.IGNORECASE,
)

_AGGREGATION_SQL = re.compile(
    r"\b(COUNT|SUM|AVG|MAX|MIN|GROUP\s+BY|HAVING)\b",
    re.IGNORECASE,
)

_AGGREGATION_SPARQL = re.compile(
    r"\b(COUNT|SUM|AVG|MAX|MIN|GROUP\s+BY|HAVING)\b",
    re.IGNORECASE,
)

_RELATIONAL_KEYWORDS = re.compile(
    r"\b(welche|zeige|liste|nenne|gib\s*mir|"
    r"hat|haben|gehort|zugeordnet|verbunden|verknupft|"
    r"show|list|give|which|display|belongs|related|associated)\b",
    re.IGNORECASE,
)

_FAKTISCH_KEYWORDS = re.compile(
    r"\b(was\s+ist|wer\s+ist|wo\s+ist|wann|"
    r"what\s+is|who\s+is|where\s+is|when)\b",
    re.IGNORECASE,
)


def classify_question_type(
    question_text: str,
    sql_query: str = "",
    sparql_query: str = "",
    join_depth: int = 0,
    triple_path_depth: int = 0,
) -> str:
    """Auto-classify a question into one of the QuestionType categories.

    Used ONLY for user-created questions (is_fixture=False). Fixture questions
    use question_type_from_id() instead.

    Priority: unbeantwortbar > multi-hop > aggregativ > relational > faktisch
    """
    q = question_text.lower()

    if bool(_UNANSWERABLE_KEYWORDS.search(question_text)):
        return QuestionType.UNBEANTWORTBAR.value

    is_multi_hop = (
        join_depth >= 2
        or triple_path_depth >= 3
        or bool(re.search(r"\b(uber\s+alle|across|through|via|und\s+deren|und\s+dessen)\b", q))
    )
    if is_multi_hop:
        return QuestionType.MULTI_HOP.value

    has_agg_keywords = bool(_AGGREGATION_KEYWORDS.search(question_text))
    has_agg_sql      = bool(_AGGREGATION_SQL.search(sql_query))    if sql_query    else False
    has_agg_sparql   = bool(_AGGREGATION_SPARQL.search(sparql_query)) if sparql_query else False

    if has_agg_keywords or has_agg_sql or has_agg_sparql:
        return QuestionType.AGGREGATIV.value

    has_relational = bool(_RELATIONAL_KEYWORDS.search(question_text))
    has_join       = join_depth >= 1 or triple_path_depth >= 2

    if has_relational and has_join:
        return QuestionType.RELATIONAL.value
    if has_join:
        return QuestionType.RELATIONAL.value

    return QuestionType.FAKTISCH.value


def question_type_from_id(question_id: str) -> str:
    """Return the corpus-defined QuestionType for a fixture question.

    Maps Q-number ranges as defined in corpus.py:
        Q001-Q005  -> unbeantwortbar
        Q006-Q025  -> faktisch
        Q026-Q045  -> relational
        Q046-Q065  -> aggregativ
        Q066-Q085  -> multi-hop
        Q086+      -> faktisch (fallback; user-created should not reach here)
    """
    try:
        n = int(question_id[1:])
    except (ValueError, IndexError):
        return QuestionType.FAKTISCH.value

    if   n <=  5: return QuestionType.UNBEANTWORTBAR.value
    elif n <= 25: return QuestionType.FAKTISCH.value
    elif n <= 45: return QuestionType.RELATIONAL.value
    elif n <= 65: return QuestionType.AGGREGATIV.value
    elif n <= 85: return QuestionType.MULTI_HOP.value
    return QuestionType.FAKTISCH.value


# =============================================================================
# Complexity extraction
# =============================================================================

def _expand_sparql_abbreviations(sparql_query: str) -> str:
    """Expand SPARQL semicolon (;) and comma (,) abbreviations to full triples."""
    _TERM = r"(?:\?[\w]+|<[^>]+>|[\w]+:[\w\d_-]+)"
    _PRED = r"(?:[\w:?<>\[\]|/^*+-]+)"

    lines = sparql_query.split("\n")
    expanded_lines: list[str] = []
    current_subject: str = ""
    current_predicate: str = ""

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("FILTER", "BIND", "OPTIONAL", "UNION",
                                                "SELECT", "CONSTRUCT", "WHERE", "PREFIX",
                                                "{", "}", "#", "LIMIT", "ORDER", "GROUP",
                                                "HAVING", "VALUES")):
            expanded_lines.append(line)
            continue

        subj_match = re.match(rf"^\s*({_TERM})\s+", stripped)
        if subj_match:
            current_subject = subj_match.group(1)

        expanded_lines.append(line)
        po_matches = list(re.finditer(rf"({_PRED})\s+({_TERM})", stripped))
        if po_matches:
            current_predicate = po_matches[0].group(1) if subj_match else current_predicate

    result = sparql_query
    semi_pattern = re.compile(
        rf"({_TERM})(\s+{_PRED}\s+{_TERM}\s*(?:,\s*{_TERM}\s*)*);",
        re.MULTILINE,
    )
    max_iter = 50
    for _ in range(max_iter):
        m = semi_pattern.search(result)
        if not m:
            break
        subject = m.group(1)
        result = result[:m.end() - 1] + " . " + subject + result[m.end():]

    return result


def extract_triple_path_depth(sparql_query: str) -> int:
    """Return the length of the longest connected variable-chain in a SPARQL query."""
    if not sparql_query:
        return 0

    expanded = _expand_sparql_abbreviations(sparql_query)

    triple_re = re.compile(
        r"(\?[\w]+|<[^>]+>|[\w]+:[\w\d_]+)"
        r"\s+"
        r"(?:[\w:?<>\[\]|/^*+]+)"
        r"\s+"
        r"(\?[\w]+|<[^>]+>|[\w]+:[\w\d_]+)"
        r"\s*[.;}]",
        re.IGNORECASE,
    )
    edges: list[tuple[str, str]] = []
    for m in triple_re.finditer(expanded):
        s, o = m.group(1), m.group(2)
        if s.startswith("?") and o.startswith("?") and s != o:
            edges.append((s, o))

    if not edges:
        return len(re.findall(r"\?\w+\s+\S+\s+\S+\s*[.;}]", expanded))

    graph: dict[str, set[str]] = defaultdict(set)
    all_nodes: set[str] = set()
    for s, o in edges:
        graph[s].add(o)
        all_nodes.update([s, o])

    def _dfs(node: str, visited: set[str]) -> int:
        return max(
            (_dfs(nb, visited | {nb}) + 1 for nb in graph[node] if nb not in visited),
            default=0,
        )

    return max(_dfs(v, {v}) for v in all_nodes)


def _branch_depth(sql_fragment: str) -> int:
    """Compute join depth for a single SQL branch (no UNION splitting)."""
    q = sql_fragment.strip()
    if not q:
        return 0

    depth = 0

    depth += len(re.findall(
        r"\b(?:INNER\s+|LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|"
        r"FULL\s+(?:OUTER\s+)?|CROSS\s+|LEFT\s+SEMI\s+|LEFT\s+ANTI\s+)?JOIN\b",
        q, re.IGNORECASE,
    ))

    depth += len(re.findall(r"\bLATERAL\s+VIEW\b", q, re.IGNORECASE))

    from_matches = re.finditer(
        r"\bFROM\s+(.+?)(?:\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|"
        r"\bLIMIT\b|\bHAVING\b|\bLATERAL\b|\bJOIN\b|$)",
        q, re.IGNORECASE,
    )
    for m in from_matches:
        from_clause = m.group(1)
        paren_depth_local = 0
        for ch in from_clause:
            if ch == "(":
                paren_depth_local += 1
            elif ch == ")":
                paren_depth_local -= 1
            elif ch == "," and paren_depth_local == 0:
                depth += 1

    depth += len(re.findall(
        r"\b(?:EXISTS|IN)\s*\(\s*SELECT\b", q, re.IGNORECASE,
    ))

    return depth


def extract_join_depth(sql_query: str) -> int:
    """Compute the effective join depth of a Spark SQL / Databricks SQL query."""
    if not sql_query:
        return 0

    q = re.sub(r"--[^\n]*", " ", sql_query)
    q = re.sub(r"/\*.*?\*/", " ", q, flags=re.DOTALL)
    q = re.sub(r"\s+", " ", q).strip()

    cte_depths: dict[str, int] = {}
    main_query = q

    if re.match(r"\s*WITH\b", q, re.IGNORECASE):
        pos = q.upper().find("WITH") + 4
        while pos < len(q):
            while pos < len(q) and q[pos] == " ":
                pos += 1
            name_match = re.match(r"(\w+)\s+AS\s*\(", q[pos:], re.IGNORECASE)
            if not name_match:
                break
            cte_name = name_match.group(1).lower()
            body_start = pos + name_match.end()
            paren_count = 1
            i = body_start
            while i < len(q) and paren_count > 0:
                if q[i] == "(":
                    paren_count += 1
                elif q[i] == ")":
                    paren_count -= 1
                i += 1
            cte_body = q[body_start:i - 1]
            cte_depths[cte_name] = _branch_depth(cte_body)
            pos = i
            while pos < len(q) and q[pos] in " ,":
                pos += 1

        main_query = q[pos:]

    union_count = len(re.findall(r"\bUNION\b(?:\s+ALL)?\b", main_query, re.IGNORECASE))

    branches = re.split(r"\bUNION\b(?:\s+ALL)?\b", main_query, flags=re.IGNORECASE)
    max_branch = 0
    for branch in branches:
        bd = _branch_depth(branch)
        max_cte = 0
        for cte_name, cte_depth in cte_depths.items():
            if re.search(r"\b" + re.escape(cte_name) + r"\b", branch, re.IGNORECASE):
                max_cte = max(max_cte, cte_depth)
        bd += max_cte
        max_branch = max(max_branch, bd)

    return max_branch + union_count


def approx_tokens(text: str) -> int:
    """Approximate GPT-4o token count for a text string (~4 chars per token)."""
    return max(1, len(text) // 4)


def _df_preview(df) -> list[dict] | None:
    """Return a JSON-serializable representation of a DataFrame (all rows)."""
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return None
    return df.fillna("").astype(str).to_dict(orient="records")


def _build_genie_metrics(genie_results: Optional[dict]) -> dict:
    """Build flat metrics dict from Genie Space results."""
    if not genie_results:
        return {}

    metrics: dict[str, dict] = {}
    for label, res in genie_results.items():
        wall_ms = round(res.get("total_elapsed", 0.0) * 1000.0, 1)
        error = res.get("error")
        status = res.get("status", "UNKNOWN")
        metrics[label] = {
            "wall_ms": wall_ms,
            "status": status,
            "error": error,
            "exec_success": error is None and status == "COMPLETED",
            "generated_sql": res.get("generated_sql"),
            "row_count": res.get("row_count"),
            "result_preview":    res.get("result_preview"),
            "latency_breakdown": res.get("latency_breakdown"),
        }
    return metrics


def build_run_record(
    run_id: str,
    question_id: str,
    question_text: str,
    ont_result: dict,
    sql_result: dict,
    schema_context_ont: str,
    schema_context_sql: str,
    model_name: str = "GPT-4o",
    repetition: int = 1,
    genie_results: Optional[dict] = None,
    is_fixture: bool = False,
) -> dict:
    """Assemble a flat metrics dict from pipeline outputs.

    Args:
        is_fixture: When True, question_type is derived from the Q-number range
                    defined in corpus.py — no auto-classification is performed.
                    join_depth and triple_path_depth are always computed from the
                    generated queries regardless of this flag.
                    When False (user-created questions), question_type is
                    auto-classified from question text + generated query structure.
    """
    _ont_results_df = ont_result.get("results_df")
    _ont_triples_df = ont_result.get("triples_df")
    ont_df = _ont_results_df if _ont_results_df is not None else _ont_triples_df

    sql_df    = sql_result.get("results_df")
    ont_err   = ont_result.get("error")
    sql_err   = sql_result.get("error")
    ont_query = ont_result.get("generated_query", "")
    sql_query = sql_result.get("generated_sql",   "")

    # Always compute structural complexity from the generated queries
    jd = extract_join_depth(sql_query)
    tp = extract_triple_path_depth(ont_query)

    # Question type: fixture -> derive from Q-number range (no auto-compute)
    #                user-created -> auto-classify from text + query structure
    if is_fixture:
        q_type = question_type_from_id(question_id)
    else:
        q_type = classify_question_type(
            question_text=question_text,
            sql_query=sql_query,
            sparql_query=ont_query,
            join_depth=jd,
            triple_path_depth=tp,
        )

    ont_tb = ont_result.get("token_breakdown") or {}
    sql_tb = sql_result.get("token_breakdown") or {}

    ont_prompt_tc       = ont_tb.get("total_prompt_tokens") or ont_result.get("prompt_token_count") or ont_result.get("prompt_tokens", 0)
    ont_query_tc        = ont_tb.get("output_tokens") or ont_result.get("query_token_count") or ont_result.get("completion_tokens", 0)
    ont_schema_tc       = ont_tb.get("schema_context_tokens") or ont_result.get("schema_context_size_tokens") or approx_tokens(schema_context_ont)
    ont_instructions_tc = ont_tb.get("instructions_tokens", 0)
    ont_user_msg_tc     = ont_tb.get("user_message_tokens", 0)

    graph_rag_applied       = ont_tb.get("graph_rag_applied", False)
    graph_rag_reduction_pct = ont_tb.get("graph_rag_reduction_pct", 0.0)

    sql_graph_rag_applied       = sql_tb.get("graph_rag_applied", False)
    sql_graph_rag_reduction_pct = sql_tb.get("graph_rag_reduction_pct", 0.0)

    sql_prompt_tc       = sql_tb.get("total_prompt_tokens") or sql_result.get("prompt_token_count") or sql_result.get("prompt_tokens", 0)
    sql_query_tc        = sql_tb.get("output_tokens") or sql_result.get("query_token_count") or sql_result.get("completion_tokens", 0)
    sql_schema_tc       = sql_tb.get("schema_context_tokens") or sql_result.get("schema_context_size_tokens") or approx_tokens(schema_context_sql)
    sql_instructions_tc = sql_tb.get("instructions_tokens", 0)
    sql_user_msg_tc     = sql_tb.get("user_message_tokens", 0)

    _ont_llm_ms      = round(ont_result.get("sparql_gen_time",  0.0) * 1000, 1)
    _ont_exec_ms     = round(ont_result.get("sparql_exec_time", 0.0) * 1000, 1)
    _ont_viz_ms      = round(ont_result.get("viz_time",         0.0) * 1000, 1)
    _ont_total_ms    = round(ont_result.get("total_elapsed",    0.0) * 1000, 1)
    _ont_overhead_ms = round(max(0.0, _ont_total_ms - _ont_llm_ms - _ont_exec_ms - _ont_viz_ms), 1)

    # Graph RAG sub-phase (embedding + traversal) extracted from token_breakdown
    _ont_rag_meta = ont_tb.get("graph_rag_meta", {})
    _ont_rag_ms   = round(
        _ont_rag_meta.get("embedding_ms", 0.0) + _ont_rag_meta.get("traversal_ms", 0.0), 1
    )

    _sql_llm_ms      = round(sql_result.get("sql_gen_time",  0.0) * 1000, 1)
    _sql_exec_ms     = round(sql_result.get("sql_exec_time", 0.0) * 1000, 1)
    _sql_total_ms    = round(sql_result.get("total_elapsed", 0.0) * 1000, 1)
    _sql_overhead_ms = round(max(0.0, _sql_total_ms - _sql_llm_ms - _sql_exec_ms), 1)

    # Graph RAG sub-phase (embedding + live SQL retrieval) extracted from token_breakdown
    _sql_rag_meta = sql_tb.get("graph_rag_meta", {})
    _sql_rag_ms   = round(
        _sql_rag_meta.get("embedding_ms", 0.0) + _sql_rag_meta.get("retrieval_ms", 0.0), 1
    )

    return {
        "run_id":                       run_id,
        "model_name":                   model_name,
        "question_id":                  question_id,
        "question":                     question_text,
        "repetition":                   repetition,

        # Complexity — question_type from corpus for fixtures, auto-detected for user questions
        "question_type":                q_type,
        "join_depth":                   jd,
        "triple_path_depth":            tp,

        "ont_llm_ms":      _ont_llm_ms,
        "ont_exec_ms":     _ont_exec_ms,
        "ont_viz_ms":      _ont_viz_ms,
        "ont_total_ms":    _ont_total_ms,
        "ont_overhead_ms": _ont_overhead_ms,
        "ont_rag_ms":      _ont_rag_ms,       # Graph RAG sub-phase (embedding + traversal)

        "sql_llm_ms":      _sql_llm_ms,
        "sql_exec_ms":     _sql_exec_ms,
        "sql_total_ms":    _sql_total_ms,
        "sql_overhead_ms": _sql_overhead_ms,
        "sql_rag_ms":      _sql_rag_ms,       # Graph RAG sub-phase (embedding + live retrieval)

        "ont_prompt_token_count":           ont_prompt_tc,
        "ont_query_token_count":            ont_query_tc,
        "ont_schema_context_size_tokens":   ont_schema_tc,
        "ont_instructions_tokens":          ont_instructions_tc,
        "ont_user_message_tokens":          ont_user_msg_tc,
        "ont_graph_rag_applied":            graph_rag_applied,
        "ont_graph_rag_reduction_pct":      graph_rag_reduction_pct,

        "sql_prompt_token_count":           sql_prompt_tc,
        "sql_query_token_count":            sql_query_tc,
        "sql_schema_context_size_tokens":   sql_schema_tc,
        "sql_instructions_tokens":          sql_instructions_tc,
        "sql_user_message_tokens":          sql_user_msg_tc,
        "sql_graph_rag_applied":            sql_graph_rag_applied,
        "sql_graph_rag_reduction_pct":      sql_graph_rag_reduction_pct,

        # Legacy keys (backward compat)
        "ont_prompt_tokens":            ont_prompt_tc,
        "ont_query_tokens":             ont_query_tc,
        "sql_prompt_tokens":            sql_prompt_tc,
        "sql_query_tokens":             sql_query_tc,
        "schema_ont_approx_tokens":     ont_schema_tc,
        "schema_sql_approx_tokens":     sql_schema_tc,

        "ont_error_type":               (_ont_et := classify_error_type(ont_err, ont_df, panel="ont")),
        "sql_error_type":               (_sql_et := classify_error_type(sql_err, sql_df, panel="sql")),
        "ont_error_msg":                ont_err or "",
        "sql_error_msg":                sql_err or "",

        "ont_syntax_valid":             _ont_et not in (ErrorType.SYNTAX_ERROR.value, ErrorType.UNKNOWN_PREDICATE.value),
        "sql_syntax_valid":             _sql_et not in (ErrorType.SYNTAX_ERROR.value, ErrorType.UNKNOWN_COLUMN.value),
        "ont_exec_success":             ont_err is None,
        "sql_exec_success":             sql_err is None,
        "ont_row_count":                len(ont_df) if isinstance(ont_df, pd.DataFrame) else 0,
        "sql_row_count":                len(sql_df) if isinstance(sql_df, pd.DataFrame) else 0,

        "ont_generated_query":          ont_query,
        "sql_generated_query":          sql_query,

        "ont_result_preview":           _df_preview(ont_df),
        "sql_result_preview":           _df_preview(sql_df),

        "ont_first_attempt_success":    ont_err is None,
        "sql_first_attempt_success":    sql_err is None,

        "human_evaluation":             None,
        "ont_human_score":              None,
        "sql_human_score":              None,

        "genie_results":                _build_genie_metrics(genie_results),
    }
