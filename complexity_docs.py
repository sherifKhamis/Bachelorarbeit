# =============================================================================
# complexity_docs.py — Komplexitätsdokumentation für den Fragen-Corpus
#
# Enthält die interaktive HTML-Dokumentation der Komplexitätsdimensionen:
#   • Fragetyp-Übersicht (5 Stufen mit Beschreibung)
#   • Unbeantwortbare Fragen — erwartetes LLM-Verhalten
#   • SQL join_depth Beispiele (Depth 0–3 mit Hover-Tooltips)
#   • SPARQL triple_path_depth Beispiele (Depth 1–4 mit Hover-Tooltips)
#   • Abgedeckte Ontologie-Entitäten & SQL-Tabellen
#   • Fragetyp-zu-Frage Mapping (welche Frage testet was)
#
# Separiert aus corpus.py zur besseren Wartbarkeit — reine Render-Funktionen
# ohne Business-Logik oder State-Management.
#
# Exported:
#   • render_complexity_docs()  — Rendert den vollständigen Expander
# =============================================================================

from __future__ import annotations

import streamlit as st


def render_complexity_docs() -> None:
    """Render the full complexity documentation expander.

    Contains styled HTML with hover tooltips, code examples, and tables
    explaining all complexity dimensions of the 85-question corpus.
    """
    with st.expander("📖 Komplexitätsdimensionen der Fragen", expanded=False):
        _render_overview()
        _render_unanswerable_section()
        _render_question_mapping()
        _render_sql_depth_examples()
        _render_sparql_depth_examples()
        _render_entity_coverage()


# =============================================================================
# Internal rendering functions
# =============================================================================

def _render_overview() -> None:
    """Render the top-level complexity overview table."""
    st.markdown("""
Die 85 Default-Fragen sind nach **zunehmender Komplexität** geordnet und decken
alle Entitäten und Relationen der Ontologie bzw. der SQL-Datenbank ab.

#### Fragetypen

| Typ | Bereich | Fragen | Beschreibung | SQL join_depth | SPARQL triple_path_depth |
|-----|---------|--------|--------------|----------------|--------------------------|
| **Unbeantwortbar** | Q001–Q005 | 5 | Daten existieren NICHT im Schema — testet Halluzination vs. Ablehnung | n/a | n/a |
| **Faktisch** | Q006–Q025 | 20 | Direkte Eigenschaft eines Objekts (single property lookup) | 0 | 1 |
| **Relational** | Q026–Q045 | 20 | Verbindung zwischen zwei Entitäten (1-hop navigation) | 1 | 1–2 |
| **Aggregativ** | Q046–Q065 | 20 | Zählen, Summen, Durchschnitte über Mengen | 0–2 | 1–2 |
| **Multi-Hop** | Q066–Q085 | 20 | Ketten mehrerer Relationen (cross-entity, cross-area) | 2–3 | 3–4 |

""")


def _render_unanswerable_section() -> None:
    """Render documentation for unanswerable questions (Q001–Q005)."""
    st.markdown("""
#### Unbeantwortbare Fragen (Q001–Q005)

Diese Fragen klingen plausibel im Produktionskontext, aber die benötigten Daten
sind **weder in der Ontologie noch in der SQL-Datenbank** vorhanden:

| Frage | Fehlende Daten | Warum plausibel? |
|-------|---------------|------------------|
| Q001 — Stückpreis | Kein Preis-/Kostenfeld | Fahrzeuge haben VINs, aber keine Finanzdaten |
| Q002 — Mitarbeiter | Keine Personaldaten | Fehler existieren, aber ohne Zuordnung zu Personen |
| Q003 — Farbe | Keine Farbspezifikation | Lackiererei existiert als Bereich, aber speichert keine Farben |
| Q004 — Zulieferer | Keine Lieferantendaten | Bauteile und Hallen existieren, aber ohne Supply-Chain |
| Q005 — Schichtbeginn | Kein Kalender/Zeitplan | Produktionsbereiche existieren, aber ohne Zeitdimension |

**Mögliche LLM-Ergebnisse:**

| Verhalten | Error-Typ | Bewertung |
|-----------|-----------|-----------|
| LLM verweigert (Text statt Query) | `syntax_error` | ✓ Korrekt — erkennt fehlende Daten |
| LLM generiert Query mit falschem Feld | `unknown_predicate` / `unknown_column` | ⚠ Teilweise korrekt |
| LLM liefert leeres Ergebnis | `empty_result` | ⚠ Akzeptabel |
| LLM halluziniert "funktionierende" Query | `none` (erscheint als Erfolg) | ✗ Halluzination |

Im letzten Fall (Halluzination) ist **manuelle Bewertung** im Qualitäts-Tab nötig,
da der Benchmark den semantischen Fehler nicht automatisch erkennen kann.
""")


def _render_question_mapping() -> None:
    """Render the question-to-tested-concept mapping."""
    st.markdown("""
#### Getestete Konzepte pro Fragenblock

| Block | Ontologie-Pfade | SQL-Pattern | Kernentitäten |
|-------|----------------|-------------|---------------|
| **Faktisch** (Q006–Q025) | `?vin :belongsToModelType ?type` | `SELECT ... WHERE vin = X` | VIN, ModelType, Checkpoint, Hall, Area, Mark |
| **Relational** (Q026–Q045) | `?vin :hasFault ?f . ?f :isCausedInHall ?h` | `LATERAL VIEW EXPLODE(faults_worksteps)` | VIN→Fault, Fault→Hall, Checkpoint→Category, FaultVehicleLocation |
| **Aggregativ** (Q046–Q065) | `GROUP BY` + `COUNT`/`AVG` über Triple-Patterns | `COUNT(DISTINCT vin)`, `GROUP BY hall` | Alle Entitäten mit Aggregation |
| **Multi-Hop** (Q066–Q085) | 3–4 verkettete Triples + FILTER | CTE + UNION ALL + Multi-EXPLODE | Cross-Area, Fault-Chain (s→c→r) |
""")


def _render_sql_depth_examples() -> None:
    """Render interactive SQL join_depth examples with hover tooltips."""
    st.markdown("""
#### SQL-Komplexität: `join_depth`

Zählt die JOIN-Stufen der korrekten SQL-Antwort. Hover über die SQL-Befehle für Erklärungen.
""")
    st.markdown("""
<style>
.sql-kw {
    background: #1e293b; color: #93c5fd; border-radius: 4px;
    padding: 1px 5px; font-family: 'Courier New', monospace;
    font-size: 12px; font-weight: 600; cursor: help;
    border-bottom: 1px dashed #60a5fa;
}
.sql-example {
    background: #0f172a; border: 1px solid #334155; border-radius: 8px;
    padding: 12px 16px; margin: 8px 0 16px 0; font-family: 'Courier New', monospace;
    font-size: 12px; line-height: 1.6; color: #e2e8f0; overflow-x: auto;
}
.sql-comment { color: #64748b; font-style: italic; }
.depth-badge {
    display: inline-block; background: #7e22ce; color: #fff;
    border-radius: 6px; padding: 2px 8px; font-size: 11px;
    font-weight: 700; margin-right: 8px;
}
</style>

<p><span class='depth-badge'>Depth 0</span> Eine Tabelle, kein JOIN — einfaches Filtern (Q006–Q025)</p>
<div class='sql-example'>
<span class='sql-comment'>-- Welchen Fahrzeugtyp hat VIN X?</span><br>
<abbr class='sql-kw' title='SELECT: Wählt Spalten aus der Ergebnismenge aus.'>SELECT</abbr> model_type<br>
<abbr class='sql-kw' title='FROM: Gibt die Quelltabelle an.'>FROM</abbr> assembly_faults<br>
<abbr class='sql-kw' title='WHERE: Filtert Zeilen nach einer Bedingung.'>WHERE</abbr> vin = <span style='color:#fbbf24;'>\\'W1KLF4HB0TA278667\\'</span>
</div>

<p><span class='depth-badge'>Depth 1</span> LATERAL VIEW EXPLODE — verschachteltes Array aufklappen (Q026–Q045)</p>
<div class='sql-example'>
<span class='sql-comment'>-- Welche Fehler hat VIN X? (Array-Explosion)</span><br>
<abbr class='sql-kw' title='SELECT: Wählt Spalten aus Haupttabelle und explodiertem Array.'>SELECT</abbr> vin, fw.fault_type, fw.s_hall<br>
<abbr class='sql-kw' title='FROM: Quelltabelle mit verschachteltem faults_worksteps Array.'>FROM</abbr> assembly_faults<br>
<abbr class='sql-kw' title='LATERAL VIEW: Erzeugt für jedes Array-Element eine eigene Zeile.'>LATERAL VIEW</abbr> <abbr class='sql-kw' title='EXPLODE: Entpackt ein Array in einzelne Zeilen.'>EXPLODE</abbr>(faults_worksteps) fw_table <abbr class='sql-kw' title='AS: Alias für das explodierte Element.'>AS</abbr> fw<br>
<abbr class='sql-kw' title='WHERE: Filtert nach der Explosion.'>WHERE</abbr> vin = <span style='color:#fbbf24;'>\\'W1KLF4HB0TA278667\\'</span>
</div>

<p><span class='depth-badge'>Depth 2</span> UNION ALL + EXPLODE — mehrere Tabellen kombinieren (Q046–Q065)</p>
<div class='sql-example'>
<span class='sql-comment'>-- Alle Fehler über alle Bereiche hinweg</span><br>
<abbr class='sql-kw' title='SELECT: Einheitliche Spalten aus jeder Teiltabelle.'>SELECT</abbr> vin, fw.fault_type, fw.s_hall, <span style='color:#fbbf24;'>\\'Assembly\\'</span> <abbr class='sql-kw' title='AS: Fester Bereichsname als Spalte.'>AS</abbr> area<br>
<abbr class='sql-kw' title='FROM: Erste Quelltabelle (Montage).'>FROM</abbr> assembly_faults<br>
<abbr class='sql-kw' title='LATERAL VIEW: Entpackt das Array.'>LATERAL VIEW</abbr> <abbr class='sql-kw' title='EXPLODE: Array → Einzelzeilen.'>EXPLODE</abbr>(faults_worksteps) t <abbr class='sql-kw' title='AS: Alias.'>AS</abbr> fw<br>
<br>
<abbr class='sql-kw' title='UNION ALL: Vereinigt Ergebnisse mehrerer SELECTs. Duplikate werden NICHT entfernt.'>UNION ALL</abbr><br>
<br>
<abbr class='sql-kw' title='SELECT: Gleiche Spaltenstruktur für Vereinigung.'>SELECT</abbr> vin, fw.fault_type, fw.s_hall, <span style='color:#fbbf24;'>\\'Paintshop\\'</span> AS area<br>
<abbr class='sql-kw' title='FROM: Zweite Quelltabelle (Lackiererei).'>FROM</abbr> paintshop_faults<br>
<abbr class='sql-kw' title='LATERAL VIEW: Array-Explosion für zweite Tabelle.'>LATERAL VIEW</abbr> <abbr class='sql-kw' title='EXPLODE: Entpackt das Array.'>EXPLODE</abbr>(faults_worksteps) t AS fw
</div>

<p><span class='depth-badge'>Depth 3</span> CTE + UNION ALL + mehrere LATERAL VIEWs + Aggregation (Q066–Q085)</p>
<div class='sql-example'>
<span class='sql-comment'>-- Fehlerverteilung pro Halle über alle Bereiche</span><br>
<abbr class='sql-kw' title='WITH: Common Table Expression (CTE) — temporäre benannte Zwischentabelle.'>WITH</abbr> all_faults <abbr class='sql-kw' title='AS: Benennt die CTE.'>AS</abbr> (<br>
&nbsp;&nbsp;<abbr class='sql-kw' title='SELECT: Extrahiert relevante Felder.'>SELECT</abbr> fw.s_hall, fw.fault_type<br>
&nbsp;&nbsp;<abbr class='sql-kw' title='FROM: Erste Tabelle.'>FROM</abbr> assembly_faults<br>
&nbsp;&nbsp;<abbr class='sql-kw' title='LATERAL VIEW: Array entpacken.'>LATERAL VIEW</abbr> <abbr class='sql-kw' title='EXPLODE: Array → Zeilen.'>EXPLODE</abbr>(faults_worksteps) t AS fw<br>
&nbsp;&nbsp;<abbr class='sql-kw' title='UNION ALL: Weitere Tabellen anhängen.'>UNION ALL</abbr><br>
&nbsp;&nbsp;SELECT fw.s_hall, fw.fault_type FROM paintshop_faults<br>
&nbsp;&nbsp;LATERAL VIEW EXPLODE(faults_worksteps) t AS fw<br>
&nbsp;&nbsp;UNION ALL<br>
&nbsp;&nbsp;SELECT fw.s_hall, fw.fault_type FROM bodyshop_faults<br>
&nbsp;&nbsp;LATERAL VIEW EXPLODE(faults_worksteps) t AS fw<br>
)<br>
<abbr class='sql-kw' title='SELECT: Hauptabfrage auf die CTE.'>SELECT</abbr> s_hall, <abbr class='sql-kw' title='COUNT(*): Zählt Zeilen pro Gruppe.'>COUNT(*)</abbr> AS fault_count<br>
<abbr class='sql-kw' title='FROM: Referenziert die CTE.'>FROM</abbr> all_faults<br>
<abbr class='sql-kw' title='GROUP BY: Gruppiert nach Halle.'>GROUP BY</abbr> s_hall<br>
<abbr class='sql-kw' title='ORDER BY: Sortiert absteigend nach Fehleranzahl.'>ORDER BY</abbr> fault_count <abbr class='sql-kw' title='DESC: Absteigende Sortierung.'>DESC</abbr>
</div>

<p style='margin-top:12px;color:#dc2626;font-size:12px;font-weight:600;'>
⚠ Ab join_depth ≥ 2 beginnen LLMs häufig, falsche Join-Keys zu halluzinieren.
</p>
""", unsafe_allow_html=True)


def _render_sparql_depth_examples() -> None:
    """Render interactive SPARQL triple_path_depth examples with hover tooltips."""
    st.markdown("""

#### SPARQL-Komplexität: `triple_path_depth`

Längster zusammenhängender Property-Pfad im generierten SPARQL. Hover über die Elemente für Erklärungen.
""", unsafe_allow_html=True)
    st.markdown("""
<style>
.sparql-kw {
    background: #1a2332; color: #7dd3fc; border-radius: 4px;
    padding: 1px 5px; font-family: 'Courier New', monospace;
    font-size: 12px; font-weight: 600; cursor: help;
    border-bottom: 1px dashed #38bdf8;
}
.sparql-var {
    color: #a5f3fc; font-family: 'Courier New', monospace;
    font-size: 12px; cursor: help;
    border-bottom: 1px dotted #67e8f9;
}
.sparql-prop {
    color: #86efac; font-family: 'Courier New', monospace;
    font-size: 12px; font-weight: 600; cursor: help;
    border-bottom: 1px dashed #4ade80;
}
.sparql-example {
    background: #0f172a; border: 1px solid #334155; border-radius: 8px;
    padding: 12px 16px; margin: 8px 0 16px 0; font-family: 'Courier New', monospace;
    font-size: 12px; line-height: 1.8; color: #e2e8f0; overflow-x: auto;
}
.sparql-comment { color: #64748b; font-style: italic; }
.depth-badge-sparql {
    display: inline-block; background: #0369a1; color: #fff;
    border-radius: 6px; padding: 2px 8px; font-size: 11px;
    font-weight: 700; margin-right: 8px;
}
</style>

<p><span class='depth-badge-sparql'>Depth 1</span> Ein Triple — direkter Zugriff auf eine Relation (Q006–Q025)</p>
<div class='sparql-example'>
<span class='sparql-comment'># Welche Fehler hat ein Fahrzeug?</span><br>
<abbr class='sparql-var' title='?fahrzeug: Variable — repräsentiert ein Fahrzeug (VIN).'>?fahrzeug</abbr> <abbr class='sparql-prop' title=':hasFault — Object Property: Verbindet Fahrzeug mit Fehlern.'>:hasFault</abbr> <abbr class='sparql-var' title='?fehler: Variable — wird an jede Fault-Instanz gebunden.'>?fehler</abbr> <span style='color:#64748b;'>.</span>
</div>

<p><span class='depth-badge-sparql'>Depth 2</span> Zwei verkettete Triples — 1-Hop Navigation (Q026–Q045)</p>
<div class='sparql-example'>
<span class='sparql-comment'># In welcher Halle wurde der Fehler verursacht?</span><br>
<abbr class='sparql-var' title='?fahrzeug: Startpunkt der Kette.'>?fahrzeug</abbr> <abbr class='sparql-prop' title=':hasFault — Erste Kante: Fahrzeug → Fehler.'>:hasFault</abbr> <abbr class='sparql-var' title='?fehler: Zwischenknoten — wird im nächsten Triple weiterverwendet.'>?fehler</abbr> <span style='color:#64748b;'>.</span><br>
<abbr class='sparql-var' title='?fehler: Dieselbe Variable — impliziter Join über gemeinsamen Variablennamen.'>?fehler</abbr> &nbsp;&nbsp;<abbr class='sparql-prop' title=':isCausedInHall — Zweite Kante: Fehler → Halle.'>:isCausedInHall</abbr> <abbr class='sparql-var' title='?halle: Endpunkt — die Verursacher-Halle.'>?halle</abbr> <span style='color:#64748b;'>.</span>
</div>

<p><span class='depth-badge-sparql'>Depth 3</span> Drei verkettete Triples — 2-Hop Navigation (Q046–Q065)</p>
<div class='sparql-example'>
<span class='sparql-comment'># In welcher Halle liegt der Checkpoint an dem der Fehler entstand?</span><br>
<abbr class='sparql-var' title='?fahrzeug: Startknoten.'>?fahrzeug</abbr> <abbr class='sparql-prop' title=':hasFault — Fahrzeug → Fehler.'>:hasFault</abbr> <abbr class='sparql-var' title='?fehler: Zwischenknoten 1.'>?fehler</abbr> <span style='color:#64748b;'>.</span><br>
<abbr class='sparql-var' title='?fehler: Impliziter Join.'>?fehler</abbr> &nbsp;&nbsp;<abbr class='sparql-prop' title=':isCausedAtCheckpoint — Fehler → Checkpoint.'>:isCausedAtCheckpoint</abbr> <abbr class='sparql-var' title='?cp: Zwischenknoten 2.'>?cp</abbr> <span style='color:#64748b;'>.</span><br>
<abbr class='sparql-var' title='?cp: Checkpoint-Variable.'>?cp</abbr> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<abbr class='sparql-prop' title=':isInHall — Checkpoint → Halle.'>:isInHall</abbr> <abbr class='sparql-var' title='?halle: Endpunkt — die Halle des Checkpoints.'>?halle</abbr> <span style='color:#64748b;'>.</span>
</div>

<p><span class='depth-badge-sparql'>Depth 4</span> Vier verkettete Triples — 3-Hop Navigation (Q066–Q085)</p>
<div class='sparql-example'>
<span class='sparql-comment'># In welchem Werk liegt der Checkpoint den das Fahrzeug durchlaufen hat?</span><br>
<abbr class='sparql-var' title='?fahrzeug: Startknoten.'>?fahrzeug</abbr> <abbr class='sparql-prop' title=':passedThroughCheckpoint — Fahrzeug → Checkpoint.'>:passedThroughCheckpoint</abbr> <abbr class='sparql-var' title='?cp: Der durchlaufene Checkpoint.'>?cp</abbr> <span style='color:#64748b;'>.</span><br>
<abbr class='sparql-var' title='?cp: Navigation weiter.'>?cp</abbr> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<abbr class='sparql-prop' title=':isInHall — Checkpoint → Halle.'>:isInHall</abbr> <abbr class='sparql-var' title='?halle: Die Halle.'>?halle</abbr> <span style='color:#64748b;'>.</span><br>
<abbr class='sparql-var' title='?halle: Weiter zum Bereich.'>?halle</abbr> &nbsp;&nbsp;&nbsp;<abbr class='sparql-prop' title=':isInArea — Halle → Area (Produktionsbereich).'>:isInArea</abbr> <abbr class='sparql-var' title='?area: Der Produktionsbereich.'>?area</abbr> <span style='color:#64748b;'>.</span><br>
<abbr class='sparql-var' title='?area: Letzte Navigation zum Werk.'>?area</abbr> &nbsp;&nbsp;&nbsp;&nbsp;<abbr class='sparql-prop' title=':isInPlant — Area → Plant (Werk).'>:isInPlant</abbr> <abbr class='sparql-var' title='?plant: Endpunkt — das Produktionswerk.'>?plant</abbr> <span style='color:#64748b;'>.</span>
</div>

<p style='margin-top:12px;font-size:12px;color:#475569;'>
💡 <strong>Impliziter Join:</strong> In SPARQL werden Triples über gemeinsame Variablennamen verbunden —
kein explizites JOIN-Keyword nötig. Je mehr Triples verkettet sind, desto tiefer der Pfad.
</p>
""", unsafe_allow_html=True)


def _render_entity_coverage() -> None:
    """Render the ontology entities and SQL tables coverage section."""
    st.markdown("""
#### Abgedeckte Ontologie-Entitäten

| Entität | Rolle | Wichtige Properties |
|---------|-------|-------------------|
| **VIN** | Fahrzeug | `hasFault`, `passedThroughCheckpoint`, `belongsToModelType`, `vinInHall/Area/Plant` |
| **Fault** | Fehler | `hasType`, `isCaused*`, `isDetected*`, `isReworked*`, `occursAtVehicleLocation` |
| **Checkpoint** | Prüfpunkt | `hasCategory`, `hasMark`, `isInHall`, `locationProgress` |
| **Hall** | Produktionshalle | `isInArea`, `categoriesInHall`, `marksInHall` |
| **Area** | Bereich | `isInPlant` (Assembly, Paintshop, Bodyshop) |
| **Plant** | Werk | Wurzel der räumlichen Hierarchie |
| **ModelType** | Fahrzeugtyp | `hasAffectedVIN`, `isProducedIn*` |
| **FaultType** | Fehlertyp | Klassifizierung von Fehlern |
| **Mark** | Prüfmarke | Identifier eines Checkpoints |
| **Category** | Checkpoint-Kategorie | z.B. QUALITY, PRODUCTION |
| **FaultVehicleLocation** | Fehlerposition | Position am Fahrzeug |

#### SQL-Tabellen

Drei identische Tabellen (eine pro Produktionsbereich) mit verschachteltem
`faults_worksteps` Array (STRUCT mit `s_hall`, `c_hall`, `r_hall` für
Verursachung/Erkennung/Nacharbeit):

| Tabelle | Bereich | Schlüsselspalten |
|---------|---------|-----------------|
| `assembly` | Montage | `vin`, `model_type`, `faults_worksteps` (Array<Struct>) |
| `paintshop` | Lackiererei | `vin`, `model_type`, `faults_worksteps` (Array<Struct>) |
| `bodyshop` | Karosseriebau | `vin`, `model_type`, `faults_worksteps` (Array<Struct>) |

**Struct-Felder in `faults_worksteps`:** `fault_type`, `s_hall` (verursacht),
`s_mark`, `s_category`, `c_hall` (erkannt), `c_mark`, `c_category`,
`r_hall` (nachgearbeitet), `r_mark`, `r_category`, `fault_vehicle_location`
""")
