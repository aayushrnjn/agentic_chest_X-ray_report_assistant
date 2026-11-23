"""MCP server: patient records in SQLite (read + save the signed report)."""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:                                   # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer  # noqa: E402
except ModuleNotFoundError:            # mcp 2.x renamed FastMCP -> MCPServer (same decorators)
    from mcp.server.mcpserver import MCPServer  # noqa: E402

from config import DB_PATH  # noqa: E402

mcp = MCPServer("patients")


def _query(sql: str, params: tuple = ()) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


@mcp.tool()
def get_patient_history(patient_id: str) -> dict:
    """Demographics, known conditions, smoking status, and the latest encounter
    (symptoms, vitals: SpO2, heart rate, temperature, and the clinical question)."""
    rows = _query("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    if not rows:
        return {"found": False, "patient_id": patient_id}
    p = rows[0]
    p["conditions"] = json.loads(p["conditions"])
    p["smoker"] = bool(p["smoker"])
    enc = _query("SELECT * FROM encounters WHERE patient_id = ? ORDER BY date DESC LIMIT 1", (patient_id,))
    if enc:
        enc[0]["symptoms"] = json.loads(enc[0]["symptoms"])
    return {"found": True, "patient": p, "latest_encounter": enc[0] if enc else None}


@mcp.tool()
def get_prior_scans(patient_id: str) -> dict:
    """Earlier chest X-rays for this patient with their recorded findings, newest first."""
    rows = _query("SELECT * FROM prior_scans WHERE patient_id = ? ORDER BY date DESC", (patient_id,))
    for r in rows:
        r["findings"] = json.loads(r["findings"])
    return {"patient_id": patient_id, "priors": rows}


@mcp.tool()
def save_report(patient_id: str, image_path: str, report_json: str, status: str, reviewer: str) -> dict:
    """Store a reviewed report. status is 'approved', 'edited' or 'rejected'."""
    if status not in {"approved", "edited", "rejected"}:
        return {"saved": False, "error": "status must be approved, edited or rejected"}
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.execute(
            "INSERT INTO reports (patient_id, image_path, status, reviewer, report_json) VALUES (?,?,?,?,?)",
            (patient_id, image_path, status, reviewer, report_json))
        con.commit()
        return {"saved": True, "report_id": cur.lastrowid}
    finally:
        con.close()


@mcp.resource("patient://{patient_id}/summary")
def patient_summary(patient_id: str) -> str:
    """Read-only text summary of a patient, usable as context by any MCP client."""
    h = get_patient_history(patient_id)
    if not h["found"]:
        return f"No patient {patient_id}"
    p, e = h["patient"], h["latest_encounter"] or {}
    return (f"{p['name']}, {p['age']}{p['sex']}, conditions: {', '.join(p['conditions']) or 'none'}, "
            f"smoker: {p['smoker']}. Latest: {e.get('clinical_question', '')}")


if __name__ == "__main__":
    mcp.run()
