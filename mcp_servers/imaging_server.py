"""MCP server: the deep learning model as tools.

Run standalone (stdio):  python mcp_servers/imaging_server.py
Every tool returns a dict (one JSON blob), small enough for an LLM context:
heatmaps are returned as file paths, never as image bytes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:                                   # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer  # noqa: E402
except ModuleNotFoundError:            # mcp 2.x renamed FastMCP -> MCPServer (same decorators)
    from mcp.server.mcpserver import MCPServer  # noqa: E402

from config import FINDINGS  # noqa: E402
from models.preprocess import check_quality  # noqa: E402

mcp = MCPServer("imaging")
_cases = None


def _model():
    from models.inference import get_model    # lazy: quality checks work without loading torch
    return get_model()


@mcp.tool()
def check_image_quality(image_path: str) -> dict:
    """Check whether a chest X-ray is good enough to interpret (blur, contrast, exposure, resolution)."""
    return check_quality(image_path)


@mcp.tool()
def classify_xray(image_path: str) -> dict:
    """Run the CNN on a chest X-ray. Returns calibrated probability, decision threshold,
    positive flag and uncertain flag for each of the 14 findings, most likely first."""
    return _model().predict(image_path)


@mcp.tool()
def get_heatmap(image_path: str, finding: str) -> dict:
    """Grad-CAM heatmap showing which image regions drove the score for one finding.
    Returns the path to the saved overlay PNG."""
    if finding not in FINDINGS:
        return {"error": f"unknown finding '{finding}'. Valid: {FINDINGS}"}
    return {"finding": finding, "heatmap_path": _model().gradcam(image_path, finding)}


@mcp.tool()
def find_similar_cases(image_path: str, k: int = 3) -> dict:
    """Retrieve the k most visually similar past cases (with their real radiology reports)."""
    global _cases
    from rag.case_index import CaseIndex
    if _cases is None:
        _cases = CaseIndex()
    hits = _cases.search(_model().embed(image_path), k=k, exclude_path=image_path)
    return {"cases": hits}


if __name__ == "__main__":
    mcp.run()
