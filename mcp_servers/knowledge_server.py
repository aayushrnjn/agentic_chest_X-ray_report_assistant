"""MCP server: RAG over the reference notes (tool + resources + prompt)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:                                   # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer  # noqa: E402
except ModuleNotFoundError:            # mcp 2.x renamed FastMCP -> MCPServer (same decorators)
    from mcp.server.mcpserver import MCPServer  # noqa: E402

from config import FINDINGS, KNOWLEDGE_DIR  # noqa: E402

mcp = MCPServer("knowledge")
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from rag.knowledge_base import HybridRetriever
        _retriever = HybridRetriever()
    return _retriever


@mcp.tool()
def search_guidelines(query: str, finding: str | None = None, k: int = 3) -> dict:
    """Hybrid (BM25 + vector) search with reranking over chest X-ray reference notes.
    Optionally restrict to one finding, e.g. finding='Pneumothorax'. Topics also include
    Report_Structure, Uncertainty_Language, Comparison_With_Priors, Urgent_Findings, AI_Assistance_Limits."""
    return {"query": query, "results": _get_retriever().search(query, k=k, topic=finding)}


@mcp.resource("guideline://{topic}")
def guideline(topic: str) -> str:
    """Full reference note for one topic (e.g. guideline://Effusion)."""
    for md in KNOWLEDGE_DIR.glob("*.md"):
        text = md.read_text()
        marker = f"## {topic}\n"
        if marker in text:
            return text.split(marker, 1)[1].split("\n## ", 1)[0].strip()
    return f"No note for '{topic}'. Findings: {', '.join(FINDINGS)}"


@mcp.prompt()
def chest_xray_report(clinical_question: str) -> str:
    """Reusable prompt template for drafting a chest X-ray report in any MCP client."""
    return ("Draft a structured chest X-ray report (technique, comparison, findings, impression). "
            "Only report findings supported by the model output or retrieved evidence, hedge possible "
            "findings, and answer this clinical question in the impression: " + clinical_question)


if __name__ == "__main__":
    mcp.run()
