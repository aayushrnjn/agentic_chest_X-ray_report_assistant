"""MCP server: LightGBM urgency estimate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:                                   # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer  # noqa: E402
except ModuleNotFoundError:            # mcp 2.x renamed FastMCP -> MCPServer (same decorators)
    from mcp.server.mcpserver import MCPServer  # noqa: E402

mcp = MCPServer("triage")
_model = None


@mcp.tool()
def estimate_urgency(probabilities: dict[str, float], age: int = 50, spo2: float = 97.0,
                     heart_rate: float = 80.0, temperature_c: float = 37.0,
                     symptoms: list[str] | None = None) -> dict:
    """Urgency level (routine / priority / urgent) from CNN finding probabilities plus vitals
    and symptoms (fever, dyspnea, chest_pain). Returns the level, class probabilities and
    the features that contributed most."""
    global _model
    if _model is None:
        from ml.triage import TriageModel
        _model = TriageModel()
    symptoms = set(symptoms or [])
    vitals = {"age": age, "spo2": spo2, "heart_rate": heart_rate, "temperature_c": temperature_c,
              "fever": int("fever" in symptoms or temperature_c > 38.0),
              "dyspnea": int("dyspnea" in symptoms), "chest_pain": int("chest_pain" in symptoms)}
    return _model.predict(probabilities, vitals)


if __name__ == "__main__":
    mcp.run()
