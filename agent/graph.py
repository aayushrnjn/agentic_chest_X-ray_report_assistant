"""Wire the nodes into a LangGraph StateGraph.

    START -> intake -> analyze --(poor image)--> non_diagnostic ------------------.
                          |                                                       v
                          '--> retrieve_evidence -> compare_priors -> triage   human_review -> finalize -> END
                                                                       |          ^
                                                                       v          |
                                                         draft_report <-> verify -'
                                              (loop while issues and revision < MAX_REVISIONS)
"""
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import make_nodes, route_after_analyze, route_after_verify
from agent.state import XrayState
from config import LLM_MODEL, LLM_TEMPERATURE


def make_llm():
    # "provider:model", e.g. "ollama:llama3.1:8b" or "openai:gpt-4o-mini"; set LLM_MODEL in the environment
    return init_chat_model(LLM_MODEL, temperature=LLM_TEMPERATURE)


def build_graph(tools: dict, llm=None, checkpointer=None):
    nodes = make_nodes(tools, llm or make_llm())
    g = StateGraph(XrayState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "intake")
    g.add_edge("intake", "analyze")
    g.add_conditional_edges("analyze", route_after_analyze, ["retrieve_evidence", "non_diagnostic"])
    g.add_edge("non_diagnostic", "human_review")
    g.add_edge("retrieve_evidence", "compare_priors")
    g.add_edge("compare_priors", "triage")
    g.add_edge("triage", "draft_report")
    g.add_edge("draft_report", "verify")
    g.add_conditional_edges("verify", route_after_verify, ["draft_report", "human_review"])
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)

    # A checkpointer is required for interrupt(): it stores the paused state by thread_id.
    return g.compile(checkpointer=checkpointer or MemorySaver())
