"""Run the agent from the command line (no UI).

python -m agent.run --image path/to/xray.png --patient P001
python -m agent.run --demo 0          # uses artifacts/demo_cases.json
It pauses at human review and asks you to approve / reject in the terminal.
"""
import argparse
import asyncio
import json
import uuid

from langgraph.types import Command

from agent.graph import build_graph
from agent.mcp_client import load_tools
from config import ARTIFACTS


async def run_case(graph, image_path: str, patient_id: str | None, decide) -> dict:
    """Run one case to completion. `decide(payload) -> dict` plays the human reviewer."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = await graph.ainvoke({"image_path": image_path, "patient_id": patient_id}, config)
    while "__interrupt__" in result:                       # paused at human_review
        payload = result["__interrupt__"][0].value
        result = await graph.ainvoke(Command(resume=decide(payload)), config)
    return result


def terminal_reviewer(payload: dict) -> dict:
    print("\n========== DRAFT FOR REVIEW ==========")
    print(json.dumps(payload["draft"], indent=2))
    print("\nReview reasons:", *payload["review_reasons"] or ["none"], sep="\n - ")
    print("Triage:", (payload.get("triage") or {}).get("level"))
    answer = input("\nApprove (a) / reject (r)? ").strip().lower()
    return {"action": "approve" if answer.startswith("a") else "reject", "reviewer": "cli-user"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--patient")
    ap.add_argument("--demo", type=int, help="index into artifacts/demo_cases.json")
    args = ap.parse_args()
    if args.demo is not None:
        case = json.loads((ARTIFACTS / "demo_cases.json").read_text())[args.demo]
        args.image, args.patient = case["image_path"], case["patient_id"]
        print("ground truth labels:", case["ground_truth"] or ["No Finding"])

    graph = build_graph(await load_tools())
    result = await run_case(graph, args.image, args.patient, terminal_reviewer)
    print("\n---------- TRACE ----------")
    for t in result["trace"]:
        print(f"[{t['step']}] {t['summary']}")
    print("\nstatus:", result["status"])


if __name__ == "__main__":
    asyncio.run(main())
