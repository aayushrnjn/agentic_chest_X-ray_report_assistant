"""End-to-end evaluation of the agent on labelled test images.

python -m eval.evaluate_agent --n 30

The human reviewer is replaced by an auto-approver so the whole graph runs.
Measures:
- finding-level precision / recall / F1 of the final report vs ground-truth labels
- verifier: how often the FIRST draft had issues, how many were fixed by the loop
- how many cases were flagged uncertain or routed as non-diagnostic
- latency and number of draft revisions
"""
import argparse
import asyncio
import json
import time

import pandas as pd

from agent.graph import build_graph
from agent.mcp_client import load_tools
from agent.run import run_case
from config import EVAL_DIR, FINDINGS, SPLITS_DIR


def auto_approve(payload: dict) -> dict:
    return {"action": "approve", "reviewer": "auto-eval"}


async def main(n: int, seed: int):
    test = pd.read_csv(SPLITS_DIR / "test.csv").sample(n=n, random_state=seed)
    graph = build_graph(await load_tools())
    rows = []
    for _, r in test.iterrows():
        truth = {f for f in FINDINGS if r[f] == 1}
        t0 = time.time()
        res = await run_case(graph, r["path"], None, auto_approve)
        report = res.get("final_report") or {}
        reported = {f["name"] for f in report.get("findings", [])}
        history = res.get("verification_history", [])
        rows.append({
            "image": r["image"], "truth": sorted(truth), "reported": sorted(reported),
            "tp": len(truth & reported), "fp": len(reported - truth), "fn": len(truth - reported),
            "first_draft_issues": len(history[0]["issues"]) if history else 0,
            "final_issues": len(history[-1]["issues"]) if history else 0,
            "revisions": res.get("revision", 0),
            "uncertain": bool(res.get("uncertain_findings")),
            "non_diagnostic": not res.get("quality", {}).get("ok", True),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"{r['image']}: truth={sorted(truth) or ['-']} reported={sorted(reported) or ['-']} "
              f"revisions={rows[-1]['revisions']}")

    df = pd.DataFrame(rows)
    tp, fp, fn = df.tp.sum(), df.fp.sum(), df.fn.sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    summary = {
        "cases": n,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 3),
        "first_draft_failed_verification": round(float((df.first_draft_issues > 0).mean()), 3),
        "fixed_by_revision_loop": int(((df.first_draft_issues > 0) & (df.final_issues == 0)).sum()),
        "still_failing_after_loop": int((df.final_issues > 0).sum()),
        "flagged_uncertain": round(float(df.uncertain.mean()), 3),
        "non_diagnostic": int(df.non_diagnostic.sum()),
        "mean_revisions": round(float(df.revisions.mean()), 2),
        "mean_seconds": round(float(df.seconds.mean()), 1),
    }
    df.to_csv(EVAL_DIR / "agent_cases.csv", index=False)
    (EVAL_DIR / "agent_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    asyncio.run(main(a.n, a.seed))
