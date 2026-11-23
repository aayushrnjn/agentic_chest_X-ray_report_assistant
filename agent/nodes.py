"""The LangGraph nodes. Each node = one step of the workflow.

Tool calls go through MCP (`call(tools, ...)`), so the agent never imports
torch, LightGBM or SQLite directly; it only knows the tool contracts.
The LLM is used in exactly one place, `draft_report`, for language. All
decisions with clinical weight (routing, verification) are explicit code.
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from agent.mcp_client import call
from agent.schemas import RadiologyReport
from agent.verify import compare_with_prior, verify_report
from config import MAX_REVISIONS

MAX_HEATMAPS = 4

SYSTEM_PROMPT = """You draft chest X-ray reports for a radiologist to review. Rules:
1. Report ONLY findings listed under CANDIDATE FINDINGS. Never add others.
2. status='present' only for findings marked confident; findings marked uncertain are 'possible'.
3. Copy probability and change_vs_prior exactly from the data given.
4. evidence_ids must come from the EVIDENCE list (always include MODEL:<name> for each finding).
5. Pertinent negatives: findings relevant to the clinical question that are NOT candidates.
6. The impression answers the clinical question, most important finding first, using hedged
   language for possible findings. Do not name a finding in the impression unless it is reported,
   resolved, or listed as a pertinent negative.
7. Mention image-quality limitations and that this is an AI-assisted draft requiring review."""


def _step(name: str, summary: str) -> list[dict]:
    return [{"step": name, "summary": summary}]


def _candidates(predictions: list[dict]) -> list[dict]:
    return [p for p in predictions if p["positive"] or p["uncertain"]]


def make_nodes(tools: dict, llm):
    structured_llm = llm.with_structured_output(RadiologyReport)

    async def intake(state):
        quality = await call(tools, "check_image_quality", image_path=state["image_path"])
        patient, priors = None, []
        if state.get("patient_id"):
            hist = await call(tools, "get_patient_history", patient_id=state["patient_id"])
            patient = hist if hist.get("found") else None
            priors = (await call(tools, "get_prior_scans", patient_id=state["patient_id"])).get("priors", [])
        msg = f"quality {'ok' if quality['ok'] else 'POOR: ' + '; '.join(quality['problems'])}; " \
              f"patient {'loaded' if patient else 'not provided'}; {len(priors)} prior scan(s)"
        return {"quality": quality, "patient": patient, "priors": priors, "revision": 0,
                "review_reasons": [], "trace": _step("intake", msg)}

    async def analyze(state):
        if not state["quality"]["ok"]:
            return {"predictions": [], "heatmaps": {}, "uncertain_findings": [],
                    "trace": _step("analyze", "skipped: image not interpretable")}
        result = await call(tools, "classify_xray", image_path=state["image_path"])
        preds = result["findings"]
        heatmaps = {}
        for p in _candidates(preds)[:MAX_HEATMAPS]:
            hm = await call(tools, "get_heatmap", image_path=state["image_path"], finding=p["finding"])
            heatmaps[p["finding"]] = hm.get("heatmap_path")
        uncertain = [p["finding"] for p in preds if p["uncertain"]]
        reasons = list(state.get("review_reasons", []))
        if uncertain:
            reasons.append(f"Model is uncertain about: {', '.join(uncertain)}")
        if not result.get("calibrated"):
            reasons.append("Model probabilities are not calibrated (calibration.json missing)")
        pos = [f"{p['finding']} {p['probability']:.2f}" for p in preds if p["positive"]]
        return {"predictions": preds, "heatmaps": heatmaps, "uncertain_findings": uncertain,
                "review_reasons": reasons,
                "trace": _step("analyze", f"positive: {', '.join(pos) or 'none'}; uncertain: {', '.join(uncertain) or 'none'}")}

    async def non_diagnostic(state):
        problems = "; ".join(state["quality"]["problems"])
        draft = RadiologyReport(
            technique=f"Image quality inadequate: {problems}.",
            comparison="Not assessed.", findings=[], impression=(
                "Non-diagnostic study: image quality does not allow reliable interpretation. "
                "AI-assisted draft; requires radiologist review."),
            recommendation="Repeat the radiograph with correct technique.").model_dump()
        return {"draft": draft, "issues": [],
                "review_reasons": [f"Non-diagnostic image: {problems}"],
                "trace": _step("non_diagnostic", "image rejected by quality check, routed to human review")}

    async def retrieve_evidence(state):
        cands = _candidates(state["predictions"])
        cases = (await call(tools, "find_similar_cases", image_path=state["image_path"], k=3)).get("cases", [])

        queries = [(f"{p['finding']} appearance on chest x-ray and follow-up", p["finding"]) for p in cands[:4]]
        enc = (state.get("patient") or {}).get("latest_encounter") or {}
        if enc.get("clinical_question"):
            queries.append((enc["clinical_question"], None))
        if state.get("uncertain_findings"):
            queries.append(("how to word uncertain findings in a report", "Uncertainty_Language"))
        if state.get("priors"):
            queries.append(("comparison with prior imaging", "Comparison_With_Priors"))

        seen, guidelines = set(), []
        for q, topic in queries:
            res = await call(tools, "search_guidelines", query=q, finding=topic, k=2)
            for r in res.get("results", []):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    guidelines.append(r)
        return {"similar_cases": cases, "guidelines": guidelines,
                "trace": _step("retrieve_evidence", f"{len(cases)} similar case(s), {len(guidelines)} guideline chunk(s)")}

    async def compare_priors(state):
        comp = compare_with_prior(state["predictions"], state.get("priors", []))
        new = [f for f, v in comp.items() if v == "new"]
        msg = "no prior available" if comp["prior_date"] is None else \
            f"vs {comp['prior_date']}: new {new or 'none'}, resolved {comp['resolved'] or 'none'}"
        return {"prior_comparison": comp, "trace": _step("compare_priors", msg)}

    async def triage(state):
        p = state.get("patient") or {}
        demo, enc = p.get("patient") or {}, p.get("latest_encounter") or {}
        args = {"probabilities": {x["finding"]: x["probability"] for x in state["predictions"]},
                "symptoms": enc.get("symptoms") or []}
        for key, src in [("age", demo.get("age")), ("spo2", enc.get("spo2")),
                         ("heart_rate", enc.get("heart_rate")), ("temperature_c", enc.get("temperature_c"))]:
            if src is not None:
                args[key] = src
        result = await call(tools, "estimate_urgency", **args)
        reasons = list(state.get("review_reasons", []))
        if result.get("level") == "urgent":
            reasons.append("Triage model estimates URGENT: notify the referring clinician")
        return {"triage": result, "review_reasons": reasons,
                "trace": _step("triage", f"{result.get('level')} {result.get('probabilities')}")}

    def _evidence(state) -> tuple[list[dict], set[str]]:
        items = [{"id": f"MODEL:{p['finding']}", "text": f"CNN probability {p['probability']} (threshold {p['threshold']})"}
                 for p in _candidates(state["predictions"])]
        items += [{"id": g["id"], "text": g["text"][:400]} for g in state.get("guidelines", [])]
        items += [{"id": c["case_id"], "text": f"similarity {c['similarity']}; impression: {c['impression'][:250]}"}
                  for c in state.get("similar_cases", [])]
        return items, {i["id"] for i in items}

    async def draft_report(state):
        evidence, _ = _evidence(state)
        comp = state["prior_comparison"]
        candidates = [{"name": p["finding"], "probability": p["probability"],
                       "confidence": "uncertain" if p["uncertain"] else "confident",
                       "change_vs_prior": comp.get(p["finding"], "no_prior")}
                      for p in _candidates(state["predictions"])]
        not_seen = [p["finding"] for p in state["predictions"] if not (p["positive"] or p["uncertain"])]
        p = state.get("patient") or {}
        context = {
            "clinical_context": {"patient": p.get("patient"), "encounter": p.get("latest_encounter")},
            "image_quality": state["quality"],
            "CANDIDATE FINDINGS": candidates,
            "findings not seen by the model": not_seen,
            "prior comparison": {"date": comp.get("prior_date"), "resolved_since_prior": comp.get("resolved", [])},
            "triage": {"level": state["triage"].get("level")},
            "EVIDENCE": evidence,
        }
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(json.dumps(context, indent=1, default=str))]
        if state.get("issues"):
            messages.append(HumanMessage("Your previous draft failed verification. Fix ALL of these:\n- "
                                         + "\n- ".join(state["issues"])
                                         + "\nPrevious draft:\n" + json.dumps(state.get("draft"))))
        revision = state.get("revision", 0) + 1
        try:
            draft = (await structured_llm.ainvoke(messages)).model_dump()
            msg = f"draft v{revision}: {len(draft['findings'])} finding(s)"
        except Exception as e:  # malformed JSON from the LLM etc.
            draft = None
            msg = f"draft v{revision} failed: {type(e).__name__}"
        return {"draft": draft, "revision": revision, "trace": _step("draft_report", msg)}

    async def verify(state):
        if state.get("draft") is None:
            issues = ["The LLM did not return a valid report. Return JSON matching the schema."]
        else:
            _, ids = _evidence(state)
            issues = verify_report(state["draft"], state["predictions"], state["prior_comparison"], ids)
        return {"issues": issues,
                "verification_history": [{"revision": state.get("revision", 0), "issues": issues}],
                "trace": _step("verify", "passed" if not issues else f"{len(issues)} issue(s): {issues[0]}")}

    async def human_review(state):
        reasons = list(state.get("review_reasons", []))
        if state.get("issues"):
            reasons.append(f"Verifier issues remain after {state.get('revision')} draft(s): " + " | ".join(state["issues"]))
        # interrupt() pauses the graph and saves its state (checkpointer).
        # The UI shows this payload, then resumes with Command(resume={...}).
        decision = interrupt({"draft": state.get("draft"), "review_reasons": reasons,
                              "triage": state.get("triage"), "heatmaps": state.get("heatmaps")})
        return {"review": decision, "review_reasons": reasons,
                "trace": _step("human_review", f"{decision.get('action')} by {decision.get('reviewer', 'unknown')}")}

    async def finalize(state):
        review = state["review"]
        action = review.get("action", "reject")
        status = {"approve": "approved", "edit": "edited"}.get(action, "rejected")
        final = review.get("report") or state.get("draft")
        saved = {}
        if state.get("patient_id"):
            saved = await call(tools, "save_report", patient_id=state["patient_id"], image_path=state["image_path"],
                               report_json=json.dumps(final), status=status,
                               reviewer=review.get("reviewer", "unknown"))
        return {"final_report": final, "status": status,
                "trace": _step("finalize", f"{status}; saved: {saved.get('saved', False)}")}

    return {"intake": intake, "analyze": analyze, "non_diagnostic": non_diagnostic,
            "retrieve_evidence": retrieve_evidence, "compare_priors": compare_priors, "triage": triage,
            "draft_report": draft_report, "verify": verify, "human_review": human_review, "finalize": finalize}


def route_after_analyze(state) -> str:
    return "retrieve_evidence" if state["quality"]["ok"] else "non_diagnostic"


def route_after_verify(state) -> str:
    if state.get("issues") and state.get("revision", 0) < MAX_REVISIONS:
        return "draft_report"            # loop back with the issues as feedback
    return "human_review"
