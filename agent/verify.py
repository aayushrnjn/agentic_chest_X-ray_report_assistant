"""Deterministic report verifier: checks every claim in the LLM draft against facts we hold.

Plain Python on purpose: an LLM checking another LLM can miss the same
mistakes, while these rules are exact, fast and unit-testable.
"""
import re

from config import FINDINGS


def _mentions(text: str, finding: str) -> bool:
    words = finding.lower().replace("_", " ")
    return re.search(rf"\b{re.escape(words)}\b", text.lower()) is not None


def verify_report(draft: dict, predictions: list[dict], comparison: dict, evidence_ids: set[str]) -> list[str]:
    issues = []
    pred = {p["finding"]: p for p in predictions}
    reported = {}

    for f in draft.get("findings", []):
        name = f.get("name", "")
        if name not in FINDINGS:
            issues.append(f"'{name}' is not one of the 14 allowed finding names.")
            continue
        if name in reported:
            issues.append(f"'{name}' is listed twice.")
        reported[name] = f
        p = pred[name]

        if not (p["positive"] or p["uncertain"]):
            issues.append(f"'{name}' is not supported: the model gives {p['probability']} "
                          f"(threshold {p['threshold']}). Remove it.")
            continue
        if f.get("status") == "present" and (p["uncertain"] or not p["positive"]):
            issues.append(f"'{name}' must be 'possible', not 'present': the model is uncertain about it.")
        if abs(float(f.get("probability", -1)) - p["probability"]) > 0.02:
            issues.append(f"'{name}' probability {f.get('probability')} does not match the model ({p['probability']}).")
        expected_change = comparison.get(name, "no_prior")
        if f.get("change_vs_prior") != expected_change:
            issues.append(f"'{name}' change_vs_prior should be '{expected_change}', not '{f.get('change_vs_prior')}'.")
        ids = f.get("evidence_ids") or []
        if not ids:
            issues.append(f"'{name}' has no evidence_ids.")
        unknown = [i for i in ids if i not in evidence_ids]
        if unknown:
            issues.append(f"'{name}' cites evidence that does not exist: {unknown}.")

    # omissions: a confident positive finding must not be dropped
    for p in predictions:
        if p["positive"] and not p["uncertain"] and p["finding"] not in reported:
            issues.append(f"'{p['finding']}' is a confident model finding ({p['probability']}) but is missing from the report.")

    # the impression may only name findings that are reported or explicitly negated
    negatives = " ".join(draft.get("pertinent_negatives", [])) + " " + " ".join(draft.get("resolved_since_prior", []))
    for name in FINDINGS:
        if _mentions(draft.get("impression", ""), name) and name not in reported and not _mentions(negatives, name):
            issues.append(f"The impression mentions '{name}', which is not among the reported findings, "
                          f"resolved findings or pertinent negatives.")
    return issues


def compare_with_prior(predictions: list[dict], priors: list[dict]) -> dict:
    """finding -> 'new' / 'persistent' / 'no_prior', plus 'resolved': [...] and the prior's date."""
    current = [p["finding"] for p in predictions if p["positive"] or p["uncertain"]]
    usable = [p for p in priors if p.get("findings") is not None]
    if not usable:
        return {**{f: "no_prior" for f in current}, "resolved": [], "prior_date": None}
    latest = usable[0]
    before = set(latest["findings"])
    out = {f: ("persistent" if f in before else "new") for f in current}
    out["resolved"] = sorted(f for f in before if f not in current)
    out["prior_date"] = latest.get("date")
    return out
