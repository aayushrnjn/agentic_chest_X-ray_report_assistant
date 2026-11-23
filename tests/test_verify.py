"""Unit tests for the verifier and prior comparison (no models needed).   python -m pytest tests"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.verify import compare_with_prior, verify_report  # noqa: E402
from config import FINDINGS  # noqa: E402


def preds(**overrides):
    """All 14 findings negative, then override some: name=(prob, threshold, positive, uncertain)."""
    out = []
    for f in FINDINGS:
        p, th, pos, unc = overrides.get(f, (0.05, 0.5, False, False))
        out.append({"finding": f, "probability": p, "threshold": th, "positive": pos, "uncertain": unc})
    return out


P = preds(Effusion=(0.81, 0.4, True, False), Nodule=(0.36, 0.3, True, True))
COMP = {"Effusion": "new", "Nodule": "persistent", "resolved": [], "prior_date": "2025-06-14"}
IDS = {"MODEL:Effusion", "MODEL:Nodule", "KB:Effusion#0"}


def finding(name, status, prob, change, ids):
    return {"name": name, "status": status, "probability": prob, "change_vs_prior": change,
            "description": "x", "evidence_ids": ids}


def good_draft():
    return {"findings": [finding("Effusion", "present", 0.81, "new", ["MODEL:Effusion", "KB:Effusion#0"]),
                         finding("Nodule", "possible", 0.36, "persistent", ["MODEL:Nodule"])],
            "pertinent_negatives": ["No pneumothorax"], "resolved_since_prior": [],
            "impression": "New effusion. Possible persistent nodule. No pneumothorax."}


def test_good_draft_passes():
    assert verify_report(good_draft(), P, COMP, IDS) == []


def test_hallucinated_finding_caught():
    d = good_draft()
    d["findings"].append(finding("Mass", "present", 0.9, "new", ["MODEL:Mass"]))
    assert any("'Mass' is not supported" in i for i in verify_report(d, P, COMP, IDS))


def test_uncertain_upgraded_to_present_caught():
    d = good_draft()
    d["findings"][1]["status"] = "present"
    assert any("must be 'possible'" in i for i in verify_report(d, P, COMP, IDS))


def test_wrong_probability_and_fake_evidence_caught():
    d = good_draft()
    d["findings"][0]["probability"] = 0.95
    d["findings"][0]["evidence_ids"] = ["KB:Made_Up#3"]
    issues = verify_report(d, P, COMP, IDS)
    assert any("does not match" in i for i in issues)
    assert any("does not exist" in i for i in issues)


def test_omitted_confident_finding_caught():
    d = good_draft()
    d["findings"] = d["findings"][1:]
    d["impression"] = "Possible nodule."
    assert any("missing from the report" in i for i in verify_report(d, P, COMP, IDS))


def test_impression_mentions_unreported_finding():
    d = good_draft()
    d["impression"] += " Mild cardiomegaly."
    assert any("'Cardiomegaly'" in i for i in verify_report(d, P, COMP, IDS))


def test_compare_with_prior():
    priors = [{"date": "2025-06-14", "findings": ["Nodule", "Atelectasis"]}]
    c = compare_with_prior(P, priors)
    assert c["Effusion"] == "new" and c["Nodule"] == "persistent" and c["resolved"] == ["Atelectasis"]
    assert compare_with_prior(P, [])["Effusion"] == "no_prior"
