"""The LangGraph state: one dict that every node reads from and writes to.

Nodes return only the keys they change. `trace` and `verification_history`
use operator.add so each node's entries are appended instead of overwritten.
"""
import operator
from typing import Annotated, Any, Optional, TypedDict


class XrayState(TypedDict, total=False):
    # inputs
    patient_id: Optional[str]
    image_path: str

    # intake
    patient: Optional[dict]            # demographics + latest encounter
    priors: list[dict]
    quality: dict

    # analysis
    predictions: list[dict]            # calibrated CNN output, one dict per finding
    heatmaps: dict[str, str]           # finding -> overlay PNG path
    uncertain_findings: list[str]
    review_reasons: list[str]          # why a human must look carefully

    # evidence
    similar_cases: list[dict]
    guidelines: list[dict]
    prior_comparison: dict             # finding -> new / persistent / no_prior, plus 'resolved'
    triage: dict

    # report loop
    draft: Optional[dict]
    issues: list[str]
    revision: int
    verification_history: Annotated[list[dict], operator.add]

    # human review
    review: Optional[dict]
    final_report: Optional[dict]
    status: str

    trace: Annotated[list[dict[str, Any]], operator.add]
