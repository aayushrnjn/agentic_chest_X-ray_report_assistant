"""Pydantic schema the LLM must fill (LangChain structured output)."""
from typing import Literal

from pydantic import BaseModel, Field


class ReportedFinding(BaseModel):
    name: str = Field(description="Exactly one of the 14 finding names, e.g. 'Effusion', 'Pleural_Thickening'")
    status: Literal["present", "possible"] = Field(
        description="'present' only if the model marked it positive and not uncertain; otherwise 'possible'")
    probability: float = Field(description="The calibrated probability copied exactly from the model output")
    change_vs_prior: Literal["new", "persistent", "no_prior"] = Field(
        description="Copied from the prior comparison provided")
    description: str = Field(description="One or two sentences describing the finding in radiology language")
    evidence_ids: list[str] = Field(
        description="IDs of supporting evidence: 'MODEL:<name>', 'KB:<topic>#<n>', 'CASE:<id>'. At least one.")


class RadiologyReport(BaseModel):
    technique: str = Field(description="View and image quality, including any limitation")
    comparison: str = Field(description="Prior study used for comparison with its date, or 'None available'")
    findings: list[ReportedFinding]
    resolved_since_prior: list[str] = Field(default_factory=list,
                                            description="Findings present on the prior but not now")
    pertinent_negatives: list[str] = Field(
        default_factory=list, description="Findings relevant to the clinical question that are NOT seen")
    impression: str = Field(description="2-4 sentences, most important first, answering the clinical question")
    recommendation: str = Field(description="Specific follow-up, or 'None' if not needed")
