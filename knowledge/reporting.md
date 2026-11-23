# Reporting and triage notes

Educational notes for this demo project, not clinical policy.

## Report_Structure
A chest X-ray report is usually organised as: technique (view, for example PA or AP, and any limitations), comparison (prior studies used, with dates), findings (systematic description of lungs, pleura, heart and mediastinum, bones and devices), and impression (a short conclusion answering the clinical question, most important finding first). Pertinent negatives, such as "no pneumothorax", are included when the clinical question raised that possibility. Recommendations should be specific and only given when they change management.

## Uncertainty_Language
Hedging should match the actual confidence. Commonly used scales go from "consistent with" (high confidence) through "suggestive of" and "possible" to "unlikely". An AI-assisted draft should state when a finding is near the model's decision threshold and should never upgrade a possible finding to a definite one. When image quality limits interpretation, the report should say so explicitly instead of reporting low-confidence findings.

## Comparison_With_Priors
Comparing with prior imaging is one of the most valuable steps in chest radiograph interpretation. A finding that is new since the last study is more significant than one that has been stable for years. Reports should describe change as new, increased, decreased, stable or resolved, and give the date of the comparison study. A stable nodule over two years, for example, is much less concerning than a newly appearing one.

## Urgent_Findings
Some findings call for direct communication with the referring clinician rather than a routine report: tension pneumothorax, large or enlarging pneumothorax, new large effusion with breathlessness, pulmonary edema with low oxygen saturation, and misplaced tubes or lines. A new suspicious mass usually needs expedited but not emergency follow-up. Urgency always depends on the clinical picture, especially vital signs such as oxygen saturation, heart rate and temperature.

## AI_Assistance_Limits
AI models trained on public datasets such as NIH ChestX-ray14 learn from labels that were mined automatically from reports and are partly wrong. Performance drops on images from different hospitals, machines or patient populations (domain shift). Models can rely on shortcuts such as chest drains for pneumothorax or hospital-specific markers. For these reasons an AI draft is a starting point that must be reviewed and signed by a qualified radiologist.
