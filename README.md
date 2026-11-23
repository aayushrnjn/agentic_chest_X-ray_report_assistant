# Chest X-ray Report Assistant

A research prototype that takes a chest X-ray and produces a **verified, evidence-linked draft report** for a radiologist to review.

- **Deep learning:** DenseNet-121 multi-label classifier (14 findings, NIH ChestX-ray14), Grad-CAM heatmaps, image embeddings for similar-case search
- **ML:** per-finding probability calibration (temperature + bias), per-finding thresholds, LightGBM urgency model with SHAP-style explanations
- **RAG (LangChain):** hybrid BM25 + FAISS search with reciprocal rank fusion and a cross-encoder reranker; image-to-image retrieval over real IU X-ray reports
- **MCP:** 4 servers (imaging, patient, knowledge, triage) exposing tools, resources and a prompt; usable from the agent or any MCP client
- **LangGraph:** multi-step workflow with quality routing, a draft → verify → revise loop, and human-in-the-loop approval via `interrupt()`

> Not a medical device. Not for clinical use. Labels in public datasets are noisy and the triage model is trained on rule-derived labels.

## Architecture

```
                 ┌─────────────── LangGraph agent ────────────────┐
image + patient →│ intake → analyze ─┬─ poor image → non_diagnostic ┐
                 │                   └→ retrieve_evidence          │
                 │                      → compare_priors → triage  │
                 │                      → draft_report ⇄ verify ───┤
                 │                                 human_review ←──┘ → finalize
                 └───────────────────────┬──────────────────────────┘
                                         │ MCP (stdio)
        ┌──────────────┬─────────────────┼───────────────┬───────────────┐
   imaging server  patient server   knowledge server   triage server
   CNN, Grad-CAM,  SQLite history,  hybrid RAG over    LightGBM
   similar cases   priors, reports  reference notes    urgency
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# LLM: free local option
ollama pull llama3.1:8b          # or: export LLM_MODEL=openai:gpt-4o-mini  (and pip install langchain-openai)
```

Datasets (Kaggle):
- NIH ChestX-ray14 → `data/nih/` (needs `Data_Entry_2017.csv`, `test_list.txt`, `images_*/images/*.png`)
- IU X-ray (raddar/chest-xrays-indiana-university) → `data/iu_xray/` (needs `indiana_reports.csv`, `indiana_projections.csv`, images)

Paths can be changed with `NIH_DIR`, `IU_DIR` environment variables. A GPU (Colab/Kaggle T4 is fine) is needed for training.

## Build steps (in order)

| Step | Command | Output |
|---|---|---|
| 1. Splits (patient-level) | `python -m models.dataset --max-patients 6000` | `artifacts/splits/*.csv` |
| 2. Train CNN | `python -m models.train --epochs 8` | `densenet121_best.pt` |
| 3. Calibrate | `python -m models.calibrate` | `calibration.json` |
| 4. Evaluate CNN | `python -m eval.evaluate_model` | AUROC, ECE, plots, `test_preds.npz` |
| 5. Triage model | `python -m ml.triage` | `triage_lgbm.txt` |
| 6. Knowledge index | `python -m rag.knowledge_base` | FAISS + chunks |
| 7. Case index | `python -m rag.case_index --max-cases 2000` | image embeddings + reports |
| 8. Demo patients | `python -m scripts.seed_patients` | `clinic.db`, `demo_cases.json` |
| 9. Run | `streamlit run app.py` or `python -m agent.run --demo 0` | |
| 10. Evaluate agent | `python -m eval.evaluate_agent --n 30` | precision/recall, verifier stats |
| Tests | `python -m pytest tests` | verifier unit tests |

Drop `--max-patients` to train on the full dataset (~112k images).

## Using the MCP servers from Claude Desktop or another MCP client

```json
{
  "mcpServers": {
    "xray-imaging":   {"command": "/path/to/.venv/bin/python", "args": ["/path/to/medxray/mcp_servers/imaging_server.py"]},
    "xray-knowledge": {"command": "/path/to/.venv/bin/python", "args": ["/path/to/medxray/mcp_servers/knowledge_server.py"]}
  }
}
```

## Results

Fill in after running steps 4 and 10 (`artifacts/eval/`): mean test AUROC, per-finding AUROC chart, ECE before/after calibration, reliability diagram, agent precision/recall, % of first drafts caught by the verifier.

## Limitations

- NIH labels were mined from reports by NLP and are partly wrong; "Infiltration" is especially noisy.
- No external validation: performance on other hospitals' images will be lower (domain shift).
- Triage labels are rule-derived, so the LightGBM model imitates those rules.
- Knowledge notes are short educational summaries, not licensed clinical guidelines.
