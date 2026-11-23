"""Central configuration: every path, model name and threshold lives here.

Nothing in this file imports torch or langchain, so lightweight pieces
(the patient MCP server, scripts) can import it cheaply.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------- data locations (override with environment variables) ----------
DATA_DIR = Path(os.getenv("MEDXRAY_DATA", ROOT / "data"))
NIH_DIR = Path(os.getenv("NIH_DIR", DATA_DIR / "nih"))          # NIH ChestX-ray14 (Kaggle: nih-chest-xrays/data)
IU_DIR = Path(os.getenv("IU_DIR", DATA_DIR / "iu_xray"))        # IU X-ray (Kaggle: raddar/chest-xrays-indiana-university)
KNOWLEDGE_DIR = ROOT / "knowledge"                              # markdown reference notes used for RAG

# ---------- everything the code produces ----------
ARTIFACTS = ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS / "splits"
CHECKPOINT = ARTIFACTS / "densenet121_best.pt"
CALIBRATION = ARTIFACTS / "calibration.json"
TEST_PREDS = ARTIFACTS / "test_preds.npz"
TRIAGE_MODEL = ARTIFACTS / "triage_lgbm.txt"
TRIAGE_META = ARTIFACTS / "triage_meta.json"
CASE_INDEX = ARTIFACTS / "case_index.npz"
CASE_META = ARTIFACTS / "case_meta.json"
KB_INDEX_DIR = ARTIFACTS / "kb_faiss"
KB_CHUNKS = ARTIFACTS / "kb_chunks.pkl"
HEATMAP_DIR = ARTIFACTS / "heatmaps"
UPLOAD_DIR = ARTIFACTS / "uploads"
EVAL_DIR = ARTIFACTS / "eval"
DB_PATH = ARTIFACTS / "clinic.db"

# ---------- the 14 NIH ChestX-ray14 findings ----------
FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

# ---------- model settings ----------
IMG_SIZE = 224
UNCERTAIN_MARGIN = 0.10        # |calibrated prob - threshold| below this => "possible", flagged for review

# ---------- RAG settings ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------- LLM (any LangChain provider: "ollama:llama3.1:8b", "openai:gpt-4o-mini", "anthropic:<model>") ----------
LLM_MODEL = os.getenv("LLM_MODEL", "ollama:llama3.1:8b")
LLM_TEMPERATURE = 0.0

# ---------- agent settings ----------
MAX_REVISIONS = 2              # draft -> verify loops before handing to a human anyway

for _d in (ARTIFACTS, SPLITS_DIR, HEATMAP_DIR, UPLOAD_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
