"""Image-based retrieval of similar past cases (IU X-ray: real images + real radiology reports).

Build once:  python -m rag.case_index --max-cases 2000
Each frontal IU image is passed through our CNN; its 1024-d embedding and its
report are stored. At query time the new X-ray's embedding is compared with
all stored ones by cosine similarity (a dot product, since vectors are unit length).
"""
import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CASE_INDEX, CASE_META, IU_DIR


def build(max_cases: int | None = None) -> None:
    from models.inference import get_model   # imported here so `search` does not need torch

    reports = pd.read_csv(IU_DIR / "indiana_reports.csv")
    projections = pd.read_csv(IU_DIR / "indiana_projections.csv")
    frontal = projections[projections["projection"] == "Frontal"].merge(reports, on="uid")
    frontal = frontal.dropna(subset=["impression"])
    if max_cases:
        frontal = frontal.sample(n=min(max_cases, len(frontal)), random_state=0)

    files = {p.name: str(p) for p in IU_DIR.rglob("*.png")}
    model = get_model()
    vectors, meta = [], []
    for _, row in tqdm(frontal.iterrows(), total=len(frontal), desc="embedding cases"):
        path = files.get(row["filename"])
        if not path:
            continue
        vectors.append(model.embed(path))
        meta.append({
            "case_id": f"CASE:IU-{row['uid']}",
            "image_path": path,
            "findings_text": str(row.get("findings") or "")[:600],
            "impression": str(row["impression"])[:400],
            "labels": str(row.get("Problems") or ""),
        })

    np.savez_compressed(CASE_INDEX, vectors=np.stack(vectors).astype(np.float32))
    CASE_META.write_text(json.dumps(meta))
    print(f"indexed {len(meta)} cases -> {CASE_INDEX}")


class CaseIndex:
    def __init__(self):
        self.vectors = np.load(CASE_INDEX)["vectors"]
        self.meta = json.loads(CASE_META.read_text())

    def search(self, query_vec: np.ndarray, k: int = 3, exclude_path: str | None = None) -> list[dict]:
        sims = self.vectors @ query_vec
        out = []
        for i in np.argsort(-sims):
            if exclude_path and self.meta[i]["image_path"] == exclude_path:
                continue        # do not "find" the query image itself
            out.append({**self.meta[i], "similarity": round(float(sims[i]), 3)})
            if len(out) == k:
                break
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=None)
    build(ap.parse_args().max_cases)
