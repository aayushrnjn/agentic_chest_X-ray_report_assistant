"""NIH ChestX-ray14 data preparation and PyTorch Dataset.

Run once:  python -m models.dataset --max-patients 6000
It writes train/val/test CSVs to artifacts/splits/.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from config import FINDINGS, NIH_DIR, SPLITS_DIR
from models.preprocess import load_xray


def _find_images(root: Path) -> dict:
    """Map '00000001_000.png' -> full path. Kaggle splits images into images_001..012 folders."""
    paths = {p.name: str(p) for p in root.rglob("*.png")}
    if not paths:
        raise FileNotFoundError(f"No PNG images under {root}. Set NIH_DIR to the dataset folder.")
    return paths


def prepare_splits(max_patients: int | None = None, val_frac: float = 0.1, seed: int = 42) -> None:
    df = pd.read_csv(NIH_DIR / "Data_Entry_2017.csv")
    df = df.rename(columns={"Image Index": "image", "Finding Labels": "labels", "Patient ID": "patient_id",
                            "Patient Age": "age", "Patient Gender": "sex", "View Position": "view"})

    # Multi-hot encoding: "Effusion|Mass" -> Effusion=1, Mass=1, others 0. "No Finding" -> all 0.
    for f in FINDINGS:
        df[f] = df["labels"].str.split("|").apply(lambda labs, f=f: int(f in labs))

    paths = _find_images(NIH_DIR)
    df["path"] = df["image"].map(paths)
    df = df.dropna(subset=["path"])

    # The official lists are split BY PATIENT. Never split images randomly:
    # the same patient in train and test leaks information and inflates scores.
    test_names = set((NIH_DIR / "test_list.txt").read_text().split())
    test_df = df[df["image"].isin(test_names)]
    trainval = df[~df["image"].isin(test_names)]

    rng = np.random.default_rng(seed)
    if max_patients:  # subsample whole patients for a faster (Colab-sized) run
        keep = rng.choice(trainval["patient_id"].unique(), size=min(max_patients, trainval["patient_id"].nunique()), replace=False)
        trainval = trainval[trainval["patient_id"].isin(keep)]
        keep_test = rng.choice(test_df["patient_id"].unique(), size=min(max_patients // 4, test_df["patient_id"].nunique()), replace=False)
        test_df = test_df[test_df["patient_id"].isin(keep_test)]

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(gss.split(trainval, groups=trainval["patient_id"]))

    cols = ["image", "path", "patient_id", "age", "sex", "view"] + FINDINGS
    for name, part in [("train", trainval.iloc[tr_idx]), ("val", trainval.iloc[va_idx]), ("test", test_df)]:
        part[cols].to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(part):,} images, {part['patient_id'].nunique():,} patients")

    print("\nPositive rate per finding (train):")
    print(trainval.iloc[tr_idx][FINDINGS].mean().sort_values().round(4).to_string())


class XrayDataset(Dataset):
    def __init__(self, csv_path: Path, transform):
        self.df = pd.read_csv(csv_path)
        self.labels = self.df[FINDINGS].values.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        img = self.transform(load_xray(self.df.at[i, "path"]))
        return img, torch.from_numpy(self.labels[i])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-patients", type=int, default=None, help="subsample for a quick run, e.g. 6000")
    args = ap.parse_args()
    prepare_splits(args.max_patients)
