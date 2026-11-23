"""LightGBM urgency model: CNN probabilities + patient vitals/symptoms -> routine / priority / urgent.

python -m ml.triage      (train; run after eval/evaluate_model.py so real CNN outputs are used)

IMPORTANT, and worth saying in interviews: public X-ray datasets have no
triage outcomes. So the training LABELS here come from a transparent rule set
(roughly how a clinician would prioritise) plus 5% label noise. Features use
the CNN's real test-set probabilities when available. The model therefore
learns to imitate the rules; with real data you would train on actual
outcomes (ICU admission, time-to-treatment) instead. The pipeline, feature
design and explanation via SHAP-style contributions stay the same.
"""
import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from config import FINDINGS, TEST_PREDS, TRIAGE_META, TRIAGE_MODEL

LEVELS = ["routine", "priority", "urgent"]
VITALS = ["age", "spo2", "heart_rate", "temperature_c", "fever", "dyspnea", "chest_pain"]
FEATURES = [f"p_{f}" for f in FINDINGS] + VITALS


def _rule_label(p: dict, v: dict) -> int:
    """The 'ground truth' generator. Deliberately simple and explainable."""
    if v["spo2"] < 90:
        return 2
    if p["Pneumothorax"] > 0.5 and (v["dyspnea"] or v["chest_pain"]):
        return 2
    if p["Edema"] > 0.5 and v["dyspnea"] and v["spo2"] < 94:
        return 2
    if max(p["Pneumonia"], p["Consolidation"]) > 0.5 and v["fever"] and (v["spo2"] < 94 or v["heart_rate"] > 110):
        return 2
    if max(p["Mass"], p["Pneumothorax"], p["Effusion"], p["Pneumonia"], p["Consolidation"], p["Edema"]) > 0.5:
        return 1
    if v["age"] > 70 and (v["dyspnea"] or v["fever"]):
        return 1
    if v["spo2"] < 94 or v["heart_rate"] > 110 or v["temperature_c"] > 38.5:
        return 1
    return 0


def _synthetic_dataset(n: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    if TEST_PREDS.exists():
        probs_pool = np.load(TEST_PREDS)["cal_probs"]
        probs = probs_pool[rng.integers(0, len(probs_pool), n)]
        print(f"using real CNN probabilities from {TEST_PREDS.name}")
    else:
        probs = rng.beta(0.5, 6, size=(n, len(FINDINGS)))
        print("test_preds.npz not found, sampling probabilities from a Beta distribution")

    vit = {
        "age": rng.integers(18, 95, n),
        "spo2": np.clip(rng.normal(96, 3, n), 75, 100).round(1),
        "heart_rate": np.clip(rng.normal(85, 18, n), 45, 170).round(),
        "temperature_c": np.clip(rng.normal(37.1, 0.8, n), 35, 41).round(1),
        "fever": rng.random(n) < 0.2,
        "dyspnea": rng.random(n) < 0.25,
        "chest_pain": rng.random(n) < 0.15,
    }
    vit["fever"] = vit["fever"] | (vit["temperature_c"] > 38.0)

    X = np.column_stack([probs] + [vit[k].astype(float) for k in VITALS])
    y = np.array([
        _rule_label(dict(zip(FINDINGS, probs[i])), {k: vit[k][i] for k in VITALS}) for i in range(n)
    ])
    flip = rng.random(n) < 0.05                                     # label noise
    y[flip] = rng.integers(0, 3, flip.sum())
    return X, y


def train():
    X, y = _synthetic_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    model = lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.05,
                               num_leaves=31, class_weight="balanced", verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], feature_name=FEATURES)
    pred = model.predict(X_te)
    print(classification_report(y_te, pred, target_names=LEVELS, digits=3))
    model.booster_.save_model(str(TRIAGE_MODEL))
    TRIAGE_META.write_text(json.dumps({"features": FEATURES, "levels": LEVELS,
                                       "macro_f1": round(f1_score(y_te, pred, average="macro"), 4)}, indent=2))
    print(f"saved -> {TRIAGE_MODEL}")


class TriageModel:
    def __init__(self):
        self.booster = lgb.Booster(model_file=str(TRIAGE_MODEL))

    def predict(self, probabilities: dict, vitals: dict) -> dict:
        row = [float(probabilities.get(f, 0.0)) for f in FINDINGS]
        row += [float(vitals.get(k, default)) for k, default in
                zip(VITALS, [50, 97, 80, 37.0, 0, 0, 0])]
        x = np.array([row])
        proba = self.booster.predict(x)[0]
        level = int(np.argmax(proba))
        # pred_contrib gives per-feature contributions (SHAP values) for each class
        contrib = self.booster.predict(x, pred_contrib=True)[0]
        per_class = len(FEATURES) + 1
        c = contrib[level * per_class: (level + 1) * per_class - 1]
        top = np.argsort(-c)[:4]
        return {
            "level": LEVELS[level],
            "probabilities": {lv: round(float(p), 3) for lv, p in zip(LEVELS, proba)},
            "top_factors": [{"feature": FEATURES[i], "value": row[i], "contribution": round(float(c[i]), 3)}
                            for i in top if c[i] > 0],
            "note": "trained on rule-derived labels; demo model, not clinically validated",
        }


if __name__ == "__main__":
    train()
