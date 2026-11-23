"""Create the demo patient database (SQLite) with synthetic patients.

python -m scripts.seed_patients

Patients, vitals and history are made up. Prior scans and "current" demo
images are taken from the NIH test split (if prepared), so priors carry the
dataset's ground-truth labels and the demo has real images to analyse.
"""
import json
import sqlite3

import pandas as pd

from config import DB_PATH, FINDINGS, SPLITS_DIR

SCHEMA = """
DROP TABLE IF EXISTS patients; DROP TABLE IF EXISTS encounters;
DROP TABLE IF EXISTS prior_scans; DROP TABLE IF EXISTS reports;
CREATE TABLE patients (patient_id TEXT PRIMARY KEY, name TEXT, age INT, sex TEXT,
                       conditions TEXT, smoker INT);
CREATE TABLE encounters (patient_id TEXT, date TEXT, symptoms TEXT, spo2 REAL,
                         heart_rate REAL, temperature_c REAL, clinical_question TEXT);
CREATE TABLE prior_scans (patient_id TEXT, date TEXT, image_path TEXT, findings TEXT, impression TEXT);
CREATE TABLE reports (id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, image_path TEXT,
                      created_at TEXT DEFAULT CURRENT_TIMESTAMP, status TEXT, reviewer TEXT,
                      report_json TEXT);
"""

PATIENTS = [
    ("P001", "Demo Patient A", 67, "M", ["COPD", "hypertension"], 1,
     ["dyspnea", "cough"], 91.0, 104, 37.2, "Worsening breathlessness. Rule out effusion or edema."),
    ("P002", "Demo Patient B", 34, "F", [], 0,
     ["fever", "cough"], 95.0, 98, 38.6, "Fever and productive cough for 4 days. Pneumonia?"),
    ("P003", "Demo Patient C", 58, "M", ["heart failure"], 0,
     ["dyspnea"], 93.0, 112, 36.9, "Known heart failure, more breathless. Fluid overload?"),
    ("P004", "Demo Patient D", 24, "M", [], 1,
     ["chest_pain", "dyspnea"], 94.0, 118, 36.8, "Sudden right chest pain while playing football."),
    ("P005", "Demo Patient E", 72, "F", ["previous lung nodule"], 1,
     ["cough"], 97.0, 82, 36.7, "Follow-up of known nodule. Any change?"),
    ("P006", "Demo Patient F", 45, "F", ["asthma"], 0,
     [], 98.0, 76, 36.6, "Pre-operative chest X-ray."),
]


def main():
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)

    test_csv = SPLITS_DIR / "test.csv"
    test = pd.read_csv(test_csv) if test_csv.exists() else None
    demo_cases = []

    for i, (pid, name, age, sex, conds, smoker, symptoms, spo2, hr, temp, question) in enumerate(PATIENTS):
        con.execute("INSERT INTO patients VALUES (?,?,?,?,?,?)", (pid, name, age, sex, json.dumps(conds), smoker))
        con.execute("INSERT INTO encounters VALUES (?,?,?,?,?,?,?)",
                    (pid, "2026-11-20", json.dumps(symptoms), spo2, hr, temp, question))

        if test is not None:
            # pick one real NIH patient with >=2 images: older image = prior, newer = current demo image
            multi = test.groupby("patient_id").filter(lambda g: len(g) >= 2)["patient_id"].unique()
            nih_pid = multi[i % len(multi)]
            imgs = test[test["patient_id"] == nih_pid].sort_values("image")
            prior, current = imgs.iloc[0], imgs.iloc[-1]
            prior_findings = [f for f in FINDINGS if prior[f] == 1]
            con.execute("INSERT INTO prior_scans VALUES (?,?,?,?,?)",
                        (pid, "2025-06-14", prior["path"], json.dumps(prior_findings),
                         ", ".join(prior_findings) if prior_findings else "No acute findings."))
            demo_cases.append({"patient_id": pid, "image_path": current["path"],
                               "ground_truth": [f for f in FINDINGS if current[f] == 1]})
        elif pid == "P005":
            con.execute("INSERT INTO prior_scans VALUES (?,?,?,?,?)",
                        (pid, "2025-06-14", None, json.dumps(["Nodule"]), "Small right upper lobe nodule."))

    con.commit()
    con.close()
    (DB_PATH.parent / "demo_cases.json").write_text(json.dumps(demo_cases, indent=2))
    print(f"seeded {len(PATIENTS)} patients -> {DB_PATH}")
    if demo_cases:
        print(f"wrote {len(demo_cases)} demo cases -> artifacts/demo_cases.json")


if __name__ == "__main__":
    main()
