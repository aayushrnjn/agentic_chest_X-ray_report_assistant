"""Streamlit UI.   streamlit run app.py

Flow: choose patient + image -> run agent (pauses at human review)
      -> see trace, predictions, heatmaps, evidence, draft -> approve / edit / reject.
"""
import asyncio
import json
import sqlite3
import uuid

import pandas as pd
import streamlit as st
from langgraph.types import Command

from agent.graph import build_graph
from agent.mcp_client import load_tools
from config import ARTIFACTS, DB_PATH, UPLOAD_DIR

st.set_page_config(page_title="Chest X-ray Report Assistant", layout="wide")


@st.cache_resource
def get_graph():
    # MCP tools open a fresh stdio session per call, so they work across asyncio.run() calls.
    return build_graph(asyncio.run(load_tools()))


def patients() -> list[str]:
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as con:
        return [r[0] for r in con.execute("SELECT patient_id FROM patients")]


def demo_cases() -> list[dict]:
    p = ARTIFACTS / "demo_cases.json"
    return json.loads(p.read_text()) if p.exists() else []


st.title("Chest X-ray Report Assistant")
st.caption("Research demo. AI-assisted drafts must be reviewed by a qualified radiologist. Not for clinical use.")

# ---------------- sidebar: inputs ----------------
with st.sidebar:
    st.header("Case")
    demos = demo_cases()
    source = st.radio("Image source", ["Demo case", "Upload"] if demos else ["Upload"])
    image_path, patient_id, truth = None, None, None
    if source == "Demo case":
        i = st.selectbox("Demo case", range(len(demos)), format_func=lambda k: f"{demos[k]['patient_id']}")
        image_path, patient_id, truth = demos[i]["image_path"], demos[i]["patient_id"], demos[i]["ground_truth"]
    else:
        up = st.file_uploader("Chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if up:
            image_path = str(UPLOAD_DIR / up.name)
            with open(image_path, "wb") as f:
                f.write(up.getbuffer())
        pid = st.selectbox("Patient", ["(none)"] + patients())
        patient_id = None if pid == "(none)" else pid
    if image_path:
        st.image(image_path, caption="Input image")
    run = st.button("Analyse", type="primary", disabled=image_path is None)

graph = get_graph()

if run:
    st.session_state.thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    with st.spinner("Running agent: quality check, CNN, retrieval, triage, drafting, verification..."):
        st.session_state.result = asyncio.run(
            graph.ainvoke({"image_path": image_path, "patient_id": patient_id}, st.session_state.thread))
    st.session_state.truth = truth

if "result" not in st.session_state:
    st.info("Choose a case in the sidebar and press Analyse.")
    st.stop()

state = graph.get_state(st.session_state.thread).values
paused = "__interrupt__" in st.session_state.result

# ---------------- agent trace ----------------
with st.expander("Agent steps", expanded=True):
    for t in state.get("trace", []):
        st.markdown(f"**{t['step']}** — {t['summary']}")

col1, col2 = st.columns([1, 1])

# ---------------- model output ----------------
with col1:
    st.subheader("Model findings (calibrated)")
    preds = state.get("predictions", [])
    if preds:
        df = pd.DataFrame(preds)[["finding", "probability", "threshold", "positive", "uncertain"]]
        st.dataframe(df, hide_index=True)
    if st.session_state.get("truth") is not None:
        st.caption(f"Dataset labels for this image: {', '.join(st.session_state.truth) or 'No Finding'}")
    heatmaps = state.get("heatmaps", {})
    if heatmaps:
        st.subheader("Grad-CAM")
        cols = st.columns(min(len(heatmaps), 2))
        for k, (finding, path) in enumerate(heatmaps.items()):
            cols[k % len(cols)].image(path, caption=finding)
    tri = state.get("triage")
    if tri:
        st.subheader(f"Triage: {tri.get('level', '?').upper()}")
        st.json(tri, expanded=False)

# ---------------- evidence + report ----------------
with col2:
    if state.get("similar_cases"):
        st.subheader("Similar past cases")
        for c in state["similar_cases"]:
            st.markdown(f"`{c['case_id']}` (similarity {c['similarity']}): {c['impression']}")
    if state.get("guidelines"):
        with st.expander(f"Retrieved reference notes ({len(state['guidelines'])})"):
            for g in state["guidelines"]:
                st.markdown(f"`{g['id']}` {g['text']}")

    st.subheader("Draft report")
    for r in state.get("review_reasons", []):
        st.warning(r)
    for h in state.get("verification_history", []):
        st.caption(f"verification of draft v{h['revision']}: {'passed' if not h['issues'] else str(len(h['issues'])) + ' issue(s)'}")

    draft = state.get("draft")
    if paused:
        edited = st.text_area("Edit the report JSON if needed", json.dumps(draft, indent=2), height=380)
        reviewer = st.text_input("Reviewer name", "Dr. Reviewer")
        b1, b2, b3 = st.columns(3)
        decision = None
        if b1.button("Approve"):
            decision = {"action": "approve", "reviewer": reviewer}
        if b2.button("Save edits"):
            try:
                decision = {"action": "edit", "reviewer": reviewer, "report": json.loads(edited)}
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
        if b3.button("Reject"):
            decision = {"action": "reject", "reviewer": reviewer}
        if decision:
            st.session_state.result = asyncio.run(graph.ainvoke(Command(resume=decision), st.session_state.thread))
            st.rerun()
    else:
        st.success(f"Report {state.get('status')}")
        st.json(state.get("final_report"))
