import streamlit as st
import pandas as pd
import json
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, date

import os

# ── Point to demo folder directory for loading demo/rank.py safely ──
BASE_DIR = Path(__file__).parent
DEMO_DIR = BASE_DIR / "demo"

import sys
sys.path.insert(0, str(DEMO_DIR))
try:
    from rank import (
        score_candidate,
        CID_TO_IDX,
        jd_emb,
        jd_skill_embs,
        summary_embs,
        career_embs,
        skill_embed_map,
        CONSTS,
        REFERENCE_DATE
    )
except ImportError as e:
    st.error(f"Failed to import from demo/rank.py: {e}")
    st.stop()

st.set_page_config(
    page_title="Redrob AI Candidate Discovery Sandbox",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Style Customizations
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    h1, h2, h3 {
        color: #f8fafc !important;
    }
    .card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #334155;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3b82f6;
    }
    .reasoning-text {
        font-family: monospace;
        font-size: 0.9rem;
        background-color: #0b0f19;
        padding: 10px;
        border-radius: 6px;
        border-left: 3px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Redrob AI Candidate Discovery Sandbox")
st.markdown("### Stage 1 / Stage 3 Verification Environment")
st.write("This interactive dashboard validates the scoring, integrity gates, and reasoning engine on small candidate samples or full candidate batches under the 5-minute compute budget constraint.")

# Sidebar Details
with st.sidebar:
    st.header("⚙️ Configuration")
    st.success("🧪 **Demo Mode** — using `demo/artifacts_sample` (1K candidates)")
    st.info("System running in **Optimized V7** Mode (CPU-only, no hosted LLM dependencies).")
    
    st.subheader("Population Calibration Constants")
    st.json({
        "semantic_p1": CONSTS.get("semantic_p1"),
        "semantic_p99": CONSTS.get("semantic_p99"),
        "skill_denom": CONSTS.get("skill_denom"),
        "production_p99": CONSTS.get("production_p99"),
        "prod_rescale": CONSTS.get("production_rescale")
    })
    
    st.write(f"**Reference Date**: `{REFERENCE_DATE}`")
    st.write(f"**Artifacts Dir**: `artifacts_sample`")

# Upload candidate data
SAMPLE_JSONL = BASE_DIR / "demo" / "sample_1k.jsonl"
uploaded_file = st.file_uploader("Upload a subset of candidates (.jsonl) — or leave empty to auto-load the 1K demo sample", type=["jsonl"])

if uploaded_file is not None:
    st.write("### Processing candidates...")
    candidates = []
    
    # Read uploaded JSONL
    for line in uploaded_file:
        line_str = line.decode("utf-8").strip()
        if line_str:
            try:
                candidates.append(json.loads(line_str))
            except json.JSONDecodeError:
                pass
                
    st.success(f"Loaded {len(candidates)} candidates.")
    
    # Score candidates
    scored_candidates = []
    progress_bar = st.progress(0)
    
    for i, c in enumerate(candidates):
        cid = c.get("candidate_id", "Unknown")
        score, reasoning = score_candidate(c)
        
        scored_candidates.append({
            "candidate_id": cid,
            "score": score,
            "reasoning": reasoning,
            "name": c.get("profile", {}).get("anonymized_name", "N/A"),
            "title": c.get("profile", {}).get("current_title", "N/A"),
            "yoe": c.get("profile", {}).get("years_of_experience", 0),
            "country": c.get("profile", {}).get("country", "N/A")
        })
        progress_bar.progress((i + 1) / len(candidates))
        
    df = pd.DataFrame(scored_candidates)
    
    # Sort descending
    df_sorted = df.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    df_sorted["rank"] = df_sorted.index + 1
    
    # Display results
    st.subheader("🏆 Candidate Leaderboard")
    
    # Format display columns
    display_df = df_sorted[["rank", "candidate_id", "name", "title", "yoe", "country", "score"]]
    st.dataframe(display_df, use_container_width=True)
    
    # Export CSV Option
    csv = df_sorted[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False)
    st.download_button(
        label="📥 Download Ranked Submission CSV",
        data=csv,
        file_name="submission_sample.csv",
        mime="text/csv"
    )
    
    # Inspect Top Profiles
    st.subheader("🔍 Detailed Profiles Inspection")
    top_n = st.slider("Select number of candidates to inspect:", min_value=1, max_value=min(100, len(df_sorted)), value=5)
    
    for idx, row in df_sorted.head(top_n).iterrows():
        with st.expander(f"#{row['rank']} | {row['name']} ({row['title']}) — Score: {row['score']:.4f}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**Candidate ID**: `{row['candidate_id']}`")
                st.write(f"**Experience**: `{row['yoe']} years`")
                st.write(f"**Country**: `{row['country']}`")
            
            with col2:
                st.markdown("**Evidence-based System Reasoning:**")
                st.markdown(f"<div class='reasoning-text'>{row['reasoning']}</div>", unsafe_allow_html=True)
else:
    # Auto-load demo/sample_1k.jsonl if present
    st.write("---")
    if SAMPLE_JSONL.exists():
        st.info(f"🧪 No file uploaded — auto-loading demo sample: `{SAMPLE_JSONL.name}` ({SAMPLE_JSONL.stat().st_size // 1024} KB)")
        if st.button("▶️ Run Ranker on Demo Sample (1K candidates)"):
            candidates = []
            with open(SAMPLE_JSONL, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            candidates.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            st.success(f"Loaded {len(candidates)} candidates from demo sample.")
            scored_candidates = []
            progress_bar = st.progress(0)
            for i, c in enumerate(candidates):
                cid = c.get("candidate_id", "Unknown")
                score, reasoning = score_candidate(c)
                scored_candidates.append({
                    "candidate_id": cid,
                    "score": score,
                    "reasoning": reasoning,
                    "name": c.get("profile", {}).get("anonymized_name", "N/A"),
                    "title": c.get("profile", {}).get("current_title", "N/A"),
                    "yoe": c.get("profile", {}).get("years_of_experience", 0),
                    "country": c.get("profile", {}).get("country", "N/A")
                })
                progress_bar.progress((i + 1) / len(candidates))
            df = pd.DataFrame(scored_candidates)
            df_sorted = df.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
            df_sorted["rank"] = df_sorted.index + 1
            st.subheader("🏆 Demo Leaderboard")
            display_df = df_sorted[["rank", "candidate_id", "name", "title", "yoe", "country", "score"]]
            st.dataframe(display_df, use_container_width=True)
            csv = df_sorted[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False)
            st.download_button(
                label="📥 Download Demo Submission CSV",
                data=csv,
                file_name="demo_submission.csv",
                mime="text/csv"
            )
            top_n = st.slider("Inspect top N profiles:", min_value=1, max_value=min(50, len(df_sorted)), value=5)
            for idx, row in df_sorted.head(top_n).iterrows():
                with st.expander(f"#{row['rank']} | {row['name']} ({row['title']}) — Score: {row['score']:.4f}"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**Candidate ID**: `{row['candidate_id']}`")
                        st.write(f"**Experience**: `{row['yoe']} years`")
                        st.write(f"**Country**: `{row['country']}`")
                    with col2:
                        st.markdown("**Evidence-based System Reasoning:**")
                        st.markdown(f"<div class='reasoning-text'>{row['reasoning']}</div>", unsafe_allow_html=True)
    else:
        st.write("👉 Please upload a candidates `.jsonl` file to run the ranker on a custom subset.")
