import streamlit as st
import os
import subprocess
import sqlite3
import pandas as pd

st.set_page_config(page_title="XS4ALL Web Archive Pipeline", layout="wide")

st.title("🌐 XS4ALL Web Archiving & ML Pipeline")
st.write("Upload WARC files, run the automated preprocessing, and execute the RoBERTa model.")

# Create tabs to mimic your architecture diagram flow
tab1, tab2, tab3 = st.tabs(["1. Ingest & Preprocess", "2. Dataset Exploration", "3. ML Engine"])

# Global Paths
DB_PATH = "websites.db"
INDEX_PATH = "warcs/index_warcs.json"

# --- TAB 1: INGESTION & PREPROCESSING ---
with tab1:
    st.header("Step 1: Parse & Label WARC Files")
    
    # 1. File Upload
    uploaded_files = st.file_uploader("Upload WARC files (.warc, .warc.gz)", accept_multiple_files=True)
    
    if uploaded_files:
        os.makedirs("uploaded_warcs", exist_ok=True)
        for uploaded_file in uploaded_files:
            with open(os.path.join("uploaded_warcs", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Successfully saved {len(uploaded_files)} files to disk.")

    # 2. Pipeline Execution Buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Run WARC Parser", type="primary"):
            st.info("Running parser script...")
            # Triggers your exact CLI tool
            cmd = ["python", "script/warc_parserv4.py", "--warc_dir", "uploaded_warcs", "--db_path", DB_PATH, "--index", INDEX_PATH]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success("Parsing completed successfully!")
            else:
                st.error(f"Parser error: {result.stderr}")
                
    with col2:
        if st.button("🏷️ Run Labeling Functions"):
            st.info("Running heuristic labeling script...")
            cmd = ["python", "script/labeling_function_v7.py", "--db_path", DB_PATH, "--output_train", "train.jsonl", "--output_infer", "infer.jsonl"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success("Labeling and splitting completed!")
            else:
                st.error(result.stderr)

# --- TAB 2: EXPLORATION ---
with tab2:
    st.header("Database Inspection")
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        # Fetch status stats
        df_stats = pd.read_sql_query("SELECT period, COUNT(*) as count FROM websites GROUP BY period", conn)
        st.subheader("Current Database Stats")
        st.bar_chart(df_stats.set_index('period'))
        
        # Quick view table
        df_preview = pd.read_sql_query("SELECT id, seed_url, url, year, period, warc_file FROM websites LIMIT 50", conn)
        st.subheader("Parsed Records Preview (Latest 50)")
        st.dataframe(df_preview, use_container_width=True)
        conn.close()
    else:
        st.warning("Database file not found yet. Run the parser in Tab 1.")

# --- TAB 3: ML ENGINE ---
with tab3:
    st.header("Step 2: Downstream ML Model Workspace")
    
    # Optional skeletonization switch matching your diagram logic
    skip_skeleton = st.checkbox("Skip Skeletonization (--skip-skeleton)", value=False)
    
    # Dropdown to select execution mode
    ml_mode = st.radio("Select ML Operation:", ["Train (Fine-tune RoBERTa)", "Inference (Predict Unknowns)"])
    
    if st.button("🔥 Run ML Engine", type="primary"):
        st.info(f"Initiating ML engine in {ml_mode} mode...")
        
        # Example command structure for your ML engine script
        cmd = ["python", "script/ml_engine.py", "--db_path", DB_PATH, "--mode", ml_mode.lower()]
        if skip_skeleton:
            cmd.append("--skip-skeleton")
            
        # Run process and stream logs to the app UI
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success("ML Engine completed execution!")
            st.code(result.stdout)
        else:
            st.error(result.stderr)