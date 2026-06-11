import streamlit as st
import os
import subprocess
import sqlite3
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="TIME2WARC Production Dashboard", layout="wide")

st.title("🌐 TIME2WARC: Archival Web Processing Workbench")
st.write("Ingest raw web archives, perform structural preprocessing transformations, and run downstream inference classification engines.")

DB_PATH = "websites.db"
UPLOAD_DIR = "uploaded_warcs"
OUTPUT_JSONL = "./output/websites_annotated.jsonl"

tab1, tab2, tab3 = st.tabs(["1. Ingest Raw WARC Files", "2. Execution & Classification Engine", "3. Database Inspection Workspace"])

# --- TAB 1: INGESTION ---
with tab1:
    st.header("Step 1: Parse and Ingest Archive Payloads")
    st.markdown(
        "**Note:** This phase extracts payloads from WARC records. "
        "Upload an index file containing target URLs to filter out external out-of-scope hyperlinks."
    )
    
    col_a, col_b = st.columns(2)
    with col_a:
        warc_files = st.file_uploader("Upload target WARC archives (.warc, .warc.gz)", accept_multiple_files=True)
    with col_b:
        index_file = st.file_uploader("Upload tracking target URL configuration index (.json)", accept_multiple_files=False)

    if st.button("🚀 Execute Ingestion Parsing Pipeline", type="primary"):
        if not warc_files or not index_file:
            st.error("Missing input parameters: Please ensure both WARC payloads and URL validation indexes are provided.")
        else:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            
            # Save files onto workspace disk mounts
            for wf in warc_files:
                with open(os.path.join(UPLOAD_DIR, wf.name), "wb") as f:
                    f.write(wf.getbuffer())
                    
            index_path = os.path.join(UPLOAD_DIR, index_file.name)
            with open(index_path, "wb") as f:
                f.write(index_file.getbuffer())
                
            st.info("Parsing active directories and extracting valid responses to SQLite schema tables...")
            
            # Call your ingestion script directly
            cmd = ["python", "script_app/warc_parser.py", "--warc_dir", UPLOAD_DIR, "--db_path", DB_PATH, "--index", index_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success("Ingestion routine processed effectively without errors.")
            else:
                st.error(f"Execution Error encountered during parser stream: {result.stderr}")

# --- TAB 2: INFERENCE ENGINE ---
with tab2:
    st.header("Step 2: Deep Transformer Classification Engine")
    st.write("Run the fine-tuned RoBERTa classification sequence over parsed unannotated database items.")

    skip_skeleton = st.checkbox("Skip Structural Skeletonization (Retain text block values for evaluation arrays)", value=False)
    confidence_level = st.slider("Confidence Gating Filtering Threshold", 0.0, 1.0, 0.60, 0.05)

    if st.button("🧠 Trigger Inference Model Thread", type="primary"):
        st.info("Loading architecture configuration paths and evaluating historical sequences...")
        
        # 1. Initialize a native Streamlit progress bar container
        progress_bar = st.progress(0, text="Initializing model weights...")

        if os.path.exists("pipeline.py"):
            pipeline_script_path = "pipeline.py"
        elif os.path.exists("script_app/pipeline.py"):
            pipeline_script_path = "script_app/pipeline.py"
        else:
            st.error("Missing Script: Could not locate 'pipeline.py' in the current root directory or 'script/' folder.")
            st.stop()

        # Call the refactored CLI script with explicit arguments
        cmd = [
            "python", pipeline_script_path, 
            "--db_path", DB_PATH, 
            "--threshold", str(confidence_level),
            "--output_jsonl", OUTPUT_JSONL
        ]
        if skip_skeleton:
            cmd.append("--skip-skeleton")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            # Catch our custom progress string format
            if "PROGRESS_UPDATE:" in line:
                try:
                    percentage = int(line.split(":")[1].strip())
                    # Update the web UI bar dynamically!
                    progress_bar.progress(percentage, text=f"Evaluating documents via RoBERTa... {percentage}%")
                except ValueError:
                    pass
            
        stderr_output = process.stderr.read()
        if process.returncode == 0:
            progress_bar.progress(100, text="Inference complete!")
            st.success("Sequence processing complete. Analytical parameters logged to database.")
            
            if os.path.exists(OUTPUT_JSONL):
                with open(OUTPUT_JSONL, "rb") as file:
                    st.download_button(
                        label="📥 Download Annotated JSONL Dataset Backup",
                        data=file,
                        file_name="websites_annotated.jsonl",
                        mime="application/jsonlines"
                    )
        else:
            st.error(f"Inference execution engine failure: {stderr_output}")

# --- TAB 3: INSPECTION ---
with tab3:
    st.header("Step 3: Database Schema Inspection & Data Verification")
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        
        # Summary distribution chart metric extraction layer
        df_stats = pd.read_sql_query("SELECT COALESCE(period, 'Unprocessed') as period, COUNT(*) as count FROM websites GROUP BY period", conn)
        st.subheader("Data Stratification Statistics Overview")
        st.bar_chart(df_stats.set_index('period'))
        
        # Content table view
        df_preview = pd.read_sql_query("SELECT id, seed_url, url, year, period, warc_filename FROM websites LIMIT 100", conn)
        st.subheader("Database Record Stream (Latest 100 Elements)")
        st.dataframe(df_preview, use_container_width=True)
        conn.close()
    else:
        st.warning("Active target database container not found yet. Execute parsing pipeline arrays inside Tab 1.")