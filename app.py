import streamlit as st
import os
import subprocess
import sqlite3
import pandas as pd
from pathlib import Path

import sys


st.set_page_config(page_title="TIME2WARC Production Dashboard", layout="wide")

st.title("TIME2WARC: Periodize Early Web Collections")
st.write("Ingest raw WARC files in the TIME2WARC engine, preprocess and run the classifier to inspect and download the new periodization of your collection in the database.")
st.write("This classifier performs best on eary web collections and predicts whether archived websites have been created or operationalized in any of the following periods:" \
        " **1997-1999** | **2000-2002** | **2003-2006** | **2007-2010**.   ")

DB_PATH = "websites.db"
UPLOAD_DIR = "uploaded_warcs"
OUTPUT_JSONL = "./output/websites_annotated.jsonl"

tab1, tab2, tab3 = st.tabs(["1. Ingest raw WARC files", "2. Run the classification engine", "3. Database workspace"])

# ================== TAB 1 INGEST AND PARSE ==================
with tab1:
    st.header("Parse and store payloads to the database")
    st.markdown(
        "**Note:** This phase extracts payloads from WARC records. "
        "Upload an index file containing target URLs to filter out external out-of-scope hyperlinks."
    )
    
    col_a, col_b = st.columns(2)
    with col_a:
        warc_files = st.file_uploader("Upload target WARC archives (.warc, .warc.gz)", accept_multiple_files="directory", max_upload_size=None)
    with col_b:
        index_file = st.file_uploader("Upload tracking target URL configuration index (.json)", accept_multiple_files=False)

    if st.button(" PARSE AND INGEST ", type="primary"):
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

# ================== TAB 2 CLASSIFIER ENGINE ==================
with tab2:
    st.header("Transformer period-classification Engine")
    st.write("Run the fine-tuned RoBERTa classification sequence over parsed unannotated database items.")

    # Checkbox determines both the preprocessing step AND the model weights used
    skip_skeleton = st.checkbox("Skip **Skeletonization** (removal of all (natural) language embedded between <tags></tags>)", value=False)
    confidence_level = st.slider("Prediction confidence filtering threshold", 0.0, 1.0, 0.60, 0.05)

    if st.button("Run the classifier!", type="primary"):
        st.info("Loading architecture configuration paths and evaluating historical sequences...")
        
        # 1. Initialize progress bar
        progress_bar = st.progress(0, text="Initializing model weights...")

        # 2. Locate the pipeline script safely
        if os.path.exists("pipeline.py"):
            pipeline_script_path = os.path.abspath("pipeline.py")
        elif os.path.exists("script_app/pipeline.py"):
            pipeline_script_path = os.path.abspath("script_app/pipeline.py")
        else:
            st.error("Missing Script: could not locate 'pipeline.py' in the current root directory or 'script_app/' folder.")
            st.stop()

        db_target = os.path.abspath(DB_PATH)
        output_target = os.path.abspath(OUTPUT_JSONL)

        # 3. DYNAMIC MODEL SWITCHING
        if skip_skeleton:
            # If skipping skeletonization, use the v2_4 model
            st.warning("Skeletonization bypassed. Loading default model: models/v2_3_optimal_weights_fold1")
            model_target = os.path.abspath("./models/v2_3_optimal_weights_fold1")
        else:
            # Default model for skeletonized data
            model_target = os.path.abspath("./models/v2_4_optimal_weights_fold2")

        # 4. Construct the crash-proof execution command
        cmd = [
            sys.executable, pipeline_script_path, 
            "--db_path", db_target, 
            "--model_path", model_target, 
            "--threshold", str(confidence_level),
            "--output_jsonl", output_target
        ]
        
        # Append the flag if the checkbox is ticked
        if skip_skeleton:
            cmd.append("--skip-skeleton")

        st.info(f"Executing Engine: {' '.join(cmd)}")

        # 5. Run the subprocess
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            if "PROGRESS_UPDATE:" in line:
                try:
                    percentage = int(line.split(":")[1].strip())
                    progress_bar.progress(percentage, text=f"Evaluating documents via RoBERTa... {percentage}%")
                except ValueError:
                    pass
        
        stderr_output = process.stderr.read()
        
        # 6. Evaluate the exit state
        if process.returncode == 0:
            progress_bar.progress(100, text="Classification complete!")
            st.success("Sequence processing complete. Analytical parameters logged to database.")
            
            if os.path.exists(output_target):
                with open(output_target, "rb") as file:
                    st.download_button(
                        label="📥 Download annotated JSONL dataset",
                        data=file,
                        file_name="websites_annotated.jsonl",
                        mime="application/jsonlines"
                    )
        else:
            st.error(f"Inference execution engine failure. Exit Code: {process.returncode}")
            st.code(stderr_output if stderr_output.strip() else "EMPTY TRACEBACK: Windows killed the process (likely Out of Memory).")

# ================ TAB 3: DATABASE VIEWER ==============
with tab3:
    st.header("Database viewer for inspecting results")
    
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(websites)")
        existing_columns = [info[1] for info in cursor.fetchall()]
        
        if 'predicted_period_confidence' in existing_columns:
            query = """
                SELECT 
                    seed_url, 
                    year, 
                    period, 
                    confidence, 
                    warc_filename, 
                    payload 
                FROM websites 
                WHERE period IS NOT NULL
            """
        else:
            st.warning("Column 'predicted_period_confidence' is missing. Falling back to old schema.")
            query = """
                SELECT 
                    seed_url, 
                    year, 
                    period, 
                    warc_filename, 
                    payload 
                FROM websites 
                WHERE period IS NOT NULL
            """
            
        # 2. Safely execute the query
        df_full = pd.read_sql_query(query, conn)

        # 3. Process dataframe for the UI (Truncate Payload)
        if not df_full.empty:
            df_display = df_full.copy()
            df_display['payload'] = df_display['payload'].fillna('').apply(
                lambda x: x[:50] + "..." if len(x) > 50 else x
            )

            st.subheader("Predicted Domain Outcomes")
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # 4. Create the Downloadable JSON containing the full payload
            st.subheader("Export Data")
            st.write("Download the complete, untruncated database records.")
            
            json_dump = df_full.to_json(orient="records", force_ascii=False, indent=2)
            st.download_button(
                label="📥 Download Annotated Database (JSON)",
                data=json_dump,
                file_name="time2warc_final_predictions.json",
                mime="application/json"
            )
        else:
            st.info("No predictions found in the database yet. Run the ML Engine in Tab 2.")
            
        conn.close()
    else:
        st.warning("Active target database container not found yet. Execute parsing pipeline in Tab 1.")