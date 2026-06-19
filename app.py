import streamlit as st
import os
import subprocess
import sqlite3
import pandas as pd
import pathlib
from pathlib import Path
import sys
import shutil
import base64

# =============== STYLING ==========================
def load_css_fonts(css_path, fonts_dir_path):
    font_face_css = ""
    fonts_dir = pathlib.Path(fonts_dir_path)

    if fonts_dir.exists():
        for font_file in fonts_dir.glob("*.ttf"):
            font_name = font_file.stem
            with open(font_file, "rb") as f_font:  # Gebruik f_font
                font_data = base64.b64encode(f_font.read()).decode("utf-8")
            
            font_face_css += f"""
            @font-face {{
                font-family: "{font_name}";
                src: url(data:font/truetype;charset=utf-8;base64,{font_data}) format("truetype");
                font-weight: normal;
                font-style: normal;
            }}
            """
    with open(css_path, "r", encoding="utf-8") as f_css:
        css_content = f_css.read()

    st.html(f"<style>{font_face_css}\n{css_content}</style>")

css_path = pathlib.Path("./assets/styles.css")
fonts_dir = pathlib.Path("./assets/fonts")

load_css_fonts(css_path, fonts_dir)

icon_path = pathlib.Path("assets/img/application_hourglass-0.png")
diagram_path = pathlib.Path("assets/img/app_process_diagram.png")

with open(icon_path, "rb") as f:
    icon_base64 = base64.b64encode(f.read()).decode("utf-8")

with open(diagram_path, "rb") as f:
    diagram_base64 = base64.b64encode(f.read()).decode("utf-8")

# ============== HEADING ==========================
st.set_page_config(page_title="TIME2WARC Dashboard", layout="wide")

st.html(f"""
    <h1 id="time-2-warc-periodize-early-web-collections" class="retro-title">
        <img src="data:image/png;base64,{icon_base64}" class="retro-icon" />
        TIME2WARC 
        <p> Periodize Early Web Collections 1997-1999 | 2000-2002 | 2003-2006 | 2007-2010</p>
    </h1>

   <div style="margin-top: 20px; margin-left: auto; margin-right: auto; width: 80%; background: #c0c0c0; box-shadow: inset -1px -1px #0a0a0a, inset 1px 1px #fff, inset -2px -2px grey, inset 2px 2px #dfdfdf; padding: 3px; box-sizing: border-box;">
        
        <div style="background: #000080; background: linear-gradient(90deg, #000080, #1084d0); height: 30px; min-height: 30px; display: flex; align-items: center; padding: 0 8px; box-sizing: border-box;">
            <div style="font-family: 'MS-Sans-Serif', Arial, sans-serif; font-size: 15px; color: #ffffff; font-weight: bold; margin: 0; padding: 0; line-height: 1;">
                Application Process Diagram
            </div>
        </div>
        
        <div style="margin: 2px;">
            <div class="diagram-container" style="background: #ffffff; border: 2px solid #808080; box-shadow: inset 1px 1px #0a0a0a, inset -1px -1px #fff; padding: 0px; display: flex; justify-content: center;">
                <img src="data:image/png;base64,{diagram_base64}" class="retro-diagram" />
            </div>
            <div>
                <p>The TIME2WARC engine helps you discover when the websites in your early web collection possibly were created or last operationalized.</p>
                <p>Early web collections often face similar difficulties: there's only one version of a site, or harvests started long after it was created. While creators sometimes leave "last modified" clues, or metadata drops a hint, these indicators don't always give a truthful representation of the time period the website actually belongs to.</p>
                <p><b>TIME2WARC lets the source code speak for themselves.</b> It analyzes the site's code against web computing history. Does it rely on 90s table layouts, or show early signs of CSS and Javascript? Let the specialized Transformer model predict the true historical period of your web collection—and inspect the results for yourself!</p>
            </div>
        </div>
        
    </div>
""")

DB_PATH = "websites.db"
UPLOAD_DIR = "uploaded_warcs"
OUTPUT_JSONL = "./output/websites_annotated.jsonl"

st.header("Let's start!")

tab1, tab2, tab3 = st.tabs(["1. Ingest raw WARC files", "2. Run the classification engine", "3. Database workspace"])

# ================== TAB 1 INGEST AND PARSE ==================
with tab1:
    st.header("Parse WARCs and store payloads to the database")
    st.markdown(
        "**Note:** This phase extracts payloads from WARC records. "
        "Upload an index file containing target URLs to filter out external out-of-scope hyperlinks."
    )
    
    col_a, col_b = st.columns(2)
    with col_a:
        warc_files = st.file_uploader("Upload target WARC archives (.warc, .warc.gz)", accept_multiple_files="directory")
    with col_b:
        index_file = st.file_uploader("Upload tracking target URL configuration index (.json)", accept_multiple_files=False)

    with col_a:
        if st.button(" ▶️ Parse WARCs  ", key="win98"):
            if not warc_files or not index_file:
                st.error("Missing input parameters: Please ensure both WARC payloads and URL validation indexes are provided.")
            else:
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                
                # Save files onto workspace disk mounts
                for wf in warc_files:
                    file_path = os.path.join(UPLOAD_DIR, wf.name)

                    # Any subdirectories embedded in wf.name are created first
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    with open(file_path, "wb") as f:
                        f.write(wf.getbuffer())
                        
                index_path = os.path.join(UPLOAD_DIR, index_file.name)
                with open(index_path, "wb") as f:
                    f.write(index_file.getbuffer())
                    
                st.info("⏳Parsing active directories and extracting valid responses to SQLite schema tables...")
                
                # Call your ingestion script directly
                cmd = [sys.executable, "script_app/warc_parser.py", "--warc_dir", UPLOAD_DIR, "--db_path", DB_PATH, "--index", index_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success("✅ Ingestion routine processed effectively without errors.")
                else:
                    st.error(f"🚧 Execution Error encountered during parser stream: {result.stderr}")
    
    with col_b:
        if st.button(" ⟳ Restart ", type="secondary"):
            if os.path.exists(DB_PATH):
                try:
                    os.remove(DB_PATH)
                    os.remove("./uploaded_warcs")
                    st.success("Database file permanently deleted. Re-run the parser to create a new one.")
                except PermissionError:
                    st.error("Cannot delete the file right now. Another process is using it.")
            else:
                st.warning("The database file does not exist.")
            if os.path.exists("./uploaded_warcs"):
                try:
                    shutil.rmtree("./uploaded_warcs")
                    st.success("Database and uploaded WARCs removed.")
                except Exception as e:
                    st.error(f"Failed to delete directory: {e}")

# ================== TAB 2 CLASSIFIER ENGINE ==================
with tab2:
    st.header("Transformer period-classification Engine")

    st.html(f"""
            <details class="win95-treeview">
                <summary>ℹ️ Information on the model and tunable parameters</summary>
                <div class="win95-textbox">
                    <p>Before using this application, please review the model's training limitations and performance metrics to ensure an accurate interpretation of your classification results.</p>
            
                    <h3>Training data and Domain limitation</h2>
                    <p>This model is highly domain-specific. For this study, RoBERTa-base was fine-tuned on a small, niche dataset consisting exclusively of <b>XS4ALL personal homepages</b> extracted from the web archive of the <b>KB National Library of the Netherlands</b>.</p>
                    <p>Furthermore, the original data distribution and the selection interests of the web curators created a <b>severe class imbalance</b>. Because of this, the model is significantly better at recognizing the dominant, early periods (1997–1999 and 2000–2002). Exercise extreme caution when interpreting results for the later time buckets.</p>
            
                    <h3>Performance Metrics (Stratified 5-Fold Cross-Validation)</h3>
                    <p>The dataset was evaluated across two distinct configurations. Please note the substantial performance drop when natural language text is removed:</p>
                    <div class="win95-grid">
                        <table>
                            <thead>
                                <tr>
                                    <th>Metric (Mean ± σ)</th>
                                    <th>Default Model (v2.3)<br><small>(Text + HTML)</small></th>
                                    <th>Skeletonized Model (v2.4)<br><small>(HTML Only)</small></th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr><td><b>Macro F₁-Score</b></td><td>0.500 ± 0.072</td><td>0.422 ± 0.091</td></tr>
                                <tr><td><b>Standard Accuracy</b></td><td>63.2%</td><td>35.4%</td></tr>
                                <tr><td><b>Windowed Accuracy (±1 class)</b></td><td>82.9%</td><td>50.8%</td></tr>
                            </tbody>
                        </table>
                    </div>
            
                    <h3>The 1000-character constraint and payload behavior</h3>
                    <p>Due to computational resource limits during training, the model only processed the <b>first 2000 characters</b> of each webpage payload. We are fully aware of the generalization issues this introduces, and we hope to resolve this constraint in future iterations.
                    <p>This strict limit impacts how the application handles inputs:</p>
                    <ul>
                        <li>Multiple content types: If you select multiple content types for a single URL, they are concatenated into one continuous string. Because of the 1000-character cutoff, the model will likely only see the initial HTML portion and completely ignore subsequent blocks.</li>
                        <li>JavaScript & CSS interference: If the primary HTML code is exceptionally short, trailing JavaScript or CSS might bleed into the 1000-character window. Because the model was not explicitly trained on standalone script or style formats, encountering them may degrade prediction accuracy. However, since late-90s/early-00s CSS and JavaScript frequently lived inline or at the top of the <head>, the network is not entirely blind to these structures.</li>
                    </ul>
            
                <h3>Learn more</h3>
                <p>For a comprehensive breakdown of the training configurations, cross-validation runs, and architectural experiments, please read the full report paper available on the <a href="https://github.com/digitalctrlv/TIME2WARC">GitHub repository</a>.

                </div>
            </details>
        """
    )

    # with st.popover("Filter settings"):
    # Checkbox determines both the preprocessing step AND the model weights used
    st.subheader("Option 1: Skeletonization of source code")
    skip_skeleton = st.checkbox("Skip **Skeletonization** (removal of all (natural) language embedded between <tags></tags>)", value=True, help="Unchecking this box removes all natural language between HTML tags (e.g., leaving only <html><h1><p class='paragraph'></p></html>) and switches to Model v2.4. While v2.4 was built to target pure web grammar for date attribution, it unfortunately underperformed compared to v2.3 (which keeps the text context intact). For the most reliable results, it is recommended to leave this box checked, but feel free to experiment!")

    st.subheader("Option 2: Select preferred MIME content-types")
    with st.popover("Filter content types:"):
        selected_content_types = st.multiselect(
            "Select target content types to classify",
            options=['text/html', 
                    'application/http', 
                    'text/css', 
                    'application/javascript', 
                    'text/javascript'], 
            default=["text/html"]
        )

    st.subheader("Option 3: Prediction confidence filtering threshold")
    confidence_level = st.slider("How confident should the model be in its predictions?", 0.0, 1.0, 0.60, 0.05)

    st.subheader("Ready?")
    if st.button("Run the classifier!", type="primary"):
        if not selected_content_types:
            st.error("Execution halted: You must select at least one content type.")
            st.stop()
        st.info("Evaluating web historical patterns... Maybe browse through a web archive while you're waiting?")
        
        progress_bar = st.progress(0, text="Calculating...")

        if os.path.exists("pipeline.py"):
            pipeline_script_path = os.path.abspath("pipeline.py")
        elif os.path.exists("script_app/pipeline.py"):
            pipeline_script_path = os.path.abspath("script_app/pipeline.py")
        else:
            st.error("Missing Script: could not locate 'pipeline.py' in the current root directory or 'script_app/' folder.")
            st.stop()

        db_target = os.path.abspath(DB_PATH)
        output_target = os.path.abspath(OUTPUT_JSONL)

        # DYNAMIC MODEL SWITCHING
        if skip_skeleton:
            # If skipping skeletonization, use the v2_4 model
            model_target = os.path.abspath("./models/v2_3_optimal_weights_fold1")
        else:
            # Default model for skeletonized data
            model_target = os.path.abspath("./models/v2_4_optimal_weights_fold2")

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

        cmd.append("--content_types")
        cmd.extend(selected_content_types)

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
        
        if process.returncode == 0:
            progress_bar.progress(100, text="Classification complete!")
            
            st.success("Classification data saved to database. Go to Tab 3 to view and export your annotated dataset.")

        else:
            st.error(f"Inference execution engine failure. Exit Code: {process.returncode}")
            st.code(stderr_output if stderr_output.strip() else "EMPTY TRACEBACK: Windows killed the process (likely Out of Memory).")
    
    if st.button(" ⟳ Restart and remove database ", type="secondary"):
        if os.path.exists(DB_PATH):
            try:
                os.remove(DB_PATH)
                st.success("Database file permanently deleted. Re-run the parser to create a new one.")
            except PermissionError:
                st.error("Cannot delete the file right now. Another process is using it.")
        else:
            st.warning("The database file does not exist.")
        if os.path.exists("./uploaded_warcs"):
            try:
                shutil.rmtree("./uploaded_warcs")
                st.success("Database and uploaded WARCs removed.")
            except Exception as e:
                st.error(f"Failed to delete directory: {e}")

# ================ TAB 3: DATABASE VIEWER ==============
with tab3:
    st.header("Database viewer for inspecting results")
    
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(websites)")
        existing_columns = [info[1] for info in cursor.fetchall()]
        
        if not existing_columns:
            st.info("The database is currently empty. Please upload and parse WARC files in Tab 1.")
        else:
            if 'confidence' in existing_columns:
                query = """
                    SELECT 
                        seed_url, 
                        MAX(year) as year, 
                        MAX(period) as period, 
                        MAX(confidence) as confidence, 
                        MAX(warc_filename) as warc_filename, 
                        GROUP_CONCAT(payload, CHAR(10) || CHAR(10)) as payload 
                    FROM websites
                    GROUP BY seed_url 
                    HAVING MAX(period) IS NOT NULL
                """
            else:
                st.warning("Column 'confidence' is missing. Falling back to old schema.")
                query = """
                    SELECT 
                        seed_url, 
                        MAX(year) as year, 
                        MAX(period) as period, 
                        MAX(warc_filename) as warc_filename, 
                        GROUP_CONCAT(payload, CHAR(10) || CHAR(10)) as payload 
                    FROM websites
                    GROUP BY seed_url 
                    HAVING MAX(period) IS NOT NULL
                """
                
            # Safely execute the query
            df_full = pd.read_sql_query(query, conn)
            
            base_url = "http://webarchief.kb.nl:8080/archived/"
            timestamp = df_full['warc_filename'].str.extract(r'IAH-(\d{14})', expand=False)
            df_full['wayback_link'] = base_url + timestamp + "/https://" + df_full['seed_url'].astype(str)

            # Process dataframe for the UI (truncate payload)
            if not df_full.empty:
                df_display = df_full.copy()
                df_display['payload'] = df_display['payload'].fillna('').apply(
                    lambda x: x[:50] + "..." if len(x) > 50 else x
                )

                st.subheader("Predicted outcomes")
                st.dataframe(df_display, 
                            use_container_width=True, 
                            hide_index=True,
                            column_config={
                                "wayback_link": st.column_config.LinkColumn(
                                    label="Open in the Wayback Machine!",
                                    display_text="🔗 Open website"
                                )
                            })
                
                # Downloadable json with full payloads
                st.subheader("Export data")
                st.write("Download the complete, untruncated database records.")
                
                json_dump = df_full.to_json(orient="records", force_ascii=False, indent=2)
                st.download_button(
                    label="📥 Download annotated database (JSON)",
                    data=json_dump,
                    file_name="time2warc_final_predictions.json",
                    mime="application/json"
                )
            else:
                st.info("No predictions found in the database yet. Run the ML Engine in Tab 2.")

        
        st.write("⚠️**Note on download size**: Depending on your selection, this file can be very large. The download will retrieve every archived MIME content-type you selected in the previous step. To prevent unexpected waiting times, run the default query below to check the exact number of payloads you are about to download per seed URL.")

        # SQL EXPLORATION
        st.markdown("---")
        st.subheader("Custom SQL Workspace")
        st.write("Write raw SQLite queries to filter, aggregate, and explore your parsed data.")

        st.html("""
        <details class="win95-treeview" style="margin-bottom: 15px;">
            <summary>💡 View useful example queries</summary>
            <div class="win95-textbox" style="background: #fff; padding: 10px;">
                <p style="margin-bottom: 8px !important;"><b>1. Unique domains per web period:</b><br>
                <code>SELECT period, COUNT(DISTINCT seed_url) as unique_sites, ROUND(AVG(confidence), 2) as avg_confidence FROM websites GROUP BY period ORDER BY unique_sites DESC;</code></p>
                <p style="margin-bottom: 8px !important;"><b>2. High confidence sites by period:</b><br>
                <code>SELECT period, COUNT(DISTINCT seed_url) as highly_confident_sites FROM websites WHERE confidence > 0.85 GROUP BY year ORDER BY year ASC;</code></p>
                <p style="margin-bottom: 0px !important;"><b>3. Sites grouped by web period:</b><br>
                <code>SELECT period, GROUP_CONCAT(DISTINCT seed_url) as associated_urls, COUNT(DISTINCT seed_url) as total_sites FROM websites GROUP BY period ORDER BY period ASC;</code></p>
            </div>
        </details>
        """)
            
        user_query = st.text_area("SQL Query", value="SELECT seed_url," \
        "COUNT(*) as payloads, " \
        "MAX(period) as period, " \
        "MAX(confidence) as confidence, " \
        "MAX(year) as year, " \
        "GROUP_CONCAT(DISTINCT content_type) as content_types " \
        "FROM websites " \
        "GROUP BY seed_url " \
        "ORDER BY payloads DESC;")
        
        if st.button("Execute SQL"):
            try:
                custom_df = pd.read_sql_query(user_query, conn)
                st.dataframe(custom_df, use_container_width=True, hide_index=True)
                
                # Provide an instant CSV download for whatever they queried
                st.download_button(
                    label="Download Query Results (CSV)",
                    data=custom_df.to_csv(index=False).encode('utf-8'),
                    file_name="custom_sql_export.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"SQL Error: {e}")    
            
        conn.close()
    else:
        st.warning("Active target database container not found yet. Execute parsing pipeline in Tab 1.")

