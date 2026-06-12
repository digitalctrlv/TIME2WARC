#!/usr/bin/env python3
import os
import time
import requests
import sys
import argparse
import subprocess
import streamlit as st
import sqlite3
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from labeling_function import SignalExtractor
from preprocess_skeletonize import extract_html_skeleton

# === Test the imports
try:
    from labeling_function import SignalExtractor
    HAS_SIGNAL_EXTRACTOR = True
except ImportError:
    HAS_SIGNAL_EXTRACTOR = False

try:
    from preprocess_skeletonize import extract_html_skeleton
    HAS_SKELETONIZER = True
except ImportError:
    HAS_SKELETONIZER = False

# =========== FUNCTIONS ====================

# === Masking and skeletonization
def process_payload(payload, run_skeletonized=False):
        if not isinstance(payload, str):
                return ""

        if run_skeletonized and HAS_SKELETONIZER:
            payload = extract_html_skeleton(payload)
        
        if HAS_SIGNAL_EXTRACTOR and hasattr(SignalExtractor, 'regex_patterns'):
            for signal_name, pattern in SignalExtractor.regex_patterns.items():
                payload = pattern.sub("", payload)
        
        return payload[:1000]


# === Retrieving concatinated payloads from SQLite database
# === Also defined in sqlite_db.py but copied locally here for debugging
def get_ml_dataset_fixed(db_path, target_types=None):
    """Yields merged domain payloads from the SQLite database as dictionaries with working IDs."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if target_types is None:
        query = '''
            SELECT 
                MIN(id) as id, 
                seed_url, 
                GROUP_CONCAT(payload, CHAR(10) || CHAR(10)) as payload
            FROM (SELECT * FROM websites ORDER BY LENGTH(url) ASC) 
            GROUP BY seed_url
        '''
        cursor.execute(query)
    else:
        if isinstance(target_types, str):
            target_types = [target_types]
        placeholders = ','.join(['?'] * len(target_types))
        query = f'''
            SELECT 
                MIN(id) as id, 
                seed_url, 
                GROUP_CONCAT(payload, CHAR(10) || CHAR(10)) as payload
            FROM (SELECT * FROM websites WHERE content_type IN ({placeholders}) ORDER BY LENGTH(url) ASC) 
            GROUP BY seed_url
        '''
        cursor.execute(query, tuple(target_types))    
        
    for row in cursor:
        yield dict(row)
    conn.close()

# ================================================================

# ======================== EXECUTION OF THE MODEL ================
def query_huggingface(payload, api_url, headers, retries=3, progress_callback=None):
    """Sends text to HF API. Alerts the UI if the model is waking up and prints strict errors."""
    for attempt in range(retries):
        try:
            response = requests.post(api_url, headers=headers, json={"inputs": payload}, timeout=20)
            if response.status_code == 200:
                return response.json()
            elif "is currently loading" in response.text:
                msg = f"Model booting up on HF (Attempt {attempt+1}/{retries}). Waiting 15s..."
                if progress_callback: progress_callback(0, msg)
                time.sleep(15)
            else:
                # Force print the exact HTTP error code and message from Hugging Face
                print(f"API HTTP ERROR {response.status_code}: {response.text}")
                time.sleep(2)
        except Exception as e:
            # Force print the exact connection failure
            print(f"CONNECTION FAILURE: {e}")
            time.sleep(2)
    return None

def execute_pipeline(db_path, threshold, output_jsonl, skip_skeleton, hf_token, progress_callback=None):
    """Main execution engine callable directly from Streamlit."""
    
    if skip_skeleton:
        # REPLACE THIS PLACEHOLDER WHEN VERSION 2.4 IS UPLOADED
        HF_REPO = "anoukflinkert/time2warc-v2_4-placeholder" 
    else:
        HF_REPO = "anoukflinkert/time2warc-roberta"

    API_URL = f"https://api-inference.huggingface.co/models/{HF_REPO}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    PERIOD_LABELS = ["19971999", "20002002", "20032006", "20072010"]

    if progress_callback: progress_callback(5, "Streaming records from database...")
    
    raw_records = list(get_ml_dataset_fixed(db_path, target_types='text/html'))
    if not raw_records:
        return False, "No records available or processed inside the database."

    df = pd.DataFrame(raw_records)
    total_rows = len(df)
    predicted_periods = []
    confidence_scores = []

    for idx, row in df.iterrows():
        cleaned_text = process_payload(row['payload'], run_skeletonized=not skip_skeleton)
        cleaned_text = cleaned_text[:2500] 

        api_result = query_huggingface(cleaned_text, API_URL, headers, retries=3, progress_callback=progress_callback)

        print(f"DEBUG API RESPONSE: {api_result}")
        period_assignment = "api_error"
        confidence = 0.0

        if api_result and isinstance(api_result, list) and isinstance(api_result[0], list):
            predictions = api_result[0]
            best_pred = max(predictions, key=lambda x: x['score'])
            
            raw_label = best_pred['label']
            confidence = float(best_pred['score'])

            try:
                pred_index = int(raw_label.split("_")[1]) if "LABEL_" in str(raw_label) else int(raw_label)
            except ValueError:
                pred_index = -1 

            if confidence >= threshold and 0 <= pred_index < len(PERIOD_LABELS):
                period_assignment = PERIOD_LABELS[pred_index]
            else:
                period_assignment = "uncertain"

        predicted_periods.append(period_assignment)
        confidence_scores.append(confidence)

        percent_complete = int(((idx + 1) / total_rows) * 100)
        if progress_callback:
            # Scale the loop progress between 5% and 95%
            scaled_percent = 5 + int(percent_complete * 0.9)
            progress_callback(scaled_percent, f"Evaluating documents via API... {percent_complete}%")

    df['predicted_period'] = predicted_periods
    df['predicted_period_confidence'] = confidence_scores

    Path(output_jsonl).parent.mkdir(exist_ok=True, parents=True)
    df.to_json(output_jsonl, orient='records', lines=True, force_ascii=False)

    if progress_callback: progress_callback(98, "Synchronizing database...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for idx, row in df.iterrows():
        cursor.execute("UPDATE websites SET period = ?, confidence = ? WHERE id = ?", 
                       (row['predicted_period'], row['predicted_period_confidence'], row['id']))
    conn.commit()
    conn.close()

    if progress_callback: progress_callback(100, "Classification complete!")
    return True, "Sequence processing complete. Analytical parameters logged to database."