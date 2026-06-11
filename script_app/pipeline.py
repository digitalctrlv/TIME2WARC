#!/usr/bin/env python3
import os
import sys
import argparse
import sqlite3
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from labeling_function import SignalExtractor
from sqlite_db import init_db, get_ml_dataset
from preprocess_skeletonize import extract_html_skeleton

# Test tje imports
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

try:
    from sqlite_db import get_ml_dataset
    HAS_DB_METHOD = True
except ImportError:
    HAS_DB_METHOD = False

def process_payload(payload, run_skeletonized=False):
        if not isinstance(payload, str):
                return ""

        if run_skeletonized and HAS_SKELETONIZER:
            payload = extract_html_skeleton(payload)
        
        if HAS_SIGNAL_EXTRACTOR and hasattr(SignalExtractor, 'regex_patterns'):
            for signal_name, pattern in SignalExtractor.regex_patterns.items():
                payload = pattern.sub("", payload)
        
        return payload[:1000]

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

def main():
    parser = argparse.ArgumentParser(description="TIME2WARC Engine")
    parser.add_argument("--db_path", default="websites.db", help="Path to production SQLite database")
    parser.add_argument("--model_path", default="anoukflinkert/TIME2WARC_masked", help="Path to fine-tuned model weights directory")
    parser.add_argument("--skip-skeleton", action="store_true", help="Disable skeletonization preprocessing")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold classification gating parameter")
    parser.add_argument("--output_jsonl", default="./output/websites_annotated.jsonl", help="Output path for downloaded results")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Executing on device: {device}")

    # Creating the labels
    print("Initializing dynamic label restoration mapping...", flush=True)
    label_encoder = LabelEncoder()
    
    label_encoder.fit(["19971999", "20002002", "20032006", "20072010"])

    print(f"Loading the model...")
    # Windows forward-slashes unfortunately...
    local_model_str = os.path.abspath(args.model_path)
    
    if not os.path.exists(local_model_str):
        print(f"Error: Local model directory not found at: {local_model_str}", flush=True)
        sys.exit(1)

    model = RobertaForSequenceClassification.from_pretrained(local_model_str)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = model.to(device)
    model.eval()

    # Loading data
    print("Streaming records with 'text/html' content type from the SQLite database")
    raw_records = list(get_ml_dataset_fixed(args.db_path, target_types='text/html'))

    if not raw_records:
           print("No records available or processed inside the database")
           return
    
    df = pd.DataFrame(raw_records)
    print("Retrieved data")

    predicted_periods = []
    confidence_scores = []

    print("Beginning the classification loop...This might take a while...")
    total_rows = len(df)
    print("Maybe browse a little through a web archive?")

    # Start of the loop
    for idx, row in df.iterrows():
        cleaned_text = process_payload(row['payload'], run_skeletonized=not args.skip_skeleton)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            encoding = tokenizer(
                text=cleaned_text,
                max_length=512,
                padding=True, 
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))
        confidence = float(np.max(probs))

        # Check against the threshold for classification
        if confidence >= args.threshold:
            period_assignment = label_encoder.inverse_transform([pred_label])[0]
        else:
            period_assignment = "uncertain"

        predicted_periods.append(period_assignment)
        confidence_scores.append(confidence)

        if idx > 0 and idx % 100 == 0:
            print(f"Evaluated sequences: {idx}/{len(df)}...")

        percent_complete = int(((idx + 1) / total_rows) * 100)
        print(f"Progress update: {percent_complete}", flush=True)
        # End of loop

    df['predicted_period'] = predicted_periods
    df['predicted_period_confidence'] = confidence_scores

    # Downloadable file with predictions
    Path(args.output_jsonl).parent.mkdir(exist_ok=True, parents=True)
    df.to_json(args.output_jsonl, orient='records', lines=True, force_ascii=False)

    print("Writing predictions back to SQLite tracking records...")
    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    for idx, row in df.iterrows():
        cursor.execute("""
            UPDATE websites 
            SET period = ?, confidence = ? 
            WHERE id = ?
        """, (row['predicted_period'], row['predicted_period_confidence'], row['id']))
        
    conn.commit()
    conn.close()
    print("Database synchronized with engine prediction outputs.")

if __name__ == "__main__":
    main()