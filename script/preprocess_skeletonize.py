#!/usr/bin/env python3
import json
import re
import argparse
from pathlib import Path

def extract_html_skeleton(payload):
    """
    Transforms raw HTML into a dense structural skeleton string.
    Removes natural language text content, but preserves full inline 
    JavaScript blocks, CSS stylesheets, and HTML tag attributes.
    """
    if not isinstance(payload, str) or not payload.strip():
        return ""
    
    # We need to temporarily save JS and CSS codes to prevent them from being stripped in the next steps
    scripts = []
    styles = []
    
    def save_script(match):
        scripts.append(match.group(0))
        return f"___SCRIPT_PLACEHOLDER_{len(scripts)-1}___"
        
    def save_style(match):
        styles.append(match.group(0))
        return f"___STYLE_PLACEHOLDER_{len(styles)-1}___"
    
    # Replace these by placeholders
    payload = re.sub(r'<script\b[^>]*>[\s\S]*?</script>', save_script, payload, flags=re.I)
    payload = re.sub(r'<style\b[^>]*>[\s\S]*?</style>', save_style, payload, flags=re.I)
    
    # 2. Strip HTML comments
    payload = re.sub(r'', '', payload)
    
    # 3. Strip natural language between tags
    payload = re.sub(r'>[^<]+<', '><', payload)
    
    # 4. Place the JS and CSS back in their original positions
    for i, script_content in enumerate(scripts):
        payload = payload.replace(f"___SCRIPT_PLACEHOLDER_{i}___", script_content)
    for i, style_content in enumerate(styles):
        payload = payload.replace(f"___STYLE_PLACEHOLDER_{i}___", style_content)
    
    # 5. Normalize whitespace
    payload = re.sub(r'\s+', ' ', payload).strip()
    
    return payload

def process_file(input_path_str, output_path_str):
    """Reads a JSONL file, skeletonizes payloads, and writes to output."""
    input_path = Path(input_path_str)
    if not input_path.exists():
        print(f"Warning: Target file missing: {input_path}")
        return 0
        
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
             
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            
            # Skeletonization of the payload
            record['payload'] = extract_html_skeleton(record.get('payload', ''))
            
            fout.write(json.dumps(record) + '\n')
            count += 1
            
    return count

def main():
    parser = argparse.ArgumentParser(description="Extract HTML skeletons from pre-split train and inference sets.")
    # Replaced --input_full with train and infer inputs
    parser.add_argument("--input_train", required=True, help="Path to the labeled training jsonl.")
    parser.add_argument("--input_infer", required=True, help="Path to the labeled inference jsonl.")
    parser.add_argument("--output_train", required=True, help="Path to save the skeletonized training file.")
    parser.add_argument("--output_infer", required=True, help="Path to save the skeletonized inference file.")
    args = parser.parse_args()
    
    print(f"Reading labeled training records from {args.input_train}...")
    train_count = process_file(args.input_train, args.output_train)
    
    print(f"Reading labeled inference records from {args.input_infer}...")
    infer_count = process_file(args.input_infer, args.output_infer)
    
    print(f"\nProcessing complete:")
    print(f"-> Training Instances Saved: {train_count} to {args.output_train}")
    print(f"-> Inference Instances Saved: {infer_count} to {args.output_infer}")

if __name__ == "__main__":
    main()

# py script/3preprocess_skeletonize.py --input_full metadata/labeled_outputv7.jsonl --output_train training/train_w23.jsonl --output_infer inference/infer_w23.jsonl
# python script/preprocess_skeletonize.py --input_train training/train_solr_test.jsonl --input_infer inference/infer_solr_test.jsonl --output_train training/train_skeleton_solr_subset.jsonl --output_infer inference/infer_skeleton_solr_subset.jsonl