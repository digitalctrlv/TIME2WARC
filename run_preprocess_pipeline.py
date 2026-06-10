#!/usr/bin/env python3
import subprocess
import sys
import argparse

def execute_cmd(cmd):
    print(f"\nRunning command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Pipeline execution halted. Error code encountered.")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="ETL pipeline for processing WARCs for ML training.")
    parser.add_argument("--warc_dir", type=str, required=True, help="Directory containing target raw WARC records.")
    parser.add_argument("--database", type=str, required=True, help="Path to the database")
    parser.add_argument("--url_index", type=str, required=True, help="Path to URL mapping JSON index.")
    parser.add_argument("--skip_skeleton", action="store_true", help="Skip the skeletonization preprocessing.")
    parser.add_argument("--manual_annotations", type=str, default="metadata/manual_annotations.json", help="Path to the manual period annotations user would like to add. Default is set in metadata directory.")
    parser.add_argument("--name", type=str, required=True, help="Name for the output dataset")

    args = parser.parse_args()
    
    dataset_name = args.name

    db_path = args.database
    
    # Step 2 outputs (Masked and labeled versions)
    mask_train = f"training/labeled_train_{dataset_name}.jsonl"
    mask_infer = f"inference/labeled_infer_{dataset_name}.jsonl"
    
    # Step 3 outputs (Masked, labeled and skeletonized versions)
    skeleton_train = f"training/train_{dataset_name}.jsonl"
    skeleton_infer = f"inference/train_{dataset_name}_unknown.jsonl"
    
    print("=== STARTING ARCHIVAL DATA PROCESSING PIPELINE ===")
    
    # 1. Parse and process WARCs to store in the SQLite database
    execute_cmd([
        sys.executable, "script/warc_parserv4.py", # sys.executable contains absolute path to the python interpreter running master script
        "--warc_dir", args.warc_dir,
        "--db_path", db_path,
        "--index", args.url_index
    ])
    
    # 2. Label and split train/infer to use for just masked dataset
    execute_cmd([
        sys.executable, "script/labeling_function_v8.py",
        "--db_path", db_path,
        "--manual_json", args.manual_annotations,
        "--output_train", mask_train,
        "--output_infer", mask_infer
    ])
    
    # OPTIONAL: 3. Skeletonize the HTML-structure
    if not args.skip_skeleton:
        execute_cmd([
            sys.executable, "script/preprocess_skeletonize.py",
            "--input_train", mask_train,
            "--input_infer", mask_infer,
            "--output_train", skeleton_train,
            "--output_infer", skeleton_infer
        ])
    
        print("\n=== PIPELINE RUN COMPLETE ===")
        print(f"Data refined and structural signatures isolated.")
        print(f"Ready for model training. Standalone training file: {skeleton_train}")
    else:
        print("\n=== PIPELINE RUN COMPLETE (Skeletonization Skipped) ===")
        print(f"Data split but not skeletonized.")
        print(f"Train set: {mask_train}")
        print(f"Infer set: {mask_infer}")

if __name__ == "__main__":
    main()