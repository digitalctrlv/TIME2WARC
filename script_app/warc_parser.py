from warcio.archiveiterator import ArchiveIterator
from urllib.parse import urlparse
from argparse import ArgumentParser
from pathlib import Path
import json
import sqlite3
from sqlite_db import init_db


# ============================= Parsing the  WARCS =============================
# __The First Step in the pipeline__
# This is the first step of the ML classifcation pipeline.
# In the previous notebook index_forming we created an index of seed urls (step 0).
# We will use it to instruct the parser to create json line objects used
# for our classification task.

# This script is part of the run_pipeline script which runs in the command line.
# However, you can view and activate this script by itself in the terminal using 
# the arguments below.

# __WARCS directory__
# Additionally, you need a directory of WARCs. We will be extracting the payload,
# which is understood as the actual source code harvested from the original website,
# stripped from WARC metadata.

# ==============================================================================

class WARCParser:
    """Reads WARC files, filters records by domain index, and saves payloads straight to SQLite."""
    def __init__(self, warc_dir, db_path, url_index):
        self.warc_dir = warc_dir
        self.db_path = db_path
        self.url_index = url_index

        # Pre-compile index paths sorting by length descending for quick waterfall matching
        self.sorted_index_keys = sorted(url_index.keys(), key=len, reverse=True)

        # Create database and tables if they don't exist
        init_db(self.db_path)

    def parse(self):
        from tqdm import tqdm

        directory = Path(self.warc_dir)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Content-types saved to the database
        accepted_types = (
            'text/html', 
            'application/http', 
            'text/css', 
            'application/javascript', 
            'text/javascript'
        )
        
        # Parse through the WARC directory
        warc_files = list(directory.rglob('*.warc')) + list(directory.rglob('*.warc.gz'))

        if not warc_files:
            print(f"No WARC files found in {self.warc_dir}")
            conn.close()
            return
        
        for file_path in tqdm(warc_files, desc="Processing WARCs", unit="file"):
            warc_filename = Path(file_path).name
            file_records_found = 0

            try:
                # Now we start iterating over the WARC
                with open(file_path, 'rb') as f:
                    for record in ArchiveIterator(f, check_digests='ignore', arc2warc=True):
    
                        if record.rec_type == 'response':
                            content_type = None

                            if record.http_headers is not None:
                                status_code = record.http_headers.get_statuscode()
                                if status_code != '200':
                                    continue # skip everything that is not a successful response

                                # Let's look what is inside this response
                                content_type = record.http_headers.get_header('Content-Type')
                                if not (content_type and content_type.startswith(accepted_types)):
                                    continue
                            
                            # Filtering out non-accepted content types
                            if content_type and content_type.startswith(accepted_types):
                                url = record.rec_headers.get_header('WARC-Target-URI')
                                parsed_url = urlparse(url)
                                netloc = parsed_url.netloc
                                
                                # Domain matching with index file
                                matched_key = None
                                
                                # 1: Exact match on domain
                                if netloc in self.url_index:
                                    matched_key = netloc    

                                #Other strategies to handle possible mismatches    
                                # 2: Match WARC domain (has 'www.') against JSON (no 'www.')
                                elif netloc.replace('www.', '') in self.url_index:
                                    matched_key = netloc.replace('www.', '')
                                    
                                # 3: Match WARC domain (no 'www.') against JSON (has 'www.')
                                elif f"www.{netloc}" in self.url_index:
                                    matched_key = f"www.{netloc}"
                                    
                                else:
                                    # 4: Subdirectory/Path match
                                    cleaned_url = url.replace('http://', '').replace('https://', '')
                                    for key in self.url_index.keys():
                                        if cleaned_url.startswith(key):
                                            matched_key = key
                                            break
                                
                                if matched_key:
                                    file_records_found += 1
                                    warc_record_id = record.rec_headers.get_header('WARC-Record-ID')
                                    payload_bytes = record.content_stream().read()
                                    payload_text = payload_bytes.decode('utf-8', errors='ignore')
                                    year = self.url_index.get(matched_key, "Unknown")

                                    # Write to SQLite database
                                    cursor.execute('''
                                        INSERT INTO websites (seed_url, url, year, content_type, payload, warc_filename, warc_record_id)
                                        VALUES (?, ?, ?, ?, ?, ?, ?)
                                    ''', (matched_key, url, year, content_type, payload_text, warc_filename, warc_record_id))
                
                if file_records_found == 0:
                    print(f"Warning: No records found in {file_path}")
                    pass

                conn.commit()

            except Exception as e:
                tqdm.write(f"Error processing {file_path}: {e}")
                conn.rollback()  # Rollback in case of error to avoid partial commits

        conn.close()
        print("\nAll WARC files processed. Database safely populated.")

def load_url_year_mapping(json_file):
    """"Loads the json file that contains the urls and their corresponding years."""
    with open(json_file, 'r') as f:
        return json.load(f)
       
# -----------------------
# Execution logic for standalone mode only

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run WARC Parser manually.")
    parser.add_argument("--warc_dir", default="warcs/toy", help="Directory containing WARC files")
    parser.add_argument("--db_path", default="websites.db", help="Path to output SQLite database")
    parser.add_argument("--index", default="url_index.json", help="Path to URL index JSON file")
    
    args = parser.parse_args()
    
    url_index = load_url_year_mapping(args.index)
    parser_instance = WARCParser(args.warc_dir, args.db_path, url_index)
    parser_instance.parse()

    # python script/warc_parserv3.py --warc_dir warcs/solr --db_path websites.db --index warcs/index_warcs.json
    # python script/warc_parserv3.py --warc_dir D:\warcs\Anouk_xs4all_0423 --db_path websites.db --index warcs/index_warcs.json