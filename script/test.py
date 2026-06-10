import pandas as pd
import torch
from numpy import record
from warcio.archiveiterator import ArchiveIterator
from html.parser import HTMLParser
from urllib.parse import urlparse
from argparse import ArgumentParser
import json
import os.path
import sys
from pathlib import Path
import hashlib
import shutil
from transformers import RobertaForSequenceClassification, RobertaTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import sqlite3
from pathlib import Path
import time
from labeling_function_v8 import SignalExtractor
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from statistics import stdev

# db_path = "websites.db"

# if not Path(db_path).exists():
#     print(f"Error: {db_path} does not exist yet.")
#     exit()

# # Check physical file size
# file_size_mb = Path(db_path).stat().st_size / (1024 * 1024)
# print(f"Database File Size: {file_size_mb:.2f} MB\n")

# conn = sqlite3.connect(db_path)

# cursor = conn.cursor()

# # 1. Total unique websites (unique seed_urls)
# cursor.execute("SELECT COUNT(DISTINCT seed_url) FROM payloads")
# unique_sites = cursor.fetchone()[0]
# print(f"Total Unique Websites: {unique_sites}")

# # 2. Breakdown per content type
# print("\nRow count per Content Type:")
# cursor.execute("""
#     SELECT content_type, COUNT(*) 
#     FROM payloads 
#     GROUP BY content_type
# """)

# for content_type, count in cursor.fetchall():
#     print(f" - {content_type}: {count} rows")

# conn.close()]
# conn = sqlite3.connect("websites.db")
# cursor = conn.cursor()

# # Vraag alle tabelnamen op
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# tabellen = [row[0] for row in cursor.fetchall()]
# print(f"Tabellen in database: {tabellen}")

# # Tel de rijen per tabel
# for tabel in tabellen:
#     cursor.execute(f"SELECT COUNT(*) FROM {tabel}")
#     print(f" - Tabel '{tabel}' bevat {cursor.fetchone()[0]} rijen.")

# conn.close()

# import time
# from RoBERTA_v5_chunk_slidwind import ChunkedHTMLDataset, tokenizer

# ===== 1. Load data =====
# df = pd.read_json('./metadata/labeled_outputv7.jsonl', lines=True)
# unknown = (df.query("period_bucket  == 'unknown'"))
# known = (df.query("period_bucket  != 'unknown'"))
# print(len(unknown))
# print(len(known))
# print(len(df))

# label_encoder = LabelEncoder()
# df['label'] = label_encoder.fit_transform(df['period_bucket'])

# print("1. Initializing dataset...")
# start_time = time.time()
# # Instantiate your dataset exactly as you do before passing it to the DataLoader
# dataset = ChunkedHTMLDataset(df['payloads'], df['labels'], tokenizer, max_len=512, stride=256) 
# print(f"Dataset initialized in {time.time() - start_time:.2f} seconds.")

# print("2. Fetching item 0...")
# start_time = time.time()
# # This triggers __getitem__(0) directly
# item = dataset[0] 
# print(f"Item 0 fetched in {time.time() - start_time:.2f} seconds.")

# print("3. Inspecting result:")
# print(f"Tokens shape: {item['input_ids'].shape}")
# t = []

from sqlite_db import get_ml_dataset, get_records, check_total_records, get_all_seed_urls

# all_records = list(get_ml_dataset('websites.db'))
# total_entries = len(all_records)
# print(total_entries)

# all_seeds = list(get_records_by_seed('websites.db', 'rongen17.home.xs4all.nl', 'application/javascript'))
# total_seeds = len(all_seeds)
# print(all_seeds)

# all_data = list(get_records('websites.db'))

# Test if there are as many lines created for an url as for the warc it comes from
# domain_data = list(get_records('websites.db', seed_url='rrump.home.xs4all.nl'))
# file_data = list(get_records('websites.db', warc_filename='IAH-20230310123245051-00001-1635~webharvest-app02.mw.prod.bibliotheek.lcl~8443.warc.gz'))
# print(len(domain_data), len(file_data))

# names = set()
# for file in domain_data:
#     names.add(file['warc_filename'])
# print(names)

# all_html = list(get_records('websites.db', content_type='text/html'))
# domain_html = list(get_records('websites.db', seed_url='rongen17.home.xs4all.nl', content_type='text/html'))
# print(len(all_data), len(domain_data), len(all_html), len(domain_html))

# check_total_records('websites.db')
# all_urls = get_all_seed_urls('websites.db')
# print(f"Total extracted: {len(all_urls)}")

# # Execute
# find_spread_domains('websites.db')

df = pd.read_json('./training/train_solr_test.jsonl', lines=True)
# print(len(df.head(10).iloc[1]['payload']))
print(len(df.iloc[2]['payload']))