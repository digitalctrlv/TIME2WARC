import json
import re
from collections import Counter
import argparse
from argparse import ArgumentParser
from unittest import signals
from pathlib import Path
import sqlite3

from sqlite_db import get_ml_dataset

# =========================== Labeling functions =========================== #
# This component uses regular expression string matching to find occurrences of
# evident periodical markers, such as doctype declarations in an html document.

class SignalExtractor:
    """Encapsulates pattern configuration + extraction logic as a unit."""
    
    regex_patterns = {
        'html2': re.compile(r'<!doctype\s+html\s+public[^>]*?html\s+2\.0[^>]*>', re.I),
        'html32': re.compile(r'<!doctype\s+html\s+public[^>]*?html\s+3\.2[^>]*>', re.I),
        'html4.01': re.compile(r'<!doctype\s+html\s+public[^>]*?html\s+4\.01[^>]*>', re.I),
        'html4.01_any': re.compile(r'<!doctype\s+html\s+public[^>]*?html\s+4\.01(?:\s+(?:strict|transitional|frameset))?[^>]*>', re.I),
        'xhtml1_any': re.compile(r'<!doctype\s+html\s+public[^>]*?xhtml\s+1\.0(?:\s+(?:strict|transitional|frameset))?[^>]*>', re.I),
        'html5_doctype': re.compile(r'<!doctype\s+html>(?!.*dtd)', re.I), 
        'html5_charset': re.compile(r'<meta\s+charset=["\']?utf-8["\']?>', re.I),
        'bootstrap': re.compile(r'href=["\'].*?bootstrap.*?\.css["\']', re.I),
        'js_legacy': re.compile(r'language=["\']?javascript1\.[0-3]["\']?', re.I),

        # Generator signals broken down by release dates provided
        'gen_1997_1999': re.compile(
            r'<meta\s+name=["\']?(?:generator|formatter)["\']?\s+content=["\']?[^>]*?'
            r'(?:' # Open de hoofdgroep
            r'macromedia\s+flash\s*[1-4]\.0|' 
            r'macromedia\s+dreamweaver\s*(?:1\.[02]|[23]\.0)|'
            r'golive\s*5\.0|frontpage\s*4\.0|'
            r'netobjects\s*fusion\s*[1-4]|'
            r'staroffice[/\s]*(?:[124]\.0|[35]\.[01])'
            r')', re.I
        ),
        
        'gen_2000_2002': re.compile(
            r'<meta\s+name=["\']?(?:generator|formatter)["\']?\s+content=["\']?[^>]*?'
            r'(?:golive\s*6\.0|frontpage\s*5\.0|netobjects\s*fusion\s*(?:5|mx|7)\b|'
            r'dreamweaver\s*(?:4\.mx\b)|golive\s*6|'
            r'fusion\s*(?:57\.[0-9])'
            r'openoffice\.org\s*1\.0|editplus\s*2\.(?:0[01]|10[ac]?|11))', re.I
        ),

        'gen_2003_2006': re.compile(
            r'<meta\s+name=["\']?(?:generator|formatter)["\']?\s+content=["\']?[^>]*?'
            r'(?:golive\s*(?:cs\s*)?[78]\.0|'
            r'frontpage\s*5\.0|'
            r'netobjects\s*fusion\s*(?:7\.5|[89]|10)\b|'
            r'openoffice\.org\s*(?:1\.1|2\.0)|'
            r'macromedia\s+(?:dreamweaver\s*(?:MX|8\.0)|flash\s*MX)'
            r'editplus\s*2\.(?:11|12|20|21|30))', re.I
        ),
        'gen_2007_2010': re.compile(
            r'<meta\s+name=["\']?(?:generator|formatter)["\']?\s+content=["\']?[^>]*?'
            r'(?:frontpage\s*6\.0|expression\s*7\.0|'
            r'netobjects\s*fusion\s*11\b|'
            r'openoffice\.org\s*(?:2\.[1-4]|3\.[0-3])|'
            r'editplus\s*(?:2\.31|3\.0[01]))', re.I
        ),
        'gen_post_2010': re.compile(
            r'<meta\s+name=["\']?(?:generator|formatter)["\']?\s+content=["\']?[^>]*?'
            r'(?:netobjects\s*fusion\s*(?:xii|2013|2015))', re.I
        )
    }

    def __init__(self, patterns: dict = None):
        self.regex_patterns = patterns or self.regex_patterns
    
    def extract(self, payload: str) -> dict:
        """Process the payload to extract technical 'signals' and determine the period bucket."""

        if not payload:
            return {}
    
        html_lower = payload[:1000].lower() # Cut char to headers (includes our current signals) to save CPU time

        signals = {
            'legacy_dom': 'document.all' in html_lower or 'document.layers' in html_lower, 
            'js_legacy': bool(self.regex_patterns['js_legacy'].search(html_lower)),
            
            'html2_doctype': bool(self.regex_patterns['html2'].search(html_lower)),
            'html32_doctype': bool(self.regex_patterns['html32'].search(html_lower)),
            'html4.01': bool(self.regex_patterns['html4.01'].search(html_lower)),
            'html4.01_any': bool(self.regex_patterns['html4.01_any'].search(html_lower)),
            'xhtml1_any': bool(self.regex_patterns['xhtml1_any'].search(html_lower)),
            'html5_doctype': bool(self.regex_patterns['html5_doctype'].search(html_lower)), 
            'html5_charset': bool(self.regex_patterns['html5_charset'].search(html_lower)),
            'bootstrap_css' : bool(self.regex_patterns['bootstrap'].search(payload[:5000].lower())),

            # Generator signals
            'gen_1997_1999' : bool(self.regex_patterns['gen_1997_1999'].search(html_lower)),
            'gen_2000_2002' : bool(self.regex_patterns['gen_2000_2002'].search(html_lower)),
            'gen_2003_2006' : bool(self.regex_patterns['gen_2003_2006'].search(html_lower)),
            'gen_2007_2010' : bool(self.regex_patterns['gen_2007_2010'].search(html_lower)),
            'gen_post_2010' : bool(self.regex_patterns['gen_post_2010'].search(html_lower))
        }

        signals['no_match'] = not any ([
            v for k, v in signals.items() if k != 'no_match' and k != 'period_bucket'
        ])

        signals['period_bucket'] = self.assign_bucket(signals)
        return signals
    
    def assign_bucket(self, signals: dict) -> str:
        """ Waterfall assignment of period buckets.
        If a target matches the signal of the latest bucket, it is 'caught' in there.
        Other targets continue down the filter until they found a string match.
        If not, they're saved in the unknown bucket."""

        # Checked from newest to oldest to prevent false older classifications
        if signals['bootstrap_css'] or signals['gen_post_2010']:         
            return 'post_2010'
        elif signals['html5_doctype'] or signals['html5_charset'] or signals['gen_2007_2010']: 
            return '2007_2010'
        elif signals['gen_2003_2006']:                                   
            return '2003_2006'
        elif (signals['xhtml1_any']) or signals['gen_2000_2002'] or signals['html4.01'] or signals['html4.01_any']:        
            return '2000_2002'
        elif signals['legacy_dom'] or signals['html2_doctype'] or signals['js_legacy'] or signals['html32_doctype'] or signals['gen_1997_1999']:          
            return '1997_1999'
        else:                                                            
            return 'unknown'

class LabelingProcessor:
    """Manages the streaming label pipeline over a JSONL file."""

    def __init__(self, manual_annotations_path: str = None, extractor: SignalExtractor = None):
        self.extractor = extractor or SignalExtractor()
        self.bucket_counts = Counter()
        self.manual_map = {}

        # Manual annotations if provided
        if manual_annotations_path:
            p = Path(manual_annotations_path)
            if p.exists():
                with open(p, 'r', encoding='utf-8') as f:
                    manual_data = json.load(f)
                    for bucket, urls in manual_data.items():
                        for url in urls:
                            self.manual_map[url] = bucket

    def process(self, db_path: str, output_train: str, output_infer: str):
        # Open a connection to the SQLite database for writing
        self.db_path = db_path
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        with open(output_train, 'w', encoding='utf-8') as f_train, \
             open(output_infer, 'w', encoding='utf-8') as f_infer:
            
            for line_num, record in enumerate(get_ml_dataset(db_path, target_types='text/html')):
                self.process_line(record, f_train, f_infer, cursor)
        
        conn.commit()
        conn.close()
        
    def process_line(self, record, f_train, f_infer, cursor):
        domain = record.get('seed_url', '')
        signals = self.extractor.extract(record.get('payload', ''))

        # Check for manual overrides from JSON mapping
        if domain in self.manual_map:
            signals['period_bucket'] = self.manual_map[domain]

        bucket = signals['period_bucket']
        self.bucket_counts[signals['period_bucket']] += 1

        # Updating the SQLITE database with the ground truth labels
        if bucket != 'unknown':
            cursor.execute('''
                UPDATE websites 
                SET period = ? 
                WHERE seed_url = ?
            ''', (bucket, domain))
        
        # Prepare the JSON string
        output_data = json.dumps({
            'url': domain, 
            'payload': record.get('payload', ''), 
            'period_bucket': bucket
        }) + '\n'

        if bucket in ['unknown', 'post_2010']:
            f_infer.write(output_data)
        else:
            f_train.write(output_data)
    
    def report(self):
        print("Distribution of websites per bucket")
        for bucket, count in self.bucket_counts.items():
            print(f" {bucket}: {count}")

# --------------------------------
# Execution logic for standalone mode only

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label SQLite records and split into train/inference JSONL files.')
    parser.add_argument('--db_path', default='websites.db', help='Path to input SQLite database')
    parser.add_argument('--manual_json', default='metadata/manual_annotations.json', help='Path to manual annotations JSON')
    
    parser.add_argument('--output_train', required=True, help='Path to output JSONL for training')
    parser.add_argument('--output_infer', required=True, help='Path to output JSONL for inference')
    args = parser.parse_args()

    processor = LabelingProcessor(manual_annotations_path=args.manual_json)
    processor.process(args.db_path, args.output_train, args.output_infer)
    
    print(f"\nDatasets successfully generated:")
    print(f"-> Training data: {args.output_train}")
    print(f"-> Inference data: {args.output_infer}")
    processor.report()

    # py script/labeling_function_v8.py --db_path websites.db --manual_json metadata/manual_annotations.json --output_full metadata/labeled_outputv7.jsonl
    # python script/labeling_function_v8.py --db_path websites.db --manual_json metadata/manual_annotations.json --output_train training/train_solr_test.jsonl --output_infer inference/infer_solr_test.jsonl