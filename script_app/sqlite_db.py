import sqlite3

def init_db(db_path):
    """Initialized the SQLite database and creates the payloads table if it doesn't exist.
        The payloads table has the following columns:
        - seed_url: The domain or seed URL associated with the payload (e.g., 'www.example.com')
        - year: The annotated year from the index (e.g., 2005)
        - content_type: The MIME type of the payload (e.g., 'text/html')
        - payload: The actual content extracted from the WARC record (e.g., HTML source code)
        
        The combination of the seed_url and content_type helps to merge the entire website into one single row per doamin so
        downstream scripts can process the whole site at once, without having to worry about multiple records for the same domain.
        """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS websites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seed_url TEXT,
            url TEXT,
            year INTEGER,
            content_type TEXT,
            payload TEXT,
            period TEXT,
            confidence REAL,
            warc_filename TEXT,
            warc_record_id TEXT
        )
    ''')
    conn.commit()
    conn.close()


def get_ml_dataset(db_path, target_types=None):
    """Yields rows from the SQLite database as dictionaries for a specific content type.
    By default, it targets text/html content, but can be adjusted to include CSS and JavaScript if needed."""
    
    conn = sqlite3.connect(db_path)
    # Turns the tuple rows into dictionary-like objects
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if target_types is None:
        query = '''
            SELECT
                seed_url,
                GROUP_CONCAT(payload, CHAR(10) || CHAR(10)) as payload
                FROM websites
                GROUP BY seed_url
        '''
        cursor.execute(query)

    else:
        if isinstance(target_types, str):
            target_types = [target_types]

        placeholders = ','.join(['?'] * len(target_types))

        query = f'''
            SELECT
                seed_url,
                GROUP_CONCAT(payload, CHAR(10) || CHAR(10)) as payload
            FROM websites
            WHERE content_type IN ({placeholders})
            GROUP BY seed_url
        '''
        cursor.execute(query, target_types)    

    for row in cursor:
        yield dict(row)  # Convert sqlite3.Row to a regular dictionary
    
    conn.close()

def get_records(db_path, id=None, seed_url=None, url=None, year=None, 
                content_type=None, period=None, warc_filename=None, warc_record_id=None):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM websites"
    
    conditions = []
    params = []

    filters = {
        'id': id,
        'seed_url': seed_url,
        'url': url,
        'year': year,
        'content_type': content_type,
        'period': period,
        'warc_filename': warc_filename,
        'warc_record_id': warc_record_id
    }

    for column, value in filters.items():
        if value is not None:
            if column == 'content_type':
                conditions.append("content_type LIKE ?")
                params.append(f"{value}%")
            else:
                conditions.append(f"{column} = ?")
                params.append(value)
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    # Dynamic built query
    cursor.execute(query, tuple(params))

    for row in cursor:
        yield dict(row)
    
    conn.close()

def check_total_records(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM websites")
    print("Total individual files extracted:", cursor.fetchone()[0])
    conn.close()

def get_all_seed_urls(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT seed_url FROM websites")

    urls = [row[0] for row in cursor.fetchall()]

    conn.close()
    
    return urls

import sqlite3

def find_spread_domains(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # This query finds seed_urls associated with more than one unique warc_file
    query = """
        SELECT seed_url, COUNT(DISTINCT warc_filename) as warc_count
        FROM websites
        GROUP BY seed_url
        HAVING warc_count > 1
        ORDER BY warc_count DESC
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print("No domains found spanning multiple WARC files.")
    else:
        print(f"Found {len(results)} domains spread across multiple WARC files:\n")
        for domain, count in results:
            print(f" -> {domain}: found in {count} different files")
