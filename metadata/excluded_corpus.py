import pandas as pd
filter = pd.read_csv('metadata/xs4all_warcs_request.csv', sep=',', dtype={"ID": "Int64", "year": "Int64", "url": "string"})
original = pd.read_csv('metadata/total_seeds.csv', sep=';', dtype={"ID": "Int64", "year": "Int64", "url": "string"}, encoding='latin-1')

original['url'] = original['url'].str.strip('www.')
print(original.head())
