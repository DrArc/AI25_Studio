# sql/create_sql_db.py

import sqlite3
import pandas as pd
import os
import sys

# Ensure local import path for Cursor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.column_cleaner import fully_standardize_dataframe

# File paths
CSV_PATH = "sql/Ecoform_Dataset_v1.csv"
DB_PATH = "sql/comfort-database.db"

# Load dataset
df = pd.read_csv(CSV_PATH)

# Apply full universal cleaner (both cleaning + renaming)
df = fully_standardize_dataframe(df)
# Add cleaned columns for robust matching
if 'apartment_type_string' in df.columns:
    df['apartment_type_string_clean'] = df['apartment_type_string'].str.lower().str.replace(' ', '').str.replace('-', '')
if 'zone_string' in df.columns:
    df['zone_string_clean'] = df['zone_string'].str.lower().str.replace(' ', '').str.replace('-', '')

# Connect to SQLite
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Drop existing table if exists
cursor.execute("DROP TABLE IF EXISTS comfort_lookup")

# Dynamically define SQL table schema
column_defs = []
for col, dtype in zip(df.columns, df.dtypes):
    if pd.api.types.is_integer_dtype(dtype):
        sql_type = 'INTEGER'
    elif pd.api.types.is_float_dtype(dtype):
        sql_type = 'REAL'
    else:
        sql_type = 'TEXT'
    column_defs.append(f'"{col}" {sql_type}')

# Create SQL table
create_sql = f"CREATE TABLE comfort_lookup ({', '.join(column_defs)})"
cursor.execute(create_sql)

# Insert data into table
df.to_sql("comfort_lookup", conn, if_exists="append", index=False)
conn.close()

print("✅ Ecoform dataset fully cleaned & inserted into SQL")
