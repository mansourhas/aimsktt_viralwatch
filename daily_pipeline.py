import os
import glob
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# 1. Fetch Aiven Connection String from Environment
DATABASE_URL = os.environ.get("DATABASE_URL")

# Set up SQL connection (Fallback to SQLite inside data_test/ if no cloud URL is configured)
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL)
    print("🔌 Connected successfully to your Cloud Aiven PostgreSQL database!")
else:
    engine = create_engine("sqlite:///data_test/viralwatch.db")
    print("📁 DATABASE_URL not found. Saving locally to data_test/viralwatch.db.")

def clean_and_sync():
    print("🧹 Starting data cleaning and transformation...")
    
    # --- Target: Only INSP SitRep Processed Files ---
    search_path = os.path.join("data_test", "*insp_sitrep*.csv")
    sitrep_files = glob.glob(search_path)
    
    if sitrep_files:
        for file_path in sitrep_files:
            # Dynamically derive table name from filename (e.g., insp_sitrep__new_confirmed_cases__daily -> insp_sitrep_new_confirmed_cases_daily)
            filename = os.path.basename(file_path).replace(".csv", "").lower()
            table_name = filename.replace("__", "_")
            
            print(f"📦 Processing SitRep file: {file_path} -> Database Table: '{table_name}'")
            
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower().str.strip()
            
            # Clean health zone names if the column is present
            zone_cols = [c for c in df.columns if 'zone' in c or 'nom' in c]
            if zone_cols:
                df[zone_cols[0]] = df[zone_cols[0]].astype(str).str.strip().str.title()
            
            # Standardize dates if present
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                if zone_cols:
                    df = df.sort_values(by=[zone_cols[0], 'date'])

            # DB INSERTION: Write the sitrep data to its dynamic table
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"✔ '{table_name}' table written/updated in the database.")
    else:
        print("❌ Error: No CSV files matching '*insp_sitrep*.csv' found in data_test/")

    print("🎉 Ingestion & Database sync complete!")

if __name__ == "__main__":
    clean_and_sync()
