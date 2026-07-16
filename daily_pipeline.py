import os
import re
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

from data_processing import (
    clean_dataframe,
    join_insp_sitrep_csvs,
    join_flowminder_csvs,
    compute_osrm_nearest_active,
    clean_and_merge_flowminder,
    merge_worldpop
)

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL)
else:
    engine = create_engine("sqlite:///viralwatch.db")


# --- Local Paths to checked out repository files ---
DATA_REPO_DIR = Path("BDBV2026-Data")
BUILD_LONG_DIR = DATA_REPO_DIR / "build" / "long"
BUILD_DIR = DATA_REPO_DIR / "build"

# Configuration Paths
OSRM_PATH = BUILD_LONG_DIR / "osrm__travel_time.csv"
ALIASES_PATH = DATA_REPO_DIR / "data" / "aliases.csv"
WP_COUNT_PATH = BUILD_LONG_DIR / "worldpop__pop_count.csv"
WP_DENSITY_PATH = BUILD_LONG_DIR / "worldpop__pop_density.csv"


def clean_column_name(col):
    c = col.lower().strip()
    c = re.sub(r'[^a-z0-9_]', '_', c)
    c = re.sub(r'_+', '_', c)
    return c.strip('_')


def run_pipeline():
    # Setup the local Output Directory
    workspace_root = Path(".").resolve()
    output_dir = workspace_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean DB Schema on Startup (if using a persistent remote DB)
    if DATABASE_URL and "sqlite" not in DATABASE_URL:
        try:
            with engine.begin() as conn:
                conn.execute(text("DROP SCHEMA public CASCADE;"))
                conn.execute(text("CREATE SCHEMA public;"))
                conn.execute(text("GRANT ALL ON SCHEMA public TO public;"))
        except Exception:
            pass

    # --- 1. Merge and Clean INSP Sitreps dynamically ---
    print("⏳ Merging individual INSP Sitrep CSVs...")
    try:
        merged_sitrep_path = output_dir / "insp_sitrep_merged.csv"
        raw_sitrep = join_insp_sitrep_csvs(BUILD_LONG_DIR, merged_sitrep_path)
        
        # Clean Datatypes / Replace "ND"
        for col in raw_sitrep.columns:
            if col not in ["nom", "date"]:
                raw_sitrep[col] = raw_sitrep[col].replace("ND", pd.NA)
                raw_sitrep[col] = pd.to_numeric(raw_sitrep[col], errors="coerce")

        # Split and save clean zones
        zone_rows = raw_sitrep[raw_sitrep["nom"].notna()].copy()
        zone_rows = zone_rows[zone_rows["date"].notna()].copy()
        zone_rows.to_csv(output_dir / "insp_sitrep_zone_level_clean.csv", index=False)

        # Build Training Window (2026-05-14 to 2026-05-29)
        training_df = zone_rows[(zone_rows["date"] >= "2026-05-14") & (zone_rows["date"] <= "2026-05-29")].copy()
        training_path = output_dir / "insp_sitrep_training_window.csv"
        training_df.to_csv(training_path, index=False)
        
        # Upload clean training set
        training_db = clean_dataframe(training_df)
        training_db.columns = [clean_column_name(col) for col in training_db.columns]
        training_db.to_sql("insp_sitrep_training_window", engine, if_exists="replace", index=False)
        print("✅ INSP Sitrep Complete!")
    except Exception as e:
        print(f"❌ INSP Sitrep failed: {e}")

    # --- 2. Calculate OSRM Nearest Active Matrix ---
    print("⏳ Calculating nearest active-case zones using OSRM travel matrices...")
    try:
        sitrep_path = output_dir / "insp_sitrep_training_window.csv"
        out_osrm_path = output_dir / "osrm_nearest_active_feature.csv"
        
        if sitrep_path.exists():
            osrm_df = compute_osrm_nearest_active(OSRM_PATH, ALIASES_PATH, sitrep_path, out_osrm_path)
            
            osrm_db = clean_dataframe(osrm_df)
            osrm_db.columns = [clean_column_name(col) for col in osrm_db.columns]
            osrm_db.to_sql("osrm_nearest_active_feature", engine, if_exists="replace", index=False)
            print("✅ OSRM calculation Complete!")
    except Exception as e:
        print(f"❌ OSRM Matrix logic failed: {e}")

    # --- 3. Dynamic Merge and Clean Flowminder ---
    print("⏳ Dynamically compiling and cleaning Flowminder datasets...")
    try:
        merged_flowminder_path = output_dir / "flowminder_merged.csv"
        
        # Build the dynamic merged dataset from local individual files
        join_flowminder_csvs(BUILD_LONG_DIR, merged_flowminder_path)
        
        # Clean and reduce to keeping columns
        flow_df = clean_and_merge_flowminder(merged_flowminder_path, output_dir / "flowminder_clean.csv")
        
        flow_db = clean_dataframe(flow_df)
        flow_db.columns = [clean_column_name(col) for col in flow_db.columns]
        flow_db.to_sql("flowminder_clean", engine, if_exists="replace", index=False)
        print("✅ Flowminder dynamic compiler Complete!")
    except Exception as e:
        print(f"❌ Flowminder dynamic merge failed: {e}")

    # --- 4. Merge WorldPop ---
    print("⏳ Compiling WorldPop count and density parameters...")
    try:
        wp_df = merge_worldpop(WP_COUNT_PATH, WP_DENSITY_PATH, output_dir / "worldpop_merged.csv")
        
        wp_db = clean_dataframe(wp_df)
        wp_db.columns = [clean_column_name(col) for col in wp_db.columns]
        wp_db.to_sql("worldpop_merged", engine, if_exists="replace", index=False)
        print("✅ WorldPop Merging Complete!")
    except Exception as e:
        print(f"❌ Worldpop processing failed: {e}")


if __name__ == "__main__":
    run_pipeline()
