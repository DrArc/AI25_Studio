# sql_calls.py

import sqlite3
import pandas as pd
import os
import sys

# ✅ Allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.format_interpreter import standardize_input
from scripts.core.recommend_recompute import recommend_recompute

# === Path to SQLite DB ===
DB_PATH = "sql/comfort-database.db"

# === Main SQL Call Function ===
def query_or_recommend(user_input):
    """
    Try to retrieve comfort score from SQL based on zone/apartment.
    If not found, fallback to ML + recommendation pipeline.
    """

    # ✅ Standardize input keys first
    user_input = standardize_input(user_input)
    print("🔍 Standardized input:", user_input)

    abs_db_path = os.path.abspath(DB_PATH)
    print(f"🔍 Using database file: {abs_db_path}")
    conn = sqlite3.connect(abs_db_path)

    # Build WHERE clause — only use what SQL can easily match
    conditions = []
    params = []

    # Handle apartment type with more flexibility
    if "apartment_type_string" in user_input:
        apt_type_clean = user_input["apartment_type_string"].lower().replace(' ', '').replace('-', '')
        conditions.append("(apartment_type_string_clean = ?)")
        params.append(apt_type_clean)
        print(f"🔍 Added apartment type condition (clean): {apt_type_clean}")

    if "zone_string" in user_input:
        zone_clean = user_input["zone_string"].lower().replace(' ', '').replace('-', '')
        conditions.append("(zone_string_clean = ?)")
        params.append(zone_clean)
        print(f"🔍 Added zone condition (clean): {zone_clean}")

    # Handle element materials with more flexibility
    if "element_materials_string" in user_input:
        mat_clean = user_input["element_materials_string"].lower().replace(' ', '').replace('-', '')
        if 'element_materials_string_clean' in [col[1] for col in conn.execute("PRAGMA table_info(comfort_lookup)")]:
            conditions.append("(element_materials_string_clean = ?)")
            params.append(mat_clean)
            print(f"🔍 Added element materials condition (clean): {mat_clean}")

    # If no minimal keys available, skip SQL entirely
    if not conditions:
        print("⚠️ Not enough keys to query SQL — switching to model...")
        return recommend_recompute(user_input)

    sql_query = f"""
    SELECT 
        comfort_index,
        apartment_type,
        zone,
        laeq as laeq_db,
        rt60 as rt60_s,
        spl as spl_db,
        absorption_coefficient_by_area as absorption_coefficient,
        total_surface as surface_area,
        floor_height_m,
        n._of_sound_sources_int,
        average_sound_source_distance_m,
        barrier_distance_m,
        barrier_height_m
    FROM comfort_lookup
    WHERE {' AND '.join(conditions)}
    ORDER BY comfort_index DESC
    LIMIT 1;
    """
    print("📝 SQL Query:", sql_query)
    print("📝 SQL Parameters:", params)

    try:
        result = pd.read_sql_query(sql_query, conn, params=params)
        print("📦 SQL Result DataFrame:")
        print(result)
        if not result.empty:
            print("✅ Match found in SQL database.")
            # Build features from SQL result
            row = result.iloc[0]
            features = {
                'zone_string': user_input.get('zone_string'),
                'apartment_type_string': user_input.get('apartment_type_string'),
                'element_materials_string': row.get('element_materials_string', user_input.get('element_materials_string', 'Unknown')),
                'floor_height_m': row.get('floor_height_m', user_input.get('floor_height_m', 3.0)),
                'laeq_db': row.get('laeq_db', 0.0),
                'total_surface_sqm': row.get('surface_area', 0.0),
                'absorption_coefficient_by_area_m': row.get('absorption_coefficient', 0.0),
                'rt60_s': row.get('rt60_s', 0.0),
                'n._of_sound_sources_int': row.get('n._of_sound_sources_int', 0),
                'average_sound_source_distance_m': row.get('average_sound_source_distance_m', 0.0),
                'spl_db': row.get('spl_db', 0.0),
                'barrier_distance_m': row.get('barrier_distance_m', 0.0),
                'barrier_height_m': row.get('barrier_height_m', 0.0)
            }
            # Use ML model to predict comfort score
            from scripts.recommend_recompute import model
            import pandas as pd
            categorical = [
                'zone_string',
                'apartment_type_string',
                'element_materials_string'
            ]
            numeric = [
                'floor_height_m',
                'laeq_db',
                'total_surface_sqm',
                'absorption_coefficient_by_area_m',
                'rt60_s',
                'n._of_sound_sources_int',
                'average_sound_source_distance_m',
                'spl_db',
                'barrier_distance_m',
                'barrier_height_m'
            ]
            df_features = pd.DataFrame([features])[categorical + numeric]
            for col in categorical:
                df_features[col] = df_features[col].astype(str)
            for col in numeric:
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)
            comfort_score = round(float(model.predict(df_features)[0]), 3)
            print(f"✅ ML Predicted comfort score: {comfort_score}")
            # Build metrics for output
            metrics = {
                "LAeq (dB)": float(row.get('laeq_db', 0.0)),
                "RT60 (s)": float(row.get('rt60_s', 0.0)),
                "SPL (dB)": float(row.get('spl_db', 0.0)),
                "Absorption Coefficient": float(row.get('absorption_coefficient', 0.0)),
                "Surface Area (m²)": float(row.get('surface_area', 0.0))
            }
            activity = user_input.get('activity', 'Living')
            compliance_info = {
                "status": "compliant",
                "reason": "Matched from database",
                "metrics": metrics,
                "thresholds": {
                    "LAeq (dB)": 35 if activity == "Sleeping" else 40,
                    "RT60 (s)": 0.5 if activity == "Sleeping" else 0.6
                }
            }
            return {
                "comfort_score": comfort_score,
                "source": "SQL+ML",
                "compliance": compliance_info,
                "recommendations": {},
                "improved_score": None
            }
        else:
            print("⚠️ No match found in SQL database")
            print("🔄 Switching to model + compliance + recommendation...")
            return recommend_recompute(user_input)
    except Exception as e:
        print("⚠️ SQL lookup failed:", str(e))
        print("🔄 Switching to model + compliance + recommendation...")
        return recommend_recompute(user_input)
    finally:
        conn.close()
