# format_interpreter.py

import pandas as pd
import re

# === 1️⃣ Column Cleaner ===
def clean_columns(df):
    """
    Cleans column names to lowercase, removes spaces and special characters.
    Always run this after loading any dataset.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace(":", "")
    )
    return df

# === 2️⃣ Input Key Mapper ===
def standardize_input(user_input: dict):
    """
    Maps various user input keys into standard dataset keys.
    Ensures LLM output or structured input get converted to dataset-compatible keys.
    """

    key_mapping = {
        'zone': 'zone_string',
        'apartment_type': 'apartment_type_string',
        'apartmenttype': 'apartment_type_string',
        'apt_type': 'apartment_type_string',
        'apttype': 'apartment_type_string',
        'element': 'element_materials_string',
        'element_materials': 'element_materials_string',
        'floor_height': 'floor_height_m',
        'floor_level': 'floor_level',
        'floor': 'floor_level',
        'laeq': 'laeq_db',
        'l(a)eq': 'laeq_db',
        'spl': 'spl_db',
        'rt60': 'rt60_s',
        'absorption': 'absorption_coefficient_by_area_m',
        'surface_area': 'total_surface_sqm',
        'barrier_distance': 'barrier_distance_m',
        'barrier_height': 'barrier_height_m',
        'spl_after_barrier': 'spl_after_barrier_db',
        'spl_after_façade': 'spl_after_faã§ade_dampening_db',
        'comfort_index': 'comfort_index_float',
        'wall_material': 'wall_material',
        'window_material': 'window_material',
        'activity': 'activity',
        'time_period': 'time_period'
    }

    standardized = {}
    for k, v in user_input.items():
        # Clean the key
        cleaned_key = k.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        # Get the standardized key
        std_key = key_mapping.get(cleaned_key, cleaned_key)
        # Clean the value if it's a string
        if isinstance(v, str):
            v = v.strip()
        standardized[std_key] = v
    
    # Ensure we have the critical keys
    if 'apartment_type' not in standardized and 'Apartment_Type' in user_input:
        standardized['apartment_type'] = user_input['Apartment_Type']
    if 'zone' not in standardized and 'Zone' in user_input:
        standardized['zone'] = user_input['Zone']
    
    # Normalize apartment type to match database format
    if 'apartment_type_string' in standardized:
        apt_type = standardized['apartment_type_string']
        if isinstance(apt_type, str):
            # Normalize apartment type variations
            apt_type_upper = apt_type.upper()
            if '1BED' in apt_type_upper or '1B' in apt_type_upper:
                standardized['apartment_type_string'] = '1Bed'
            elif '2BED' in apt_type_upper or '2B' in apt_type_upper:
                standardized['apartment_type_string'] = '2Bed'
            elif '3BED' in apt_type_upper or '3B' in apt_type_upper:
                standardized['apartment_type_string'] = '3Bed'
            # If it's already in correct format, keep it
            elif apt_type in ['1Bed', '2Bed', '3Bed']:
                standardized['apartment_type_string'] = apt_type
            else:
                # Default to 2Bed if we can't determine
                standardized['apartment_type_string'] = '2Bed'
    
    return standardized

# === Optional: Small helper ===
def preview_columns(file_path):
    df = pd.read_csv(file_path)
    df = clean_columns(df)
    print(df.columns.tolist())
