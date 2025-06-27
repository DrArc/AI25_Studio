# utils/infer_from_inputs.py

import pandas as pd
import os
import sys
import numpy as np
from utils.reference_data import material_directory

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.column_cleaner import fully_standardize_dataframe

DATA_PATH = "sql/Ecoform_Dataset_v1.csv"
df = pd.read_csv(DATA_PATH)
df = fully_standardize_dataframe(df)
print('DEBUG: Standardized DataFrame columns:', df.columns.tolist())

model_features = [
    'zone_string',
    'apartment_type_string',
    'floor_height_m',
    'laeq_db',
    'day/nightstring',
    'total_surface_sqm',
    'element_materials_string',
    'absorption_coefficient_by_area_m',
    'rt60_s',
    'n._of_sound_sources_int',
    'average_sound_source_distance_m',
    'spl_db',
    'barrier_distance_m',
    'barrier_height_m',
    'spl_after_barrier_db',
    'spl_after_facade_dampening_db',
    'comfort_index_float',
    'absorption_norm',
    'rt60_norm',
    'spl_norm',
    'comfortindex_v2',
    'spl_per_surface',
    'unnamed_20',
    'unnamed_21',
    'unnamed_22',
    'unnamed_23',
    'unnamed_24',
    'unnamed_25',
    'unnamed_26'
]

def infer_features(apartment_type, zone, element=None, element_material=None, floor_level=None, wall_material=None, window_material=None, time_period=None):
    """
    Infer missing features based on available input parameters.
    Returns a complete feature vector for model prediction.
    """
    print("[DEBUG] infer_features called with:", apartment_type, zone, "time_period:", time_period)
    df = pd.read_csv(DATA_PATH)
    df = fully_standardize_dataframe(df)
    # Add cleaned columns for robust matching
    df['apartment_type_string_clean'] = df['apartment_type_string'].str.lower().str.replace(' ', '').str.replace('-', '')
    df['zone_string_clean'] = df['zone_string'].str.lower().str.replace(' ', '').str.replace('-', '')
    apartment_type_clean = str(apartment_type).lower().replace(' ', '').replace('-', '')
    zone_clean = str(zone).lower().replace(' ', '').replace('-', '')
    print('[DEBUG] Unique apartment_type_string_clean:', df['apartment_type_string_clean'].unique())
    print('[DEBUG] Unique zone_string_clean:', df['zone_string_clean'].unique())
    
    # Determine day/night string based on time_period
    day_night_string = 'day'  # Default
    if time_period:
        if time_period.lower() in ['night', 'evening', 'late']:
            day_night_string = 'night'
        else:
            day_night_string = 'day'
    
    # Use original case for model compatibility
    features = {
        'zone_string': zone,
        'apartment_type_string': apartment_type,
        'floor_height_m': 3.0,
        'laeq_db': 0.0,
        'day/nightstring': day_night_string,  # Use determined value
        'total_surface_sqm': 0.0,
        'element_materials_string': element if element else 'Unknown',
        'absorption_coefficient_by_area_m': 0.0,
        'rt60_s': 0.0,
        'n._of_sound_sources_int': 0,
        'average_sound_source_distance_m': 0.0,
        'spl_db': 0.0,
        'barrier_distance_m': 0.0,
        'barrier_height_m': 0.0,
        'spl_after_barrier_db': 0.0,
        'spl_after_facade_dampening_db': 0.0,
        'comfort_index_float': 0.0,
        'absorption_norm': 0.0,
        'rt60_norm': 0.0,
        'spl_norm': 0.0,
        'comfortindex_v2': 0.0,
        'spl_per_surface': 0.0,
        'unnamed_20': 0.0,
        'unnamed_21': 0.0,
        'unnamed_22': 0.0,
        'unnamed_23': 0.0,
        'unnamed_24': 0.0,
        'unnamed_25': 0.0,
        'unnamed_26': 0.0
    }
    # Try exact match first (robust)
    match = df[
        (df['apartment_type_string_clean'] == apartment_type_clean) &
        (df['zone_string_clean'] == zone_clean)
    ]
    print(f"[DEBUG] Exact match rows: {len(match)}")
    tier = None
    if not match.empty:
        tier = "Tier 1: Exact apartment + zone match"
        numeric_features = match.select_dtypes(include=['float64', 'int64']).columns
        for feature in numeric_features:
            if feature in model_features:
                features[feature] = float(match[feature].mean())
    else:
        # Try partial match for apartment type (robust)
        match = df[
            (df['apartment_type_string_clean'].str.contains(apartment_type_clean)) &
            (df['zone_string_clean'].str.contains(zone_clean))
        ]
        print(f"[DEBUG] Partial match rows: {len(match)}")
        if not match.empty:
            tier = "Tier 2: Partial apartment + zone match"
            numeric_features = match.select_dtypes(include=['float64', 'int64']).columns
            for feature in numeric_features:
                if feature in model_features:
                    features[feature] = float(match[feature].mean())
        else:
            match = df[df['apartment_type_string_clean'].str.contains(apartment_type_clean)]
            print(f"[DEBUG] Apartment only match rows: {len(match)}")
            if not match.empty:
                tier = "Tier 3: Apartment only match"
                numeric_features = match.select_dtypes(include=['float64', 'int64']).columns
                for feature in numeric_features:
                    if feature in model_features:
                        features[feature] = float(match[feature].mean())
            else:
                tier = "Tier 4: Global mean fallback"
                print("Using global mean fallback!")
                
                # Use hardcoded apartment dimensions
                apartment_dimensions = {
                    "1Bed": {"volume": 58, "area": 19.33, "height": 3.0},
                    "2Bed": {"volume": 81, "area": 27.00, "height": 3.0},
                    "3Bed": {"volume": 108, "area": 36.00, "height": 3.0},
                }
                
                # Get dimensions for the apartment type
                dims = apartment_dimensions.get(apartment_type, apartment_dimensions["2Bed"])
                volume = dims["volume"]
                surface_area = dims["area"] * 2 + (dims["area"] / dims["height"]) * 2 * dims["height"]  # Floor + ceiling + walls
                
                absorptions = []
                if wall_material:
                    for coef, name in material_directory['wall']:
                        if name.lower() == wall_material.lower():
                            absorptions.append(coef)
                            break
                if window_material:
                    for coef, name in material_directory['window']:
                        if name.lower() == window_material.lower():
                            absorptions.append(coef)
                            break
                avg_abs = sum(absorptions) / len(absorptions) if absorptions else 0.1
                A = avg_abs * surface_area
                if A > 0:
                    rt60 = 0.161 * volume / A
                else:
                    rt60 = 0.0
                SPL_source = 85
                if A > 0:
                    spl = SPL_source - 10 * np.log10(A)
                else:
                    spl = SPL_source
                features = {
                    'zone_string': zone,
                    'apartment_type_string': apartment_type,
                    'floor_height_m': round(float(floor_level) * 3.0, 2) if floor_level is not None else 3.0,
                    'laeq_db': 45,
                    'day/nightstring': day_night_string,
                    'total_surface_sqm': surface_area,
                    'element_materials_string': element if element else 'Unknown',
                    'absorption_coefficient_by_area_m': avg_abs,
                    'rt60_s': rt60,
                    'n._of_sound_sources_int': 1,
                    'average_sound_source_distance_m': 2.0,
                    'spl_db': spl,
                    'barrier_distance_m': 0.0,
                    'barrier_height_m': 0.0,
                    'spl_after_barrier_db': spl * 0.9,  # Assume 10% reduction after barrier
                    'spl_after_facade_dampening_db': spl * 0.85,  # Assume 15% reduction after facade
                    'comfort_index_float': 0.7,  # Default comfort index
                    'absorption_norm': avg_abs / 0.5,  # Normalize absorption coefficient
                    'rt60_norm': rt60 / 0.5,  # Normalize RT60
                    'spl_norm': spl / 60.0,  # Normalize SPL
                    'comfortindex_v2': 0.7,  # Default comfort index v2
                    'spl_per_surface': spl / surface_area,
                    'unnamed_20': 0.0,
                    'unnamed_21': 0.0,
                    'unnamed_22': 0.0,
                    'unnamed_23': 0.0,
                    'unnamed_24': 0.0,
                    'unnamed_25': 0.0,
                    'unnamed_26': 0.0
                }
    
    # === CRITICAL: Always override floor_height_m with user's floor_level input ===
    if floor_level is not None:
        features['floor_height_m'] = round(float(floor_level) * 3.0, 2)
        print(f"[DEBUG] Overriding floor_height_m with user input: floor_level={floor_level} -> floor_height_m={features['floor_height_m']}")
    
    for feature in model_features:
        if feature not in features or pd.isna(features[feature]):
            features[feature] = 0.0
    print(f"[DEBUG] Using tier: {tier}")
    print("[DEBUG] Features returned:", features)
    return features, tier
