# recommend_recompute.py

import sys
import os
import json
import joblib
import pandas as pd
import numpy as np

# === Add root path for package imports ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.infer_from_inputs import infer_features
from utils.reference_data import (
    DAY_RANGES, NIGHT_RANGES, material_directory,
    RT60_target, RT60_max_dev
)
from utils.format_interpreter import standardize_input

# === Paths ===
MODEL_PATH = "model/ecoform_xgb_comfort_model1.pkl"
GUIDANCE_JSON = "knowledge/compliance_guidance.json"
COMPLIANCE_JSON = "knowledge/compliance_thresholds_extended.json"

# === Load model once ===
model = joblib.load(MODEL_PATH)

# === Load compliance JSON (WHO/ISO fallback) ===
with open(COMPLIANCE_JSON) as f:
    loaded = json.load(f)
    compliance_thresholds = {entry["use"]: entry for entry in loaded}

# === Activity comfort thresholds ===
activity_thresholds = {
    "Sleeping": 0.85, "Working": 0.75, "Learning": 0.80, "Living": 0.70,
    "Healing": 0.80, "Co-working": 0.75, "Exercise": 0.60, "Dining": 0.65
}

# === Helper: LAeq compliance check ===
def check_la_eq_compliance(zone, laeq, period='day'):
    db_range = DAY_RANGES.get(zone) if period.lower() == 'day' else NIGHT_RANGES.get(zone)
    if not db_range or laeq is None:
        return False, db_range
    min_db, max_db = db_range
    return min_db <= laeq <= max_db, db_range

# === Helper: Best material upgrade ===
def recommend_best_material(material_type, current_abs):
    better = [mat for mat in material_directory[material_type] if mat[0] > current_abs]
    if better:
        return sorted(better, reverse=True)[0]
    return None

# === Main full pipeline ===
def recommend_recompute(user_input):
    """
    Compute comfort score using ML model and provide recommendations.
    """
    try:
        # === Clean input keys ===
        user_input = standardize_input(user_input)
        print("🔍 Standardized input for ML:", user_input)

        # Extract required parameters
        zone = user_input.get("zone_string")
        apartment_type = user_input.get("apartment_type_string")
        element_material = user_input.get("element_materials_string")
        wall_material = user_input.get("wall_material")
        window_material = user_input.get("window_material")
        floor_level = user_input.get("floor_level")
        activity = user_input.get("activity", "Living")
        period = user_input.get("time_period", "day")

        if not zone or not apartment_type:
            raise ValueError("Both zone_string and apartment_type_string are required")

        print(f"🔍 Running ML inference for: {apartment_type} in {zone}")
        
        # === Run inference for missing features ===
        features, tier = infer_features(
            apartment_type=apartment_type,
            zone=zone,
            element=element_material,
            element_material=wall_material,
            floor_level=floor_level,
            wall_material=wall_material,
            window_material=window_material,
            time_period=period
        )
        print(f"✅ Features inferred using {tier}")
        print("📦 Inferred features:", features)

        # === Model prediction ===
        try:
            # Create DataFrame with all required features
            df = pd.DataFrame([features])
            
            # Ensure exact column names and order as in training
            categorical = [
                'zone_string',
                'apartment_type_string',
                'day/nightstring',
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
            # Reorder columns to match training data
            df = df[categorical + numeric]
            # Ensure correct dtypes
            for col in categorical:
                df[col] = df[col].astype(str)
            for col in numeric:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            print("📦 DataFrame dtypes before prediction:", df.dtypes)
            print("📦 DataFrame values before prediction:", df.iloc[0].to_dict())
            print("📦 DataFrame before prediction:", df)
            # Make prediction
            comfort_score = round(float(model.predict(df)[0]), 3)
            print(f"✅ Predicted comfort score: {comfort_score}")
            print("DEBUG: Model prediction succeeded, building output...")
        except Exception as e:
            print(f"❌ Error in model prediction: {str(e)}")
            print("📦 DataFrame columns:", df.columns.tolist())
            print("📦 DataFrame shape:", df.shape)
            raise
        print("DEBUG: Proceeding to compliance and recommendations...")

        # === LAeq compliance ===
        laeq_value = features.get("laeq_db") or features.get("l(a)eq_db") or None
        laeq_ok, db_range = check_la_eq_compliance(zone, laeq_value, period)

        # === RT60 compliance ===
        rt60_value = features.get("rt60_s", None)
        rt60_ok = rt60_value is not None and RT60_target <= rt60_value <= RT60_max_dev

        # === Overall ISO compliance fallback ===
        iso_rule = compliance_thresholds.get(activity)
        iso_failures = []
        if iso_rule:
            if laeq_value and laeq_value > iso_rule.get("laeq_max", np.inf):
                iso_failures.append(f"LAeq > {iso_rule['laeq_max']} dB")
            if rt60_value and rt60_value > iso_rule.get("rt60_max", np.inf):
                iso_failures.append(f"RT60 > {iso_rule['rt60_max']} s")

        # === Comfort score compliance ===
        comfort_threshold = activity_thresholds.get(activity, 0.7)
        comfort_score_ok = comfort_score >= comfort_threshold
        
        # Only add comfort score failure if it's actually below threshold
        if not comfort_score_ok:
            iso_failures.append(f"Comfort score < {comfort_threshold}")

        # Determine overall compliance
        is_compliant = (laeq_ok and rt60_ok and not iso_failures and comfort_score_ok)
        
        # Create detailed compliance explanation
        compliance_details = []
        if laeq_ok:
            compliance_details.append(f"✅ LAeq ({laeq_value:.1f} dB) within zone range {db_range}")
        else:
            compliance_details.append(f"❌ LAeq ({laeq_value:.1f} dB) outside zone range {db_range}")
            
        if rt60_ok:
            compliance_details.append(f"✅ RT60 ({rt60_value:.2f}s) within acceptable range ({RT60_target}-{RT60_max_dev}s)")
        else:
            compliance_details.append(f"❌ RT60 ({rt60_value:.2f}s) outside acceptable range ({RT60_target}-{RT60_max_dev}s)")
            
        if comfort_score_ok:
            compliance_details.append(f"✅ Comfort score ({comfort_score:.3f}) meets threshold ({comfort_threshold})")
        else:
            compliance_details.append(f"❌ Comfort score ({comfort_score:.3f}) below threshold ({comfort_threshold})")

        # === Load guidance JSON for general recommendations ===
        with open(GUIDANCE_JSON) as f:
            guidance = json.load(f)

        recommendations = {}

        if not laeq_ok:
            laeq_str = f"{laeq_value:.2f}" if laeq_value is not None else "N/A"
            laeq_recs = guidance["LAeq_non_compliant"]["general_recommendations"]
            laeq_recs_str = "\n- " + "\n- ".join(laeq_recs) if isinstance(laeq_recs, list) else str(laeq_recs)
            recommendations["LAeq_zone"] = (
                f"Measured LAeq is {laeq_str} dB, which is outside the allowed range {db_range}.{laeq_recs_str}"
            )

        if not rt60_ok:
            rt60_str = f"{rt60_value:.2f}" if rt60_value is not None else "N/A"
            rt60_recs = guidance["RT60_non_compliant"]["general_recommendations"]
            rt60_recs_str = "\n- " + "\n- ".join(rt60_recs) if isinstance(rt60_recs, list) else str(rt60_recs)
            recommendations["RT60"] = (
                f"Measured RT60 is {rt60_str} s, which is outside the allowed range ({RT60_target}-{RT60_max_dev} s).{rt60_recs_str}"
            )

        if not comfort_score_ok:
            score_str = f"{comfort_score:.2f}" if comfort_score is not None else "N/A"
            recommendations["Comfort Score"] = (
                f"Comfort score is {score_str}, below the required threshold of {comfort_threshold} for {activity}. "
                "Consider improving both noise insulation and absorption."
            )

        if iso_failures:
            recommendations["ISO"] = (
                "ISO/WHO compliance failed: " + "; ".join(iso_failures)
            )

        # === Material substitution (absorption driven) ===
        current_wall = wall_material or features.get("wall_material", None)
        wall_abs = None
        print("Current wall material:", current_wall)
        print("Material directory wall names:", [name for coef, name in material_directory['wall']])
        for coef, name in material_directory['wall']:
            # Allow partial (substring) match, case-insensitive
            if name.lower() in (current_wall or '').lower():
                wall_abs = coef
                break
        print("Wall absorption coefficient:", wall_abs)
        best_wall = recommend_best_material('wall', wall_abs) if wall_abs else None

        # === Re-run model after material substitution ===
        improved_score = None
        if best_wall:
            try:
                features["wall_material"] = best_wall[1].lower()
                df_new = pd.DataFrame([features])
                improved_score = round(float(model.predict(df_new)[0]), 3)
                recommendations["Wall Upgrade"] = f"Try upgrading to: {best_wall[1]} (abs={best_wall[0]})"
            except Exception as e:
                print(f"❌ Error in improved score prediction: {str(e)}")
                improved_score = None

        # === Build final output ===
        # Ensure compliance and recommendations are always dicts
        # --- Add detailed metrics for ML fallback ---
        metrics = {
            "LAeq (dB)": features.get("laeq_db", None),
            "RT60 (s)": features.get("rt60_s", None),
            "SPL (dB)": features.get("spl_db", None),
            "Absorption Coefficient": features.get("absorption_coefficient_by_area_m", None),
            "Surface Area (m²)": features.get("total_surface_sqm", None)
        }
        
        # Create detailed compliance reason
        compliance_reason = f"Zone: {zone}, Period: {period}, Range: {db_range}. "
        compliance_reason += " | ".join(compliance_details)
        
        compliance_result = {
            "status": "✅ Compliant" if is_compliant else "❌ Not Compliant",
            "reason": compliance_reason,
            "metrics": metrics,
            "details": compliance_details
        }

        # Ensure recommendations is always a dict
        if not recommendations:
            recommendations = {}
        elif not isinstance(recommendations, dict):
            recommendations = {"general": recommendations}

        result = {
            "comfort_score": comfort_score,
            "source": f"Inference Tier: {tier}",
            "compliance": compliance_result,
            "recommendations": recommendations,
            "best_materials": {
                "wall_material": best_wall[1] if best_wall else current_wall,
            },
            "best_score": improved_score,
            "improved_score": improved_score
        }

        return result

    except Exception as e:
        print(f"❌ Error in recommend_recompute: {str(e)}")
        raise
