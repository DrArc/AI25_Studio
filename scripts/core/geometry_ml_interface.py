"""
Geometry-ML Interface Module
Connects IFC geometry data from InfoDock to ML pipeline for acoustic predictions
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import joblib
import os
import sys

# Add project root to allow importing from utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.infer_from_inputs import infer_features

class GeometryMLInterface:
    """
    Interface between IFC geometry data and ML pipeline
    """
    
    def __init__(self):
        """Initialize the interface with ML models"""
        self.comfort_model = None
        self.acoustic_model = None
        self.load_models()
        
    def load_models(self):
        """Load trained ML models"""
        try:
            # Load comfort prediction model - use the specific model requested
            model_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "model", "ecoform_xgb_comfort_model1.pkl"),
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "model", "ecoform_xgb_comfort_model1.pkl"),
                "model/ecoform_xgb_comfort_model1.pkl"
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    self.comfort_model = joblib.load(model_path)
                    print(f"✅ Comfort model loaded successfully from: {model_path}")
                    break
            else:
                print("⚠️ Comfort model not found. Tried paths:")
                for path in model_paths:
                    print(f"   - {path}")
                
            # Use the same model for acoustic analysis (since they're the same type)
            self.acoustic_model = self.comfort_model
            if self.acoustic_model:
                print(f"✅ Acoustic model set to same model: ecoform_xgb_comfort_model1.pkl")
            else:
                print("⚠️ Acoustic model not available")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def extract_element_data(self, ifc_element_data: Dict) -> Dict[str, Any]:
        """
        Extract and structure data from IFC element for ML processing
        
        Args:
            ifc_element_data: Dictionary containing IFC element information from InfoDock
            
        Returns:
            Structured data dictionary for ML models
        """
        extracted_data = {
            "element_type": None,
            "apartment_type": None,
            "floor_level": None,
            "area": None,
            "rt60": None,
            "spl": None,
            "location": {},
            "materials": [],
            "accessibility": {},
            "additional_properties": {}
        }
        
        try:
            print(f"🔍 Extracting data from: {ifc_element_data}")
            
            # Basic element information
            if "type" in ifc_element_data:
                extracted_data["element_type"] = ifc_element_data["type"]
                print(f"📋 Element type: {ifc_element_data['type']}")
            
            if "name" in ifc_element_data:
                name = ifc_element_data["name"]
                print(f"📋 Element name: {name}")
                # Parse apartment info from name
                apartment_info = self.parse_apartment_info(name)
                if apartment_info:
                    extracted_data["apartment_type"] = apartment_info.get("type")
                    extracted_data["floor_level"] = apartment_info.get("floor")
                    print(f"🏢 Apartment info: {apartment_info}")
            
            # Extract acoustic properties from multiple sources
            rt60_found = False
            spl_found = False
            
            # Try acoustic_properties first
            if "acoustic_properties" in ifc_element_data:
                acoustic = ifc_element_data["acoustic_properties"]
                if acoustic.get("RT60"):
                    extracted_data["rt60"] = acoustic.get("RT60")
                    rt60_found = True
                if acoustic.get("SPL"):
                    extracted_data["spl"] = acoustic.get("SPL")
                    spl_found = True
            
            # Try other_properties for acoustic data
            if "other_properties" in ifc_element_data:
                other_props = ifc_element_data["other_properties"]
                for key, value in other_props.items():
                    if "RT60" in key and not rt60_found:
                        extracted_data["rt60"] = value
                        rt60_found = True
                    elif "SPL" in key and not spl_found:
                        extracted_data["spl"] = value
                        spl_found = True
            
            # Extract area information
            if "dimensions" in ifc_element_data:
                dimensions = ifc_element_data["dimensions"]
                if "Gross Planned Area" in dimensions:
                    extracted_data["area"] = dimensions["Gross Planned Area"]
                elif "Area" in dimensions:
                    extracted_data["area"] = dimensions["Area"]
            
            # Extract location
            if "location" in ifc_element_data:
                extracted_data["location"] = ifc_element_data["location"]
            
            # Extract materials
            if "materials" in ifc_element_data:
                extracted_data["materials"] = ifc_element_data["materials"]
            
            # Extract accessibility
            if "accessibility" in ifc_element_data:
                extracted_data["accessibility"] = ifc_element_data["accessibility"]
            
            # Extract additional properties
            if "other_properties" in ifc_element_data:
                extracted_data["additional_properties"] = ifc_element_data["other_properties"]
            
            print(f"✅ Extracted data: {extracted_data}")
                
        except Exception as e:
            print(f"❌ Error extracting element data: {e}")
            
        return extracted_data
    
    def parse_apartment_info(self, space_name: str) -> Optional[Dict]:
        """Parse apartment information from space name"""
        try:
            if not isinstance(space_name, str):
                return None
            
            # Example: "LVL2_3B_36" -> floor=2, type="3B", room=36
            parts = space_name.split("_")
            if len(parts) >= 2:
                floor_part = parts[0]
                type_part = parts[1]
                
                # Extract floor level
                floor = None
                if floor_part.startswith("LVL"):
                    try:
                        floor = int(floor_part.replace("LVL", ""))
                    except:
                        pass
                
                # Extract apartment type
                apt_type = None
                if type_part in ["1B", "2B", "3B"]:
                    apt_type = type_part
                
                # Extract room number
                room = None
                if len(parts) >= 3:
                    try:
                        room = int(parts[2])
                    except:
                        room = parts[2]
                
                return {
                    "floor": floor,
                    "type": apt_type,
                    "room": room
                }
            
            return None
        except Exception as e:
            print(f"Error parsing apartment info: {e}")
            return None
    
    def predict_comfort_for_element(self, element_data: Dict) -> Dict[str, Any]:
        """
        Predict acoustic comfort for a specific IFC element
        
        Args:
            element_data: Extracted element data from InfoDock
            
        Returns:
            Prediction results including comfort score and confidence
        """
        if not self.comfort_model:
            return {"error": "Comfort model not loaded"}
        
        try:
            # Prepare features for ML model
            features = self.prepare_features_for_comfort_model(element_data)
            
            if not features:
                return {"error": "Could not prepare features for prediction"}
            
            # Make prediction
            X = pd.DataFrame([features])
            comfort_score = self.comfort_model.predict(X)[0]
            
            # Calculate confidence (you can implement this based on your model)
            confidence = self.calculate_prediction_confidence(features)
            
            return {
                "comfort_score": round(comfort_score, 3),
                "confidence": confidence,
                "features_used": features,
                "element_type": element_data.get("element_type"),
                "apartment_type": element_data.get("apartment_type"),
                "floor_level": element_data.get("floor_level")
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def prepare_features_for_comfort_model(self, element_data: Dict) -> Optional[Dict]:
        """
        Prepare features for the comfort prediction model
        """
        try:
            print(f"🔍 Preparing features for element: {element_data.get('element_type', 'Unknown')}")
            
            # Use the existing infer_features function
            apartment_type = element_data.get("apartment_type", "1Bed")
            zone = "HD-Urban-V0"  # Default zone, could be made configurable
            element = element_data.get("element_type", "Wall")
            floor = element_data.get("floor_level", 1)
            
            # Get wall and window materials from element data
            materials = element_data.get("materials", [])
            wall_material = "Concrete"  # Default
            window_material = "Double Glazing"  # Default
            
            # Try to extract materials from the materials list
            for material in materials:
                if "wall" in material.lower():
                    wall_material = material
                elif "window" in material.lower() or "glass" in material.lower():
                    window_material = material
            
            print(f"🏗️ Using materials: Wall={wall_material}, Window={window_material}")
            
            # Infer features using existing function
            features, tier = infer_features(
                apartment_type=apartment_type,
                zone=zone,
                element=element,
                wall_material=wall_material,
                window_material=window_material
            )
            
            print(f"📊 Inferred features (tier: {tier}): {features}")
            
            # Override height based on floor level
            if floor:
                features["floor_height_m"] = floor * 3.0
                print(f"🏢 Overriding height with floor level: {floor} -> {features['floor_height_m']}m")
            
            # Override with actual measured values if available
            if element_data.get("rt60"):
                features["rt60_s"] = element_data["rt60"]
                print(f"🎵 Using actual RT60: {element_data['rt60']}s")
            
            if element_data.get("spl"):
                features["spl_db"] = element_data["spl"]
                print(f"🎵 Using actual SPL: {element_data['spl']}dBA")
            
            if element_data.get("area"):
                features["total_surface_sqm"] = element_data["area"]
                print(f"📏 Using actual area: {element_data['area']}m²")
            
            # Ensure all required features are present
            required_features = [
                'zone_string', 'apartment_type_string', 'floor_height_m', 'laeq_db',
                'total_surface_sqm', 'element_materials_string', 'absorption_coefficient_by_area_m',
                'rt60_s', 'n._of_sound_sources_int', 'average_sound_source_distance_m',
                'spl_db', 'barrier_distance_m', 'barrier_height_m'
            ]
            
            for feature in required_features:
                if feature not in features or features[feature] is None:
                    features[feature] = 0.0
            
            print(f"✅ Final features prepared: {features}")
            return features
            
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_prediction_confidence(self, features: Dict) -> float:
        """
        Calculate confidence score for the prediction
        This is a simplified implementation - you can enhance it based on your needs
        """
        try:
            # Simple confidence calculation based on data completeness
            required_features = ["laeq_db", "spl_db", "rt60_s", "floor_height_m"]
            available_features = sum(1 for f in required_features if features.get(f) is not None and features.get(f) != 0.0)
            confidence = available_features / len(required_features)
            
            # Boost confidence if we have measured acoustic data
            if features.get("rt60_s") and features.get("rt60_s") > 0:
                confidence += 0.2
            if features.get("spl_db") and features.get("spl_db") > 0:
                confidence += 0.2
            
            # Boost confidence if we have area data
            if features.get("total_surface_sqm") and features.get("total_surface_sqm") > 0:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            print(f"❌ Error calculating confidence: {e}")
            return 0.5
    
    def analyze_space_acoustics(self, space_data: Dict) -> Dict[str, Any]:
        """
        Analyze acoustic properties of an entire space/apartment
        
        Args:
            space_data: Space information from InfoDock
            
        Returns:
            Comprehensive acoustic analysis
        """
        try:
            analysis = {
                "space_name": space_data.get("name"),
                "apartment_type": space_data.get("apartment_type"),
                "floor_level": space_data.get("floor_level"),
                "acoustic_metrics": {},
                "comfort_assessment": {},
                "recommendations": []
            }
            
            # Extract acoustic metrics
            if "acoustic_properties" in space_data:
                acoustic = space_data["acoustic_properties"]
                analysis["acoustic_metrics"] = {
                    "rt60": acoustic.get("RT60"),
                    "spl": acoustic.get("SPL"),
                    "area": space_data.get("dimensions", {}).get("Gross Planned Area")
                }
            
            # Assess comfort based on metrics
            if analysis["acoustic_metrics"].get("rt60"):
                rt60 = analysis["acoustic_metrics"]["rt60"]
                if rt60 < 0.3:
                    comfort_level = "Excellent"
                    analysis["recommendations"].append("Acoustic environment is optimal for speech clarity")
                elif rt60 < 0.5:
                    comfort_level = "Good"
                    analysis["recommendations"].append("Acoustic environment is suitable for most activities")
                elif rt60 < 0.8:
                    comfort_level = "Fair"
                    analysis["recommendations"].append("Consider acoustic treatments to improve speech clarity")
                else:
                    comfort_level = "Poor"
                    analysis["recommendations"].append("Acoustic treatments strongly recommended")
                
                analysis["comfort_assessment"]["rt60_comfort"] = comfort_level
            
            if analysis["acoustic_metrics"].get("spl"):
                spl = analysis["acoustic_metrics"]["spl"]
                if spl < 40:
                    noise_level = "Very Quiet"
                elif spl < 50:
                    noise_level = "Quiet"
                elif spl < 60:
                    noise_level = "Moderate"
                elif spl < 70:
                    noise_level = "Loud"
                else:
                    noise_level = "Very Loud"
                    analysis["recommendations"].append("Noise reduction measures recommended")
                
                analysis["comfort_assessment"]["noise_level"] = noise_level
            
            return analysis
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def get_ml_recommendations(self, element_data: Dict) -> List[str]:
        """
        Get ML-based recommendations for improving acoustic comfort
        
        Args:
            element_data: Element data from InfoDock
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            # Analyze RT60
            rt60 = element_data.get("rt60")
            if rt60:
                if rt60 > 0.8:
                    recommendations.append("Consider adding acoustic panels to reduce reverberation time")
                elif rt60 < 0.2:
                    recommendations.append("Space may be too acoustically dead - consider reflective surfaces")
            
            # Analyze SPL
            spl = element_data.get("spl")
            if spl:
                if spl > 60:
                    recommendations.append("Noise levels are high - consider sound insulation improvements")
                elif spl < 30:
                    recommendations.append("Space is very quiet - suitable for concentration activities")
            
            # Analyze area
            area = element_data.get("area")
            if area:
                if area < 10:
                    recommendations.append("Small space - consider furniture placement for optimal acoustics")
                elif area > 50:
                    recommendations.append("Large space - may benefit from acoustic zoning")
            
            # Material-based recommendations
            materials = element_data.get("materials", [])
            if "concrete" in str(materials).lower():
                recommendations.append("Concrete surfaces can be reflective - consider acoustic treatments")
            if "glass" in str(materials).lower():
                recommendations.append("Glass surfaces may need acoustic treatment for better sound control")
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to data processing error"]

# Global instance for easy access
geometry_ml_interface = GeometryMLInterface() 