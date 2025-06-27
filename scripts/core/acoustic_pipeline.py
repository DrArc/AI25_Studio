# scripts/acoustic_pipeline.py

from .llm_calls import extract_variables, build_answer
from .sql_calls import query_or_recommend
from utils.format_interpreter import standardize_input

# Import the geometry ML interface
try:
    from .geometry_ml_interface import geometry_ml_interface
    GEOMETRY_ML_AVAILABLE = True
except ImportError:
    print("⚠️ Geometry ML interface not available - using standard pipeline")
    GEOMETRY_ML_AVAILABLE = False

def run_pipeline(user_input: dict, user_question: str = "", geometry_data: dict = None) -> dict:
    """
    Given structured user_input, run the full acoustic pipeline.
    Now supports geometry data for enhanced predictions.
    
    Args:
        user_input: Dictionary with user parameters
        user_question: Original user question
        geometry_data: Optional IFC element data from geometry selection
    """
    user_input = standardize_input(user_input)
    
    # If geometry data is available, enhance the prediction
    if GEOMETRY_ML_AVAILABLE and geometry_data:
        enhanced_result = run_enhanced_pipeline(user_input, geometry_data)
        result = enhanced_result
    else:
        # Fall back to standard pipeline
        result = query_or_recommend(user_input)

    try:
        summary = build_answer(user_question or str(user_input), result)
    except Exception as e:
        summary = f"[LLM Summary Error] {e}"

    return {
        "input": user_input,
        "result": result,
        "summary": summary,
        "enhanced": GEOMETRY_ML_AVAILABLE and geometry_data is not None
    }

def run_enhanced_pipeline(user_input: dict, geometry_data: dict) -> dict:
    """
    Run enhanced pipeline using geometry data for more accurate predictions
    """
    try:
        print("🚀 Running enhanced pipeline with geometry data")
        
        # First, run the standard SQL pipeline to get baseline results
        standard_result = query_or_recommend(user_input)
        print(f"📊 Standard SQL result: {standard_result}")
        
        # Then enhance with geometry data if available
        if geometry_data:
            # Extract data for ML processing
            extracted_data = geometry_ml_interface.extract_element_data(geometry_data)
            print(f"🔍 Extracted geometry data: {extracted_data}")
            
            # Run comfort prediction if we have enough data
            comfort_result = None
            if extracted_data.get("element_type") or extracted_data.get("rt60") or extracted_data.get("spl"):
                comfort_result = geometry_ml_interface.predict_comfort_for_element(extracted_data)
                print(f"🎯 Comfort prediction: {comfort_result}")
            
            # Generate recommendations
            recommendations = geometry_ml_interface.get_ml_recommendations(extracted_data)
            print(f"💡 Recommendations: {recommendations}")
            
            # Create enhanced result that combines SQL and ML
            enhanced_result = {
                "sql_result": standard_result,
                "comfort_prediction": comfort_result,
                "recommendations": recommendations,
                "geometry_analysis": extracted_data,
                "enhanced": True
            }
            
            return enhanced_result
        else:
            # No geometry data, return standard result
            return standard_result
        
    except Exception as e:
        print(f"❌ Enhanced pipeline failed: {e}")
        # Fall back to standard pipeline
        return query_or_recommend(user_input)

def run_from_free_text(question: str, geometry_data: dict = None) -> dict:
    """
    Alternative entry: run the pipeline from a free-form LLM question.
    Now supports geometry data for enhanced predictions.
    """
    user_input = extract_variables(question)
    if not user_input:
        return {"error": "Could not extract parameters from question."}
    return run_pipeline(user_input, question, geometry_data)

def run_from_free_text(question, geometry_data=None):
    extracted = extract_variables(question)
    if not extracted:
        return {"error": "Both zone and apartment_type are required"}
    
    return run_pipeline(extracted, question, geometry_data)