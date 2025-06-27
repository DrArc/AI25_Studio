import sys
import os
import joblib
import pandas as pd

# Add project root to allow importing from utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.infer_from_inputs import infer_features

# === 1. Load trained model ===
model = joblib.load("model/ecoform_acoustic_comfort_model.pkl")

# === 2. User Input ===
user_input = {
    "apartment_type": "1Bed",
    "zone": "HD-Urban-V0",
    "element": "Wall",
    "floor": 3,  # floor level input
    "wall_material": "Concrete",
    "window_material": "Double Glazing"
}

# === 3. Infer full feature vector from database ===
features, tier = infer_features(
    apartment_type=user_input["apartment_type"],
    zone=user_input["zone"],
    element=user_input["element"],
    wall_material=user_input.get("wall_material"),
    window_material=user_input.get("window_material")
)

# === 4. Override Height based on floor level (matched to training data) ===
floor = user_input.get("floor")
if floor:
    features["Height"] = floor * 3.0  # Matches your dataset's height values exactly

# === 5. Predict comfort score ===
X = pd.DataFrame([features])
score = model.predict(X)[0]

# === 6. Output Results ===
print("Predicted Comfort Index:", round(score, 3))
print("Match Tier Used:", tier)
print("Acoustic Metrics Used:")
print(f"  Laeq: {features.get('Laeq')}")
print(f"  SPL: {features.get('SPL')}")
print(f"  RT60 (seconds): {features.get('RT60(seconds)')}")
print(f"  RT60 (material ac): {features.get('RT60 (material ac)')}")
print(f"  Absorption Coefficient: {features.get('Absortion_Coefficient')}")
print(f"  Surface Area (m²): {features.get('Surface_Area(m)')}")
print(f"  Facade Dampening Score: {features.get('Facade_Dampening(Score)')}")
print(f"  Final Height (m): {features.get('Height')}")