import pandas as pd 
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# === 1. Load Dataset ===
df = pd.read_csv("sql/Ecoform_Dataset_v1.csv")

# === 2. Clean column names ===
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# === 3. Define Target and Features ===
y = df["comfort_index_(float)"]

categorical = [
    'zone_(string)',
    'apartment_type_(string)',
    'day/night(string)',
    'element:_materials_(string)'
]

numeric = [
    'floor_height_(m)',
    'l(a)eq_(db)',
    'total_surface_(sqm)',
    'absorption_coefficient_by_area_(m)',
    'rt60_(s)',
    'n._of_sound_sources_(int)',
    'average_sound_source_distance_(m)',
    'spl_(db)',
    'barrier_distance_(m)',
    'barrier_height_(m)',
    'spl_after_barrier_(db)',
    'spl_after_façade_dampening_(db)'
]

# === ✅ Sanity Check ===
expected_columns = categorical + numeric
missing = [col for col in expected_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

# === 4. Build Preprocessing Pipeline ===
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numeric)
])

# === 5. Combine Preprocessing with Model ===
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# === 6. Split Data and Train ===
X = df[categorical + numeric]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# === 7. Save Trained Model ===
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/ecoform_acoustic_comfort_model.pkl")

print("✅ Model trained and saved to model/ecoform_acoustic_comfort_model.pkl")
