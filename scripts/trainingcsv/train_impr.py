import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

# === 1. Load and Clean Data ===
df = pd.read_csv("sql/parsed_data.csv")

def clean_col(col):
    col = col.strip().lower()
    col = re.sub(r'[():]', '', col)
    col = re.sub(r'\s+', '_', col)
    return col

raw_cols = [clean_col(col) for col in df.columns]
deduped_cols = []
seen = {}
for col in raw_cols:
    if col not in seen:
        seen[col] = 1
        deduped_cols.append(col)
    else:
        seen[col] += 1
        deduped_cols.append(f"{col}_{seen[col]}")
df.columns = deduped_cols

# === 2. Identify Target Column ===
target_col_candidates = [col for col in df.columns if "comfort" in col and "index" in col]
assert len(target_col_candidates) == 1, "❌ Could not uniquely identify comfort index column."
target_col = target_col_candidates[0]

y = df[target_col]
X = df.drop(columns=[target_col])

# === 3. Feature Engineering Example: Add Interactions & Ratios ===
# Example: ratio of SPL to total surface, if those columns exist
if 'spl_db' in X.columns and 'total_surface_sqm' in X.columns:
    X['spl_per_surface'] = X['spl_db'] / (X['total_surface_sqm'] + 1e-3)

# Example: interaction between zone and day/night
if 'zone' in X.columns and 'day_night' in X.columns:
    X['zone_day_night'] = X['zone'].astype(str) + "_" + X['day_night'].astype(str)

# You can add more domain-specific features here as needed

# === 4. Detect Feature Types ===
categorical = X.select_dtypes(include=["object"]).columns.tolist()
numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

# === 5. Preprocessing Pipelines ===
num_transformer = Pipeline([
    ("imputer", KNNImputer(n_neighbors=3)),         # Robust imputation
    ("scaler", StandardScaler()),
    # ("poly", PolynomialFeatures(2, include_bias=False)) # Uncomment to add interactions/quadratics
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, numeric),
    ("cat", cat_transformer, categorical)
])

# === 6. Define Model & Hyperparameter Grid ===
xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, verbosity=1)
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [3, 6],
    'regressor__learning_rate': [0.05, 0.1, 0.2]
}

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", xgb)
])

# === 7. GridSearchCV for Hyperparameter Tuning ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print(f"\n✅ Best parameters: {grid.best_params_}")
print(f"Best CV R²: {grid.best_score_:.4f}")

# === 8. Final Evaluation on Test Data ===
y_pred = grid.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Test R²: {r2:.3f}")
print(f"Test MAE: {mae:.3f}")

# === 9. Feature Importance (Top 10) ===
feature_names = []
# Get feature names from the preprocessor
if hasattr(grid.best_estimator_.named_steps["preprocessor"], "get_feature_names_out"):
    feature_names = grid.best_estimator_.named_steps["preprocessor"].get_feature_names_out()
else:
    # Fallback: just use column names
    feature_names = numeric + categorical

importances = grid.best_estimator_.named_steps["regressor"].feature_importances_
top10_idx = np.argsort(importances)[::-1][:10]
print("\nTop 10 Feature Importances:")
for i in top10_idx:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# === 10. Save the Tuned Pipeline ===
joblib.dump(grid.best_estimator_, "model/ecoform_xgb_comfort_model1.pkl")
print("✅ Tuned model saved to model/ecoform_xgb_comfort_model1.pkl")