import sys
import os
import json
import pandas as pd

# Add project root for config access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.config import *  # uses embedding_model, mode, client

# File paths
input_file = "sql/Ecoform_Dataset_v1.csv"
output_file = "knowledge/ecoform_dataset_vectors.json"

# Load CSV
df = pd.read_csv(input_file)

# Optionally: select/rename columns for clarity
fields = [
    "apartment_type_string", "zone_string", "element_materials_string",
    "floor_height_m", "l(a)eq_db", "rt60_s", "spl_db", "absorption_coefficient_by_area_m",
    "total_surface_sqm"
]

# Generate embeddings
embeddings = []
for idx, row in df.iterrows():
    # Create a text description for the row
    content = (
        f"Apartment type: {row.get('apartment_type_string', '')}, "
        f"Zone: {row.get('zone_string', '')}, "
        f"Materials: {row.get('element_materials_string', '')}, "
        f"Floor height: {row.get('floor_height_m', '')}m, "
        f"LAeq: {row.get('l(a)eq_db', '')} dB, "
        f"RT60: {row.get('rt60_s', '')} s, "
        f"SPL: {row.get('spl_db', '')} dB, "
        f"Absorption coefficient: {row.get('absorption_coefficient_by_area_m', '')}, "
        f"Surface area: {row.get('total_surface_sqm', '')} m²"
    )

    print(f"🔗 Embedding row {idx+1}/{len(df)}...")
    vector = client.embeddings.create(input=[content], model=embedding_model).data[0].embedding

    embeddings.append({
        "row_index": int(idx),
        "content": content,
        "vector": vector
    })

# Save JSON
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(embeddings, f, indent=2, ensure_ascii=False)

print(f"✅ Ecoform dataset vectors saved to: {output_file}")
