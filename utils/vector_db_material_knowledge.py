import sys
import os
import json

# Add project root for config access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.config import *  # uses embedding_model, mode, client

# File paths
input_file = "knowledge/material_acoustic_knowledge.json"
output_file = "knowledge/material_acoustic_knowledge_vectors.json"

# Load the JSON list of materials
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Generate embeddings for each material entry
embeddings = []
for i, item in enumerate(data):
    name = item.get("material", "Unnamed")
    category = item.get("category", "Unknown")
    stl = item.get("STL_dB", "N/A")
    absorption = item.get("Absorption_Coefficient_500Hz", "N/A")
    scattering = item.get("Scattering_Coefficient", "N/A")
    use = item.get("Typical_Use", "")

    content = (
        f"{name} ({category}): STL {stl} dB, "
        f"Absorption Coeff. @500Hz {absorption}, "
        f"Scattering {scattering}. Use: {use}"
    )

    print(f"🔗 Embedding {name} ({i + 1}/{len(data)})...")
    vector = client.embeddings.create(input=[content], model=embedding_model).data[0].embedding
    embeddings.append({
        "name": name,
        "category": category,
        "content": content,
        "vector": vector
    })

# Save to JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(embeddings, f, indent=2, ensure_ascii=False)

print(f"✅ Vector embeddings saved to: {output_file}")
