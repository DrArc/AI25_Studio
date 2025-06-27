import sys
import os
import json

# Add project root for config access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.config import *  # expects: client, embedding_model

# Paths
input_file = "knowledge/table_descriptions.json"
output_file = "knowledge/table_descriptions_vectors.json"

# Load JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Vectorize
embeddings = []
for i, entry in enumerate(data):
    name = entry["name"]
    content = entry["content"]

    print(f"🔗 Embedding table: {name} ({i+1}/{len(data)})")
    vector = client.embeddings.create(input=[content], model=embedding_model).data[0].embedding

    embeddings.append({
        "name": name,
        "content": content,
        "vector": vector
    })

# Save output
with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(embeddings, f_out, indent=2, ensure_ascii=False)

print(f"✅ Vector file saved to: {output_file}")
