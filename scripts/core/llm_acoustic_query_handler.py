import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
from server.config import client, embedding_model, completion_model

# Load vectorized guidance
with open("knowledge/compliance_guidance_vectors.json", "r", encoding="utf-8") as f:
    vector_data = json.load(f)

vectors = np.array([entry["vector"] for entry in vector_data])
texts = [entry["content"] for entry in vector_data]

# Embed the user query
def embed_query(text: str):
    response = client.embeddings.create(input=[text], model=embedding_model)
    return response.data[0].embedding

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_relevant_guidance(user_query: str):
    q_vec = embed_query(user_query)
    sims = [cosine_similarity(q_vec, v) for v in vectors]
    top_idx = int(np.argmax(sims))
    return texts[top_idx]

def explain_guidance(user_query: str, guidance_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an acoustic design expert explaining compliance guidance for buildings."
        },
        {
            "role": "user",
            "content": f"My question: {user_query}\n\nMatched guidance: {guidance_text}"
        }
    ]
    response = client.chat.completions.create(
        model=completion_model,
        messages=messages
    )
    return response.choices[0].message.content.strip()

def handle_llm_query(user_query: str):
    guidance = get_relevant_guidance(user_query)
    return explain_guidance(user_query, guidance)