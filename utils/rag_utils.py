import numpy as np
import json
import os
import sys
import re

# Add the project root to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.config import *

# Embedding wrapper
def get_embedding(text, model=embedding_model):
    try:
        text = text.replace("\n", " ")
        if mode == "openai":
            response = client.embeddings.create(input=[text], dimensions=768, model=model)
        else:
            response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Embedding Error: {str(e)}")
        raise e

# Compute cosine similarity
def similarity(v1, v2):
    return np.dot(v1, v2)

# Load vectorized JSON
def load_embeddings(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading embeddings from {filepath}: {str(e)}")
        raise e

# Get top-N similar entries
def get_vectors(query_vector, index_lib, n_results):
    scored = []
    for item in index_lib:
        score = similarity(query_vector, item["vector"])
        scored.append({
            "row_index": item.get("row_index", "unknown"),
            "content": item["content"],
            "score": score
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:n_results]

# RAG-style chat completion
def rag_answer(question, prompt, model=completion_model):
    try:
        print(f"[DEBUG] RAG: Sending prompt to LLM (length: {len(prompt)} chars)")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.1,
            max_tokens=4000,  # Increase token limit for complete responses
        )
        answer = completion.choices[0].message.content
        print(f"[DEBUG] RAG: Received response (length: {len(answer)} chars)")
        print(f"[DEBUG] RAG: Response preview: {answer[:300]}...")
        return answer
    except Exception as e:
        print(f"[DEBUG] RAG: Error in rag_answer: {e}")
        return f"Error generating RAG response: {str(e)}"

# Main RAG call
def sql_rag_call(question, embedding_file, n_results=3):
    print("🔍 Initiating RAG...")

    # Step 1: Embed the user's question
    question_vector = get_embedding(question)

    # Step 2: Load pre-embedded table descriptions
    index_lib = load_embeddings(embedding_file)

    # Step 3: Rank and retrieve top entries
    top_matches = get_vectors(question_vector, index_lib, n_results)
    row_indices = "\n".join([str(match["row_index"]) for match in top_matches])
    descriptions = "\n".join([match["content"] for match in top_matches])

    return row_indices, descriptions

def ecoform_rag_call(question, embedding_file="knowledge/ecoform_dataset_vectors.json", n_results=3):
    try:
        print("🔍 Initiating Ecoform RAG...")

        # Step 1: Embed the user's question
        question_vector = get_embedding(question)

        # Step 2: Load pre-embedded row vectors
        if not os.path.exists(embedding_file):
            return f"Error: Embedding file {embedding_file} not found."
        
        index_lib = load_embeddings(embedding_file)
        
        if not index_lib:
            return "Error: No data found in embedding file."

        # Step 3: Rank and retrieve top entries
        top_matches = get_vectors(question_vector, index_lib, n_results)
        
        if not top_matches:
            return "No relevant information found in the dataset."
        
        # Debug print the retrieved content
        print(f"[DEBUG] RAG: Retrieved {len(top_matches)} matches")
        for i, match in enumerate(top_matches):
            print(f"[DEBUG] RAG: Match {i+1} (score: {match['score']:.3f}): {match['content'][:200]}...")
        
        descriptions = "\n".join([match["content"] for match in top_matches])
        print(f"[DEBUG] RAG: Combined descriptions length: {len(descriptions)} chars")

        # Step 4: Build prompt and get LLM answer
        prompt = (
            "You are an expert in acoustic comfort evaluation. "
            "Given the following relevant cases from the Ecoform dataset, answer the user's question in detail.\n\n"
            f"Relevant cases:\n{descriptions}\n\n"
            "Please provide a complete, detailed answer without truncation."
        )
        answer = rag_answer(question, prompt)
        
        # Validate the response
        if answer and len(answer) > 50:  # Ensure we got a substantial response
            print(f"[DEBUG] RAG: Final answer length: {len(answer)} chars")
            return answer
        else:
            print(f"[DEBUG] RAG: Response too short or empty: {len(answer) if answer else 0} chars")
            return f"Incomplete response received. Please try again. Response length: {len(answer) if answer else 0} chars"
        
    except Exception as e:
        print(f"❌ RAG Error: {str(e)}")
        return f"Error in RAG processing: {str(e)}"
