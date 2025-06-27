import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, Request
from scripts.sql_calls import query_or_recommend
from scripts.llm_calls import extract_variables, build_answer
from scripts.recommend_recompute import run_acoustic_prediction
app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    user_input = await request.json()
    # Optionally clean up keys/values for casing here
    result = query_or_recommend(user_input)
    return {"result": result}

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, Request
from scripts.sql_calls import query_or_recommend
from scripts.llm_calls import extract_variables, build_answer

app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    user_input = await request.json()
    # Optionally clean up keys/values for casing here
    result = query_or_recommend(user_input)
    return {"result": result}

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    user_question = data.get("question", "")
    # Try to extract variables (structured input) from the question
    user_input = extract_variables(user_question)
    if user_input:
        result = query_or_recommend(user_input)
        answer = build_answer(user_question, result)
        return {"guidance": answer}
    else:
        # fallback: just chat LLM for general Q&A
        from scripts.llm_acoustic_query_handler import handle_llm_query
        answer = handle_llm_query(user_question)
        return {"guidance": answer}
