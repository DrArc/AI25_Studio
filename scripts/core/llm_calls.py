# llm_calls.py

import sys
import os
import re
import ast  # safer than eval

# ✅ Path hack to ensure imports work inside Cursor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.config import client, completion_model

# 🔹 Extract structured variables from free-form question
def extract_variables(user_question: str) -> dict:
    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": """
You are an assistant for acoustic comfort evaluation.

Your job is to extract structured inputs from user questions.
Extract ONLY these fields if mentioned (case sensitive keys below):

- apartment_type  (e.g. "1Bed", "2Bed")
- zone            (e.g. "Urban", "HD-Urban-V1")
- element_materials  (provide full string if available)
- floor_level     (numeric integer)
- activity        (Living, Sleeping, Working, Healing, Co-working)

Return ONLY a valid Python dictionary.
Omit fields if not mentioned.
Return no explanations.
"""
            },
            {
                "role": "user",
                "content": f"User Question: {user_question}"
            }
        ]
    )

    try:
        content = response.choices[0].message.content.strip()
        # ✅ Safer parsing instead of eval
        variables = ast.literal_eval(content)
        return variables
    except Exception as e:
        print("⚠️ Extraction failed:", e)
        return {}

# 🔹 Summarize acoustic score + compliance + recommendations
def build_answer(user_question: str, result: dict) -> str:
    score = result.get("comfort_score")
    source = result.get("source", "N/A")
    compliance = result.get("compliance", {})
    recommendations = result.get("recommendations", {})
    improved_score = result.get("improved_score", None)

    best_materials = result.get("best_materials", {})
    best_score = result.get("best_score", None)

    summary_prompt = f"""
User Question:
{user_question}

📊 Evaluation Summary:
- Comfort Score: {score}
- Source: {source}
- Compliance: {compliance.get("status")} — {compliance.get("reason")}

🛠 Recommendations:
{recommendations if recommendations else "None needed"}

💡 Material Upgrade Suggestions:
{best_materials if best_materials else "No upgrades suggested"}
Improved Score: {round(best_score, 3) if best_score else "N/A"}
"""

    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": """
You summarize acoustic comfort evaluations for architects and sustainability consultants.

Instructions:
- Do not repeat sentences.
- Clearly state compliance.
- If material upgrades are provided, summarize them usefully.
- Use bullets for clarity if needed.
- If compliant, avoid unnecessary suggestions.
"""
            },
            {
                "role": "user",
                "content": summary_prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()
