from server.config import client, completion_model

def handle_llm_query(user_question: str) -> str:
    system_prompt = """
You are an expert in architectural acoustics. Provide clear, concise answers to questions about:
- Acoustic comfort
- RT60 and reverberation
- LAeq and noise thresholds
- Material recommendations for acoustic design
- WHO/ISO compliance for various room types
Use layman-friendly language when needed.
"""
    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
    )
    return response.choices[0].message.content.strip()