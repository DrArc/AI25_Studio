import random
from openai import OpenAI
from server.keys import *
import sqlite3

# Mode
mode = "openai"  # "local" or "openai" or "cloudflare"

# API Clients
local_client = OpenAI(base_url="http://localhost:1234/v1//chat/completions", api_key="lm-studio")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
cloudflare_client = OpenAI(
    base_url=f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
    api_key=CLOUDFLARE_API_KEY
)

# Embedding Models
local_embedding_model = "nomic-ai/nomic-embed-text-v1.5-GGUF"
cloudflare_embedding_model = "@cf/baai/bge-base-en-v1.5"
openai_embedding_model = "text-embedding-3-small"

# Completion Models
gpt4o= [{
    "model": "gpt-4o-mini",
    "api_key": OPENAI_API_KEY,
    "cache_seed": random.randint(0, 100000),
}]

llama3 = [{
    "model": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "api_key": "meta-llama-3.1-8b-instruct",
    "api_type": "openai",
    "base_url": "http://127.0.0.1:1234",
    "cache_seed": random.randint(0, 100000),
}]

cloudflare_model = "@hf/nousresearch/hermes-2-pro-mistral-7b"

# Define what models to use according to chosen "mode"
def api_mode(mode):
    if mode == "local":
        client = local_client
        completion_model = llama3[0]['model']
        embedding_model = local_embedding_model
        return client, completion_model, embedding_model

    if mode == "cloudflare":
        client = cloudflare_client
        completion_model = cloudflare_model
        embedding_model = cloudflare_embedding_model
        return client, completion_model, embedding_model

    elif mode == "openai":
        client = openai_client
        completion_model = gpt4o[0]['model']
        embedding_model = openai_embedding_model
        return client, completion_model, embedding_model
    else:
        raise ValueError("Please specify if you want to run local or openai models")

client, completion_model, embedding_model = api_mode(mode)

# === SQL Schema Utils ===
def get_dB_schema(db_path):
    """
    Returns a dictionary of table names and their column names from the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = cursor.fetchall()
        schema[table_name] = [col[1] for col in columns]
    conn.close()
    return schema

def format_dB_context(db_path, schema_dict):
    """
    Formats schema info into a string to pass into an LLM prompt.
    """
    context_str = ""
    for table, columns in schema_dict.items():
        context_str += f"TABLE: {table}\nCOLUMNS: {', '.join(columns)}\n\n"
    return context_str.strip()