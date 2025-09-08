# agent_utils.py
import pandas as pd
import numpy as np
import time
import faiss
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# --- Load Embeddings Model and OpenAI Client ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Build FAISS Index ---
def build_faiss_index(terms: list) -> tuple:
    embeddings = embedding_model.encode(terms, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, terms, embeddings

# --- Embed and Retrieve ---
def retrieve_top_k(med_name: str, index, terms, embeddings, k=1) -> str:
    med_emb = embedding_model.encode([med_name])
    D, I = index.search(med_emb, k)
    return terms[I[0][0]] if I[0][0] < len(terms) else ""

# --- Prompt Construction ---
def build_agent_prompt(med: str, info: str, problems: list, examples: list = []) -> str:
    few_shot = "\n\n".join(examples)
    return (
        "You are a clinical decision support assistant.\n"
        "Use the medication description and patient problem list to match treatments.\n"
        f"Examples:\n{few_shot}\n"
        "---\n"
        f"Medication: {med}\n"
        f"Description: {info}\n"
        f"Patient Problems: {problems}\n"
        "Which problem(s) does this medication treat from the list above?"
    )

# --- Call OpenAI with Retry Strategy ---
def query_with_retry(prompt: str, model="gpt-4") -> str:
    temps = [0.0, 0.3, 0.7]
    for temp in temps:
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a clinical decision support assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Retrying due to error: {e}")
            time.sleep(1)
    return "Error: All retries failed"

# --- Match Output to Problems ---
def fuzzy_match_problems(response: str, problems: list, threshold=80) -> list:
    matched = []
    for p in problems:
        for token in response.split(","):
            if fuzz.partial_ratio(p.lower(), token.strip().lower()) >= threshold:
                matched.append(p)
                break
    return matched
