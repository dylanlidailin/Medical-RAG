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
import json

load_dotenv()

# --- Load Embeddings Model and OpenAI Client ---
embedding_model = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
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
# In agent_utils.py

def build_agent_prompt(med: str, info: str, problems: list) -> str:
    # Convert the list of problems into a formatted string for the prompt
    problem_list_str = ", ".join(f'"{p}"' for p in problems)

    return (
        "You are a clinical decision support assistant. Your task is to identify which of the patient's medical "
        f"problems from the provided list are likely treated by the given medication. The list is: [{problem_list_str}]."
        "\n\n"
        f"Medication: {med}\n"
        f"Description: {info}\n\n"
        "Based on the description, analyze the patient's problems and return a JSON object with a single key, "
        "'treated_problems', which contains a list of strings of the matching problems from the list. "
        "If none of the problems are treated by the medication, return an empty list."
        "\n\n"
        "Example Response for a match: {\"treated_problems\": [\"Hypertension\", \"Angina\"]}"
        "Example Response for no match: {\"treated_problems\": []}"
    )

# --- Call OpenAI with Retry Strategy ---
def query_with_retry(prompt: str, model="gpt-4-1106-preview") -> list:
    # Use a model that is optimized for JSON output, like the GPT-4 Turbo preview
    try:
        response = openai_client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"}, # Enable JSON mode
            messages=[
                {"role": "system", "content": "You are a clinical decision support assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0 # Low temperature for deterministic, structured output
        )
        # Parse the JSON string from the response
        output = json.loads(response.choices[0].message.content)
        return output.get("treated_problems", []) # Safely get the list
    except Exception as e:
        print(f"An error occurred: {e}")
        return [] # Return an empty list on failure

# --- Match Output to Problems ---
def fuzzy_match_problems(response: str, problems: list, threshold=80) -> list:
    matched = []
    for p in problems:
        for token in response.split(","):
            if fuzz.partial_ratio(p.lower(), token.strip().lower()) >= threshold:
                matched.append(p)
                break
    return matched
