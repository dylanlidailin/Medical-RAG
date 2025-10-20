# agent_utils_fhir.py
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from fhirclient import client
from fhirclient.models import medication, condition

load_dotenv()

# --- Load Embeddings Model and OpenAI Client ---
embedding_model = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- FHIR Client Setup ---
fhir_settings = {
    'app_id': 'my_medical_app',
    'api_base': 'https://hapi.fhir.org/baseR4'  # Public FHIR server
}
fhir_client = client.FHIRClient(settings=fhir_settings)

# --- Retrieve FHIR Medication by name ---
# agent_utils.py

# ... (other code is the same)

# --- Retrieve FHIR Medication by name ---
def get_fhir_medication(med_name: str) -> dict:
    """
    Searches the FHIR server for a medication by its name using the 'code:text' parameter.
    """
    try:
        # CORRECTED: Use 'code:text' which is the standard parameter for searching the
        # display name of a medication's code.
        search_params = {'code:text': med_name}
        search = medication.Medication.where(struct=search_params).perform_resources(fhir_client.server)

        if search:
            med = search[0]  # Take the first match
            name_text = med.code.text if hasattr(med, 'code') and hasattr(med.code, 'text') else med_name
            code_val = med.code.coding[0].code if hasattr(med, 'code') and med.code.coding else None

            return {
                "name": name_text,
                "code": code_val,
                "indication": name_text # Often the 'text' field also serves as the indication
            }
    except Exception as e:
        # This will catch HTTP errors and other issues, preventing a crash.
        print(f"FHIR API Error for '{med_name}': {e}")

    # Fallback if search fails or returns no results
    return {"name": med_name, "code": None, "indication": ""}


# ... (rest of the file is the same)

# --- Retrieve FHIR Conditions ---
def get_fhir_conditions(conditions: list) -> list:
    standardized = []
    for cond in conditions:
        search = condition.Condition.where(struct={'code': cond}).perform_resources(fhir_client.server)
        if search:
            standardized.append(search[0].code.text if hasattr(search[0].code, 'text') else cond)
        else:
            standardized.append(cond)
    return standardized

# --- Build FAISS Index ---
def build_faiss_index(terms: list) -> tuple:
    embeddings = embedding_model.encode(terms, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, terms, embeddings

# --- Embed and Retrieve ---
def retrieve_top_k(med_name: str, index, terms, embeddings, k=1) -> str:
    if not med_name:
        return ""
    med_emb = embedding_model.encode([med_name])
    D, I = index.search(med_emb, k)
    return terms[I[0][0]] if I[0][0] < len(terms) else ""

# --- Prompt Construction ---
def build_agent_prompt(med_name: str, med_info: str, problems: list) -> str:
    problems_standardized = get_fhir_conditions(problems)
    problem_list_str = ", ".join(f'"{p}"' for p in problems_standardized)
    
    return (
        "You are a clinical decision support assistant. Your task is to identify which of the patient's medical "
        f"problems from the provided list are likely treated by the given medication. The list is: [{problem_list_str}]."
        "\n\n"
        f"Medication: {med_name}\n"
        f"Description: {med_info}\n\n"
        "Based on the description, analyze the patient's problems and return a JSON object with a single key, "
        "'treated_problems', which contains a list of strings of the matching problems from the list. "
        "If none of the problems are treated by the medication, return an empty list."
        "\n\n"
        "Example Response for a match: {\"treated_problems\": [\"Hypertension\", \"Angina\"]}"
        "Example Response for no match: {\"treated_problems\": []}"
    )

# --- Call OpenAI with Retry Strategy ---
def query_with_retry(prompt: str, model="gpt-4-1106-preview") -> list:
    try:
        response = openai_client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a clinical decision support assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        output = json.loads(response.choices[0].message.content)
        return output.get("treated_problems", [])
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# --- Fuzzy Match Output ---
def fuzzy_match_problems(response_problems: list, patient_problems: list, threshold=80) -> list:
    """
    Compares problems returned by the AI against the patient's problem list.
    'response_problems' is a list of strings from the AI.
    'patient_problems' is the original list of strings from the patient's record.
    """
    matched = []
    # Iterate through the original list of problems from the patient's record
    for p_patient in patient_problems:
        # Iterate through the list of problems returned by the AI
        for p_response in response_problems:
            # Check if the AI's problem fuzzily matches the patient's problem
            if fuzz.partial_ratio(p_patient.lower(), p_response.strip().lower()) >= threshold:
                matched.append(p_patient)
                # Once a match is found for this patient problem, move to the next one
                break
    return matched